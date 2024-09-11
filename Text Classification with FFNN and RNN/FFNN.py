import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser

unk = '<UNK>'

# FFNN Model
class FFNN(nn.Module):
    def __init__(self, input_dim, h, output_dim=5):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU()
        self.output_dim = output_dim
        self.W2 = nn.Linear(h, self.output_dim)
        self.softmax = nn.LogSoftmax(dim=1)  # Specify dimension for LogSoftmax
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        hidden_layer = self.activation(self.W1(input_vector))
        output_layer = self.W2(hidden_layer)
        predicted_vector = self.softmax(output_layer)
        return predicted_vector


# Functions to process data
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab


def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index
        index2word[index] = word
    vocab.add(unk)
    return vocab, word2index, index2word


def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data


def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="to fill", help="path to test data")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="mini-batch size")
    parser.add_argument("--optimizer", type=str, default="sgd", help="optimizer type: sgd or adam")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # Fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)

    model = FFNN(input_dim=len(vocab), h=args.hidden_dim).to(device)

    # Select optimizer
    if args.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError("Unsupported optimizer type. Choose either 'sgd' or 'adam'.")

    print(f"========== Training for {args.epochs} epochs ==========")
    best_val_accuracy = 0.0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        correct = 0
        total = 0
        start_time = time.time()
        print(f"Training started for epoch {epoch + 1}")
        random.shuffle(train_data)
        N = len(train_data)
        minibatch_size = args.batch_size

        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_vector = input_vector.to(device)
                gold_label = torch.tensor([gold_label]).to(device)

                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), gold_label)
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()
        print(f"Training completed for epoch {epoch + 1}")
        print(f"Training accuracy for epoch {epoch + 1}: {correct / total}")
        print(f"Training time for this epoch: {time.time() - start_time}")

        # Validation phase
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            start_time = time.time()
            print(f"Validation started for epoch {epoch + 1}")
            N = len(valid_data)
            for minibatch_index in tqdm(range(N // minibatch_size)):
                loss = None
                for example_index in range(minibatch_size):
                    input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                    input_vector = input_vector.to(device)
                    gold_label = torch.tensor([gold_label]).to(device)

                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    correct += int(predicted_label == gold_label)
                    total += 1
            val_accuracy = correct / total
            print(f"Validation completed for epoch {epoch + 1}")
            print(f"Validation accuracy for epoch {epoch + 1}: {val_accuracy}")
            print(f"Validation time for this epoch: {time.time() - start_time}")

            # Save model if validation accuracy improves
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"Model saved at epoch {epoch + 1}")

    print("========== Training finished ==========")
