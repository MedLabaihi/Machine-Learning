import matplotlib.pyplot as plt

# Define the training and validation accuracy for each epoch for the FFNN model
ffnn_training_accuracy = [0.530125, 0.58475, 0.618125, 0.642875, 0.66575]
ffnn_validation_accuracy = [0.02875, 0.03625, 0.1025, 0.06625, 0.04625]

# Define the training and validation accuracy for each epoch for the RNN model
rnn_training_accuracy = [0.560125, 0.60475, 0.638125, 0.662875, 0.68575]
rnn_validation_accuracy = [0.04875, 0.05625, 0.1125, 0.07625, 0.05625]

# Define the epochs
epochs = range(1, 6)

# Create the plot
plt.figure(figsize=(12,6))

# Plot FFNN accuracies
plt.plot(epochs, ffnn_training_accuracy, 'bo-', label='FFNN Training Accuracy')
plt.plot(epochs, ffnn_validation_accuracy, 'ro-', label='FFNN Validation Accuracy')

# Plot RNN accuracies
plt.plot(epochs, rnn_training_accuracy, 'bs--', label='RNN Training Accuracy')
plt.plot(epochs, rnn_validation_accuracy, 'rs--', label='RNN Validation Accuracy')

# Add titles and labels
plt.title('Learning Curve for FFNN and RNN Models')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
