# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load dataset
data = pd.read_csv("/Users/emin/Desktop/MSc Data Science/Neural Computing/data_banknote_authentication.txt", header=None)

# Assign column names based on UCI dataset description
data.columns = ["variance", "skewness", "curtosis", "entropy", "class"]

# Display basic statistics
print("Basic Statistics of the Dataset:")
print(data.describe())

# Check class distribution
print("\nClass Distribution:")
print(data['class'].value_counts())

# Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

# Feature Distributions
data.hist(figsize=(10, 8), bins=30)
plt.suptitle("Feature Distributions")
plt.show()

# Splitting features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Train-Test Split (80-20, random shuffling)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Activation function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the Sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialise weights and biases randomly
def initialize_weights(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.01  # Input to Hidden layer weights
    b1 = np.zeros((1, hidden_size))  # Hidden layer biases
    W2 = np.random.randn(hidden_size, output_size) * 0.01  # Hidden to Output layer weights
    b2 = np.zeros((1, output_size))  # Output layer biases
    return W1, b1, W2, b2

# Forward propagation function
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1  # Compute hidden layer linear combination
    A1 = sigmoid(Z1)  # Apply activation function
    Z2 = np.dot(A1, W2) + b2  # Compute output layer linear combination
    A2 = sigmoid(Z2)  # Apply activation function
    return Z1, A1, Z2, A2

# Compute Mean Squared Error (MSE) loss
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Backpropagation algorithm for updating weights and biases
def backward_propagation(X, y, Z1, A1, Z2, A2, W1, W2, b1, b2, learning_rate):
    m = y.shape[0]

    # Compute gradients for output layer
    dZ2 = (A2 - y) * sigmoid_derivative(A2)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    # Compute gradients for hidden layer
    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    # Update weights and biases
    W1 = W1 - (learning_rate * dW1)
    b1 = b1 - (learning_rate * db1)
    W2 = W2 - (learning_rate * dW2)
    b2 = b2 - (learning_rate * db2)
    
    return W1, b1, W2, b2

# Training function for MLP
def train_mlp(X_train, y_train, X_test, y_test, hidden_size, learning_rate, epochs):
    input_size = X_train.shape[1]  # Number of input features
    output_size = y_train.shape[1]  # Number of output classes
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)  # Initialise parameters
    
    losses = []  # Track training loss
    test_accuracies = []  # Track test accuracy over epochs
    start_time = time.time()  # Start time measurement
    
    # Training loop
    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward_propagation(X_train, W1, b1, W2, b2)  # Forward pass
        loss = compute_loss(y_train, A2)  # Compute loss
        losses.append(loss)
        W1, b1, W2, b2 = backward_propagation(X_train, y_train, Z1, A1, Z2, A2, W1, W2, b1, b2, learning_rate)  # Backpropagation
        
        if epoch % 50 == 0:  # Evaluate test accuracy every 50 epochs
            _, _, _, A2_test = forward_propagation(X_test, W1, b1, W2, b2)
            predictions = np.argmax(A2_test, axis=1)
            y_true = np.argmax(y_test, axis=1)
            accuracy = np.mean(predictions == y_true)
            test_accuracies.append(accuracy)
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Test Accuracy = {accuracy:.4f}")
    
    training_time = time.time() - start_time  # Compute total training time
    return losses, test_accuracies, training_time

# Define different configurations for hidden layer size and learning rate
hidden_sizes = [10, 50, 100, 150]
learning_rates = [0.1, 0.3, 0.5]

results = {} # Store results

# Train MLP for different configurations
for hidden_size in hidden_sizes:
    for lr in learning_rates:
        print(f"Training with hidden neurons={hidden_size}, learning rate={lr}")
        losses, accuracies, training_time = train_mlp(X_train, y_train, X_test, y_test, hidden_size, lr, 500)
        results[(hidden_size, lr)] = (losses, accuracies, training_time)

# Plot training loss over epochs
for (hidden_size, lr), (losses, accuracies, training_time) in results.items():
    plt.plot(losses, label=f"Hidden={hidden_size}, LR={lr}")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.show()

# Plot test accuracy over epochs
for (hidden_size, lr), (losses, accuracies, training_time) in results.items():
    plt.plot(range(0, 500, 50), accuracies, marker='o', label=f"Hidden={hidden_size}, LR={lr}")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy Over Epochs")
plt.legend()
plt.show()

# Plot training time for different configurations
times = [results[(h, lr)][2] for h in hidden_sizes for lr in learning_rates]
plt.bar([f"H={h},LR={lr}" for h in hidden_sizes for lr in learning_rates], times)
plt.xlabel("Configuration")
plt.ylabel("Training Time (s)")
plt.title("Training Time for Different Configurations")
plt.xticks(rotation=90)
plt.show()

# Print results
print("\nFinal Comparison of Training Runs:\n")
print(f"{'Hidden Neurons':<15} | {'Learning Rate':<15} | {'Final Accuracy':<15} | {'Training Time (s)':<15}")
print("-" * 75)

for (hidden_size, lr), (losses, accuracies, training_time) in results.items():
    final_accuracy = accuracies[-1] * 100  # Convert to percentage
    print(f"{hidden_size:<15.4f} | {lr:<15.4f} | {final_accuracy:<15.4f} | {training_time:<15.4f}")
