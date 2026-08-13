# Import Libraries
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Load the dataset
file_path = "/Users/emin/Desktop/MSc Data Science/Neural Computing/Individual Coursework/adult.csv"
df = pd.read_csv(file_path)

# Fix column names
df.columns = df.columns.str.strip()

# Check for the first few rows of the dataset
df.head()

# Checking for missing values in each column
df.isnull().sum()

# Label encode all object-type columns
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le # Saving encoders for potential inverse transformation

# Show statistical summary for selected numerical features
numerical_features = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'income']
summary = df[numerical_features].describe()
print("Statistical Summary of the Dataset:")
print(summary)

# Plot histograms for each numerical feature to see their distributions
plt.figure(figsize=(12, 8))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[feature], bins=30, kde=True, color='red')
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# Define and separate the input features (X) and target variable (Y)
X = df.drop(columns=['income'])
Y = df['income']

# Splitting data into 80% training and 20% testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardise numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Defining the SVM model
svm_model = SVC(kernel='linear', C=1.0) # Building an SVM model with a linear classifier
svm_model.fit(X_train, Y_train) # Training the SVM model
y_pred_svm = svm_model.predict(X_test) # Making predictions on the model using the trained SVM model
print("SVM Accuracy:", accuracy_score(Y_test, y_pred_svm))
print(classification_report(Y_test, y_pred_svm))

# Plot Confusion Matrix for SVM
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(Y_test, y_pred_svm), annot=True, fmt='d', cmap='Reds')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix')
plt.show()

# Prepare data for PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.long)

# Wrap data in PyTorch datasets and loaders
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(seed))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Defining the CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64) # First hidden layer
        self.fc2 = nn.Linear(64, 32) # Second hidden layer
        self.fc3 = nn.Linear(32, num_classes) # Output layer
        self.relu = nn.ReLU() # ReLU activation function
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialising the CNN Model
input_size = X_train.shape[1]
num_classes = len(np.unique(Y)) # There are two classes: <=50K and >50K
cnn_model = SimpleCNN(input_size, num_classes)

# Defining the loss function and optimiser
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(cnn_model.parameters(), lr=0.001)

# Training the CNN for 10 epochs
num_epochs = 10
train_losses = []
test_losses = []
for epoch in range(num_epochs):
    cnn_model.train() # Set the model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimiser.zero_grad() # Reset gradients
        outputs = cnn_model(inputs) # Forward pass
        loss = criterion(outputs, labels) # Compute loss
        loss.backward() # Backpropagation
        optimiser.step() # Update weights
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))
    
    # Evaluate on the test data
    cnn_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader: 
            outputs = cnn_model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    test_losses.append(test_loss / len(test_loader))
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')

# Plot Training and Testing Loss Curves
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('CNN Training & Testing Loss')
plt.legend()
plt.show()

# Make predictions using the trained CNN
y_pred_cnn = []
cnn_model.eval()
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = cnn_model(inputs)
        _, predicted = torch.max(outputs, 1) # Choose the class with the highest score
        y_pred_cnn.extend(predicted.numpy())

# Plot the Confusion Matrix for the CNN
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(Y_test, y_pred_cnn), annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('CNN Confusion Matrix')
plt.show()

# Print the final CNN performance
print("CNN Accuracy:", accuracy_score(Y_test, y_pred_cnn))
print(classification_report(Y_test, y_pred_cnn))