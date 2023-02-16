import pandas as pd
import numpy as np

# Load data
data = pd.read_excel('brain_stroke.xlsx')

# Pre-processing data
# Encoding categorical data
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
data['ever_married'] = data['ever_married'].map({'Yes': 1, 'No': 0})
data['work_type'] = data['work_type'].map({'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4})
data['Residence_type'] = data['Residence_type'].map({'Urban': 1, 'Rural': 0})
data['smoking_status'] = data['smoking_status'].map({'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3})

# Drop missing values
data.dropna(inplace=True)

# Split data into training and test sets
train_data = data.sample(frac=0.8, random_state=1)
test_data = data.drop(train_data.index)

# Extract feature and target variables
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Standardize data
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Add bias term
X_train['bias'] = 1
X_test['bias'] = 1

# Define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define loss function
def loss_function(X, y, w):
    y_hat = sigmoid(X.dot(w))
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)).mean()

# Define gradient function
def gradient(X, y, w):
    y_hat = sigmoid(X.dot(w))
    return X.T.dot(y_hat - y) / len(y)

# Initialize weights
w = np.random.randn(X_train.shape[1])

# Train model using gradient descent
learning_rate = 0.01
n_epochs = 1000
for epoch in range(n_epochs):
    grad = gradient(X_train, y_train, w)
    w -= learning_rate * grad
    if epoch % 100 == 0:
        loss = loss_function(X_train, y_train, w)
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Predict target variable for test set
y_pred = sigmoid(X_test.dot(w)) >= 0.5

# Evaluate model
accuracy = (y_pred == y_test).mean()
print(f"Accuracy: {accuracy:.4f}")
