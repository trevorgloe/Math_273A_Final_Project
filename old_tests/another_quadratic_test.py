import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the quadratic activation function
class QuadraticActivation(nn.Module):
    def __init__(self):
        super(QuadraticActivation, self).__init__()

    def forward(self, x):
        return x**4  # Standard quadratic function

# Define the neural network model
class QuadNN(nn.Module):
    def __init__(self):
        super(QuadNN, self).__init__()
        self.hidden = nn.Linear(1, 50)  # Input layer (1D input) to hidden layer (50 neurons)
        self.output = nn.Linear(50, 1)  # Output layer (1D output)

        self.quad = QuadraticActivation()  # Using the quadratic activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hidden(x)  # Apply hidden layer
        x = self.quad(x)     # Apply quadratic activation
        # x = self.relu(x)
        out = self.output(x) # Output layer
        return out

# Generate data: y = sin(x)
X = np.linspace(-2 * np.pi, 2 * np.pi, 100).reshape(-1, 1)
y = np.sin(X)

# Convert to PyTorch tensors
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32)

# Initialize the model, loss function, and optimizer
model = QuadNN()
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = optim.SGD(model.parameters(), lr=0.000001)

# Training loop
num_epochs = 50000
for epoch in range(num_epochs):
    model.train()

    # Forward pass
    y_pred = model(X_train)

    # Compute the loss
    loss = criterion(y_pred, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item():.6f}')

# Testing the model
model.eval()
with torch.no_grad():
    X_test = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_pred = model(X_test_tensor).detach().numpy()

# Plot the true vs predicted values
plt.plot(X, y, label="True sin(x)", color='blue')
plt.plot(X_test, y_test_pred, label="Predicted by QuadNN", color='red', linestyle='dashed')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Approximating sin(x) using a neural network with quadratic activation")
plt.show()