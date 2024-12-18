# Necessary imports
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.cm as cm


device = ("cuda" if torch.cuda.is_available() else "cpu")
     

# # Creating an input vector
X = np.vstack([6 * np.random.rand(1000),6 * np.random.rand(1000)]).T
print(X)
# x1pts, x2pts = np.meshgrid(X[:,0], X[:,1])
     
# Creating a Vector that contains sin(x)
# y = np.sin(x1pts[:,:]*x2pts[:,:])
y = np.sin(X[:,0]* X[:,1])

print(X.shape)
print(y.shape)

# Create a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the points in 3D space
sc = ax.scatter(X[:,0], X[:,1], y, c=y, cmap='viridis')

# Made a split on our dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
# print(y_test)

# converting the datatypes from numpy array into tensors of type float
X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.FloatTensor)
# print(X_train)
# print(X_test)
# print(y_test)
# print(y_train)

# checking the shape
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# define quadratic activation function
class polynomial(nn.Module):
    def __init__(self):
        super(polynomial, self).__init__()

    def forward(self, x):
        # return torch.pow(x,2)
        return x**4
        # return 2*x
        # return torch.sigmoid(x)
    

# define Neural network
class ShallowNeuralNetwork(nn.Module):
    def __init__(self, input_num, hidden_num, output_num):
        super(ShallowNeuralNetwork, self).__init__()
        self.hidden = nn.Linear(input_num, hidden_num) # hidden layer
        self.output = nn.Linear(hidden_num, output_num) # output layer
        self.sigmoid = nn.Sigmoid() # sigmoid activation function
        self.relu = nn.ReLU() # relu activation function
        self.quad = polynomial() # polynomial activation function
        # self.
    
    def forward(self, x):
        # x = self.quad(self.hidden(x))
        # print("inital x")
        # print(x)
        x = self.hidden(x)
        # print("after hidden layer")
        # print(x)
        x = self.quad(x)
        # x = self.relu(x)
        # print("after quad")
        # print(x)
        # x = torch.pow(self.hidden(x),2) 
        # x = self.hidden(x)
        out = self.output(x)
        # print("after out")
        # print(out)
        return out


input_num = 2
hidden_num = 200
output_num = 1 # The output should be the same as the number of classes

model = ShallowNeuralNetwork(input_num, hidden_num, output_num)
model.to(device)
print(model)
# print(model(X_train))

# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

stupid_test_data = torch.tensor([0.1])
# transfers our tensor from CPU to GPU 1 if CUDA is available
if torch.cuda.is_available():
    X_train = Variable(X_train).cuda()
    y_train = Variable(y_train).cuda()
    X_test = Variable(X_test).cuda()
    y_test = Variable(y_test).cuda()
    stupid_test_data = Variable(stupid_test_data).cuda()


num_epochs = 10000 # num of epochs
# print(X_train.shape)
# print(model(X_train))
# print(model(tor0.2))
# print(model(stupid_test_data))
print("starting training")
for epoch in range(num_epochs):
    # forward propagation
    y_pred = model(X_train)
    # print(y_pred)
    loss = criterion(y_pred, y_train)
    # print("loss = ")
    # print(loss)
    
    # back propagation
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if epoch % 200 == 0:
        print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch, num_epochs, loss.item()))
print('\nTraining Complete')
     
# model_prediction = model.predict(X_test)
model_prediction = model(X_train).cpu()
# print(model_prediction)

X_test = X_train.cpu().numpy() # We are moving our tensors to cpu now
y_test = y_train.cpu().numpy()
# model_prediction = np.array(model_prediction)
model_prediction = model_prediction.detach().numpy()
# print(y_test.dtype)
# print(model_prediction)
     
# print("Accuracy Score on test data ==>> {}%".format(accuracy_score(model_prediction, y_test) * 100))
print("Squared error in test data = {}%".format(np.sum(np.power(y_test-model_prediction,2))))
print("Model parameters = ")
print(model.parameters())

fig, ax = plt.subplots(2, 1, figsize=(12, 10))

# # True Predictions
# # ax[0].scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], label='Class 0', cmap=cm.coolwarm)
# # ax[0].scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], label='Class 1', cmap=cm.coolwarm)
# ax[0].scatter(X_test, y_test)
# ax[0].set_title('Actual Predictions')
# ax[0].legend()

# # Models Predictions
# # ax[1].scatter(X_test[model_prediction==0, 0], X_test[model_prediction==0, 1], label='Class 0', cmap=cm.coolwarm)
# # ax[1].scatter(X_test[model_prediction==1, 0], X_test[model_prediction==1, 1], label='Class 1', cmap=cm.coolwarm)
# ax[1].scatter(X_test, model_prediction)
# ax[1].set_title('Our Model Predictions')
# ax[1].legend()
# # print(y_test)
# # print(model_prediction)

# Model output function
model.to("cpu")
# # x_th = torch.linspace(0, 6, 200)
# x_th = np.array([np.linspace(0,6,200)]).T
# x_th = torch.from_numpy(x_th).type(torch.FloatTensor)
# y_th = model(X_test)
# fig2 = plt.figure()
# plt.plot(x_th.detach().numpy(), y_th.detach().numpy(), 'b-')
# plt.xlabel('Inputs')
# plt.ylabel('Model outputs')

# Create a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the points in 3D space
# y_th = model(X_test).detach().numpy()
model_prediction = np.squeeze(model_prediction)
print(model_prediction.shape)
print(X_test[:,0].shape)
print(X_test[:,1].shape)
sc = ax.scatter(X_test[:,0], X_test[:,1], model_prediction, c=model_prediction, cmap='viridis')
plt.title("Model predictions")
     
plt.show()