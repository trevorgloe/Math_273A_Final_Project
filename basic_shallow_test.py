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

x1 = np.random.randn(2000)*0.5+3
x2 = np.random.randn(2000)*0.5+2

x3 = np.random.randn(2000) *0.5 + 4
x4 = np.random.randn(2000) *0.5 + 5
     

# # Creating a Matrix
# X_1 = np.vstack([x1, x2])
# X_2 = np.vstack([x3, x4])
# X = np.hstack([X_1, X_2]).T
X = np.array([np.linspace(0,6,300)]).T
     
# Creating a Vector that contains classes (0, 1)
# y = np.hstack([np.zeros(2000), np.ones(2000)])
y = np.sin(X)

print(X.shape)
print(y.shape)


# plt.scatter(X[:,0], X[:,1], c=y, cmap=cm.coolwarm, edgecolors='w');
plt.scatter(X, y)
plt.title('Dataset')
plt.xlabel('x')
plt.ylabel('y')

# Made a split on our dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
print(y_test)

# converting the datatypes from numpy array into tensors of type float
X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.FloatTensor)
print(X_train)
print(X_test)
print(y_test)
print(y_train)

# checking the shape
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
     
class ShallowNeuralNetwork(nn.Module):
    def __init__(self, input_num, hidden_num, output_num):
        super(ShallowNeuralNetwork, self).__init__()
        self.hidden = nn.Linear(input_num, hidden_num) # hidden layer
        self.output = nn.Linear(hidden_num, output_num) # output layer
        self.sigmoid = nn.Sigmoid() # sigmoid activation function
        self.relu = nn.ReLU() # relu activation function
    
    def forward(self, x):
        x = self.relu(self.hidden(x)) 
        out = self.output(x)
        return out
    
    def predict(self, x):
        # # apply softmax to output 
        # predictions = self.sigmoid(self.forward(x))
        # result = []
        # # pick the class with the maximum weight
        # for current_value in predictions:
        #     if current_value[0] > current_value[1]:
        #         result.append(0)
        #     else:
        #         result.append(1)
        result = self.sigmoid(self.forward(x))
        return result

input_num = 1
hidden_num = 80
output_num = 1 # The output should be the same as the number of classes

model = ShallowNeuralNetwork(input_num, hidden_num, output_num)
model.to(device)
print(model)
# print(model(X_train))

# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

# transfers our tensor from CPU to GPU 1 if CUDA is available
if torch.cuda.is_available():
    X_train = Variable(X_train).cuda()
    y_train = Variable(y_train).cuda()
    X_test = Variable(X_test).cuda()
    y_test = Variable(y_test).cuda()


num_epochs = 50000 # num of epochs
# print(X_train.shape)
# print(model(X_train))

for epoch in range(num_epochs):
    # forward propagation
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    # back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch, num_epochs, loss.item()))
print('\nTraining Complete')
     
# model_prediction = model.predict(X_test)
model_prediction = model(X_test).cpu()
# print(model_prediction)

X_test = X_test.cpu().numpy() # We are moving our tensors to cpu now
y_test = y_test.cpu().numpy()
# model_prediction = np.array(model_prediction)
model_prediction = model_prediction.detach().numpy()
# print(y_test.dtype)
# print(model_prediction)
     
# print("Accuracy Score on test data ==>> {}%".format(accuracy_score(model_prediction, y_test) * 100))
print("Squared error in test data = {}%".format(np.sum(np.power(y_test-model_prediction,2))))

fig, ax = plt.subplots(2, 1, figsize=(12, 10))

# True Predictions
# ax[0].scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], label='Class 0', cmap=cm.coolwarm)
# ax[0].scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], label='Class 1', cmap=cm.coolwarm)
ax[0].scatter(X_test, y_test)
ax[0].set_title('Actual Predictions')
ax[0].legend()

# Models Predictions
# ax[1].scatter(X_test[model_prediction==0, 0], X_test[model_prediction==0, 1], label='Class 0', cmap=cm.coolwarm)
# ax[1].scatter(X_test[model_prediction==1, 0], X_test[model_prediction==1, 1], label='Class 1', cmap=cm.coolwarm)
ax[1].scatter(X_test, model_prediction)
ax[1].set_title('Our Model Predictions')
ax[1].legend()
# print(y_test)
# print(model_prediction)
     
plt.show()