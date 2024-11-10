### Some functions for testing monomial activations

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# define monomial activation function
class polynomial(nn.Module):
    def __init__(self):
        super(polynomial, self).__init__()

    def forward(self, x):
        # return torch.pow(x,2)
        return x**2

def train(model, x_train, y_train, num_epochs, lr):
    # train the model using SGD in pytorch
    # inputs:
    #   model - pytroch nn.Module inheriting class for the nn model
    #   x_train - pytorch tensor of training data
    #   y_train - pytorch tensor of training outputs for the nn to match
    #   num_epochs - number of epochs used to train

    # uses the mean-squared error loss 
    criterion = torch.nn.MSELoss()
    # uses an SGD optimizer for training
    # optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    print("starting training")
    for epoch in range(num_epochs):
        # forward propagation
        y_pred = model(x_train) # evaluate the current model on the data
        # print(y_pred)
        loss = criterion(y_pred, y_train) # compute the loss
        # print("loss = ")
        # print(loss)
        
        # back propagation
        optimizer.zero_grad() # zero out the gradient to add the new one
        loss.backward() # compute the new gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # impliment a clip on the gradient size. this is somewhat of a heuristic, so it might not be the best way to do this
        optimizer.step() # do a SGD step
        
        if epoch % 200 == 0:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch, num_epochs, loss.item()))
    print('\nTraining Complete')

    # return the model, now trained
    return model


def student_teacher_train(teacher_W, teacher_v, dim_x, dim_y, N, num_epochs, lr):
    # creates and trains a shallow neural network (1 hidden layer) with the student teacher situation
    # creates the training data by creating random samples in a range of (-2,2)^dim_x and then feeding them through a simple teacher model
    # then creates a shallow neural network with monomial activation function and trains it with that data
    # inputs:
    #   teacher_W - matrix of weights for the teacher, should be m x dim_x+1
    #   teacher_v - output weights for the teacher, should be dim_y x m
    #   dim_x/dim_y - dimensions for the input and output data respectively
    #   num_epochs - number of epochs used for training
    #   N - number of data points used for training

    #create training data
    m = teacher_W.shape[0]
    x_train = 4 * (np.random.rand(N, dim_x)-0.5)
    x_train_teach = np.hstack([x_train, np.array([np.ones(N)]).T]).T # add an extra set of 1's so the linear transformation can be affine
    # print(x_train)
    # print(teacher_W)
    teacher_input_layer = teacher_W @ x_train_teach
    teacher_hidden = np.power(teacher_input_layer, 2) # quadratic activation function
    teacher_output_layer = teacher_v @ teacher_hidden # output layer
    teacher_output_layer = teacher_output_layer.T
    
    y_train = torch.from_numpy(teacher_output_layer).type(torch.FloatTensor)
    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)

    # define Neural network
    class ShallowNeuralNetwork(nn.Module):
        def __init__(self, input_num, hidden_num, output_num):
            super(ShallowNeuralNetwork, self).__init__()
            self.hidden = nn.Linear(input_num, hidden_num) # hidden layer
            self.output = nn.Linear(hidden_num, output_num, bias=False) # output layer
            self.quad = polynomial() # polynomial activation function
        
        def forward(self, x):
            x = self.hidden(x) # input linear layer
            x = self.quad(x) # quadratic activation function
            out = self.output(x) # output linear layer
            return out

    input_num = dim_x
    hidden_num = m # for now use the same width for the hidden layer at the teacher
    output_num = dim_y # The output should be the same as the number of classes
    print("creating an nn with {} hidden layers".format(m))

    model = ShallowNeuralNetwork(input_num, hidden_num, output_num)
    # model.to(device)
    print(model)

    model = train(model=model, x_train=x_train, y_train=y_train, num_epochs=num_epochs, lr=lr)
    # print(model.hidden.weight)
    # print(model.hidden.bias)
    # print(model.output.weight)
    return model # return the trained model