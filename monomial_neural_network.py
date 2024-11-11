import torch
import torch.nn as nn
import torch.nn.functional as F

# define monomial activation function
class Monomial(nn.Module):
    """
    A PyTorch neural network module that represents a monomial function.
    Methods
    -------
    __init__():
        Initializes the Monomial module.
    forward(x, power=2):
        Computes the forward pass of the monomial function.
        Parameters:
            x (torch.Tensor): The input tensor.
            power (int, optional): The exponent to which the input tensor is raised. Default is 2.
        Returns:
            torch.Tensor: The result of raising the input tensor to the specified power.
    """
    
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, power: int = 2):
        return x**power

class MonomialNeuralNetwork(nn.Module):
    """
    A neural network model where each hidden layer applies a monomial activation function.
    Args:
        input_size (int): The number of input features.
        output_size (int): The number of output features.
        layers (list, optional): A list containing the number of neurons in each hidden layer. Defaults to [50, 50].
        power (int, optional): The power to which each element in the hidden layers is raised. Defaults to 2.
    Attributes:
        layers (nn.Sequential): A sequential container of the layers in the network.
    Methods:
        forward(x):
            Defines the computation performed at every call.
            Args:
                x (torch.Tensor): Input tensor.
            Returns:
                torch.Tensor: Output tensor after passing through the network.
    """
    def __init__(
            self, input_size: int, output_size: int, 
            layers: list = [50, 50], power: int = 2
        ):
        """
        Initializes the MonomialNeuralNetwork.

        Args:
            input_size (int): The size of the input layer.
            output_size (int): The size of the output layer.
            layers (list, optional): A list containing the sizes of the hidden layers. Defaults to [50, 50].
            power (int, optional): The power to which the monomial activation function raises its input. Defaults to 2.
        """
        super().__init__()
        layer_list = []
        dim_in = input_size
        for dim_out in layers:
            layer_list.append(nn.Linear(dim_in, dim_out))
            layer_list.append(Monomial(power))
            dim_in = dim_out
        layer_list.append(nn.Linear(dim_in, output_size))
        self.layers = nn.Sequential(*layer_list)
    
    def forward(self, x):
        """
        Perform a forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor to the neural network.

        Returns:
            torch.Tensor: Output tensor after passing through the network layers.
        """
        x = self.layers(x)
        return x

def train(model, x_train, y_train, num_epochs, lr):
    """
    Train the neural network model.

    Args:
        model (nn.Module): The neural network model to train.
        x_train (torch.Tensor): The input training data.
        y_train (torch.Tensor): The output training data.
        num_epochs (int): The number of epochs to train the model.
        lr (float): The learning rate for the optimizer.

    Returns:
        nn.Module: The trained model.
    """
    criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    print("starting training")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(x_train)
        loss = F.mse_loss(output, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch, num_epochs, loss.item()))
    print('\nTraining Complete')
    return model
# def train(model, x_train, y_train, num_epochs, lr):
#     # train the model using SGD in pytorch
#     # inputs:
#     #   model - pytroch nn.Module inheriting class for the nn model
#     #   x_train - pytorch tensor of training data
#     #   y_train - pytorch tensor of training outputs for the nn to match
#     #   num_epochs - number of epochs used to train

#     # uses the mean-squared error loss 
    
#     # uses an SGD optimizer for training
#     # optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)


    
#     for epoch in range(num_epochs):
#         # forward propagation
#         y_pred = model(x_train) # evaluate the current model on the data
#         # print(y_pred)
#         loss = criterion(y_pred, y_train) # compute the loss
#         # print("loss = ")
#         # print(loss)
        
#         # back propagation
#         optimizer.zero_grad() # zero out the gradient to add the new one
#         loss.backward() # compute the new gradient
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # impliment a clip on the gradient size. this is somewhat of a heuristic, so it might not be the best way to do this
#         optimizer.step() # do a SGD step
        
#         if epoch % 200 == 0:
#             print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch, num_epochs, loss.item()))
#     print('\nTraining Complete')

#     # return the model, now trained
#     return model