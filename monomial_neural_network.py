import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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

    def __init__(self, power: int = 2):
        super().__init__()
        self.power = power

    def forward(self, x: torch.Tensor):
        return x**self.power


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
        self, input_size: int = 5, output_size: int = 1,
        hidden_layers: list = [50, 50], power: int = 2, weights: list = None
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
        weights_exist = weights is not None  # check if weights are provided
        layer_list = []  # list of layers in the network
        dim_in = input_size

        for index, dim_out in enumerate(hidden_layers):
            linear_layer = nn.Linear(dim_in, dim_out, bias = False)  # create a linear layer with NO bias
            if weights_exist:  # if weights are provided, set the weights of the layer
                with torch.no_grad():
                    linear_layer.weight.copy_(weights[index])
            layer_list.append(linear_layer)  # add the linear layer to the list
            # add the monomial activation function
            layer_list.append(Monomial(power))
            dim_in = dim_out

        output_layer = nn.Linear(dim_in, output_size, bias = False)  # create the output layer with NO bias
        if weights_exist:  # if weights are provided, set the weights of the output layer
            with torch.no_grad():
                output_layer.weight.copy_(weights[-1])
        layer_list.append(output_layer)  # add the output layer

        # create a sequential container of the layers
        self.layers = nn.Sequential(*layer_list)

    def forward(self, x: torch.Tensor):
        """
        Perform a forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor to the neural network.

        Returns:
            torch.Tensor: Output tensor after passing through the network layers.
        """
        x = self.layers(x)
        return x

    def evaluate(self, x):
        """
        Evaluate the neural network model on the input data without affecting the torch.autograd.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the neural network model.
        """
        self.eval()  # set the model to evaluation mode
        with torch.no_grad():
            return self(x)

class MonomialNeuralNetwork_noOutputWeight(nn.Module):
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
        self, input_size: int = 5, output_size: int = 1,
        hidden_layers: list = [50, 50], power: int = 2, weights: list = None
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
        weights_exist = weights is not None  # check if weights are provided
        layer_list = []  # list of layers in the network
        dim_in = input_size

        for index, dim_out in enumerate(hidden_layers):
            linear_layer = nn.Linear(dim_in, dim_out, bias = False)  # create a linear layer with NO bias
            if weights_exist:  # if weights are provided, set the weights of the layer
                with torch.no_grad():
                    linear_layer.weight.copy_(weights[index])
            layer_list.append(linear_layer)  # add the linear layer to the list
            # add the monomial activation function
            layer_list.append(Monomial(power))
            dim_in = dim_out

        # keep the output layer frozen
        output_layer = nn.Linear(dim_in, output_size, bias = False)  # create the output layer with NO bias
        # if weights_exist:  # if weights are provided, set the weights of the output layer
        #     with torch.no_grad():
        #         output_layer.weight.copy_(weights[-1])
        output_layer.weight = nn.Parameter(torch.ones(output_size,hidden_layers[-1]))
        output_layer.weight.requires_grad=False # freeze output layer
        layer_list.append(output_layer)  # add the output layer

        # create a sequential container of the layers
        self.layers = nn.Sequential(*layer_list)

    def forward(self, x: torch.Tensor):
        """
        Perform a forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor to the neural network.

        Returns:
            torch.Tensor: Output tensor after passing through the network layers.
        """
        x = self.layers(x)
        return x

    def evaluate(self, x):
        """
        Evaluate the neural network model on the input data without affecting the torch.autograd.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the neural network model.
        """
        self.eval()  # set the model to evaluation mode
        with torch.no_grad():
            return self(x)



def train(model, x_train, y_train, num_epochs, lr, print_stuff=True, epochs_reported=100):
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

    if torch.cuda.is_available():
        print("GPU is available!")
    else:
        print("GPU is not available.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x_train = Variable(x_train).cuda()
    y_train = Variable(y_train).cuda()
    # mean-squared error loss
    criterion = torch.nn.MSELoss()

    # Adam/SGD optimizer
    # Note: vanilla gradient descent = stochastic gradient descent with batch size = 1
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    print("starting training")
    losses = []
    for epoch in range(num_epochs):
        model.train()  # set the model to training mode
        # forward propagation
        y_pred = model(x_train)  # evaluate the current model on the data
        loss = criterion(y_pred, y_train)  # compute the loss
        # losses.append(loss.item())
        # back propagation
        optimizer.zero_grad()  # zero out the gradient to add the new one
        loss.backward()  # compute the new gradient
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0)  # clip the gradient norm
        optimizer.step()  # do a SGD step
        if epoch % epochs_reported == 0:
            losses.append(loss.item())
            if print_stuff:
                print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch,
                    num_epochs, loss.item()))
    print('\nTraining Complete')
    model = model.to("cpu")
    return model, losses
