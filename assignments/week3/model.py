import torch
from typing import Callable


class MLP(torch.nn.Module):
    """
    MLP implementation using pytorch
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        # initialize the class
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.hidden_count = hidden_count
        self.activation = activation
        self.initializer = initializer

        self.layers = torch.nn.ModuleList()

        # Loop over layers and create each one
        for i in range(hidden_count):
            self.layers += [torch.nn.Linear(input_size, hidden_size)]
            input_size = hidden_size

        # Create final layer
        self.out = torch.nn.Linear(hidden_size, num_classes)

        # Initialize the weights
        for layer in self.layers:
            self.initializer(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        self.initializer(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        for layer in self.layers:
            x = layer(x)
            x = self.activation()(x)

        # Get outputs
        return self.out(x)
