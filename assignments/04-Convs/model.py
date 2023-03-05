import torch


class Model(torch.nn.Module):
    """
    A custome made CNN as part of homework 4. Needs to get minimum of 55% accuracy asap.
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_channels, out_channels=32, kernel_size=3
        )
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = torch.nn.Linear(12544, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)
        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        computes the output of the model
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
