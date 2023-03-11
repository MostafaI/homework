import torch


class Model(torch.nn.Module):
    """
    A custome made CNN as part of homework 4. Needs to get minimum of 55% accuracy asap.
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_channels, out_channels=8, kernel_size=3
        )
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.fc1 = torch.nn.Linear(576, num_classes)
        # self.fc2 = torch.nn.Linear(128, num_classes)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.dropout = torch.nn.Dropout(0.4)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.bn3 = torch.nn.BatchNorm1d(576)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        computes the output of the model
        """
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(self.bn3(x))
        # x = self.fc2(torch.relu(x))
        return x
