import torch.nn as nn


class CTNetModel(nn.Module):
    def __init__(self):

        super(CTNetModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=14, out_channels=64, kernel_size=(5, 5))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3))
        self.bn4 = nn.BatchNorm2d(96)
        self.conv5 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3))
        self.bn5 = nn.BatchNorm2d(96)
        self.conv6 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(2, 2))
        self.fc1 = nn.Linear(3456, 128)
        self.fc2 = nn.Linear(128, 10)
        self.fc3 = nn.Linear(10, 1)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)


        self.conv_layers = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(),
            self.conv2,
            self.bn2,
            nn.ReLU(),
            self.pool,
            self.conv4,
            self.bn4,
            nn.ReLU(),
            self.conv5,
            self.bn5,
            nn.ReLU(),
            self.pool)

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            self.fc1,
            nn.Dropout(0.25),
            self.fc2,
            self.fc3,
            nn.Sigmoid())

    def forward(self, x):
        out = self.conv_layers(x)
        out = self.fc_layer(out)
        # print(out)
        return out
