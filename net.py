import torch.nn as nn


class Model(nn.Module):
    """Convolutional neural network from the article.
     (gaussian noise is added in dataset class)"""
    def __init__(self, n_classes):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.999),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.999),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.999),
            nn.LeakyReLU(0.1)
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.drop1 = nn.Dropout(0.5)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=0.999),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=0.999),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=0.999),
            nn.LeakyReLU(0.1)
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.drop2 = nn.Dropout(0.5)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            nn.BatchNorm2d(512, momentum=0.999),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256, momentum=0.999),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128, momentum=0.999),
            nn.LeakyReLU(0.1)

        )

        self.pool3 = nn.AvgPool2d(6)

        self.dense = nn.Linear(128, n_classes)
        self.batch_norm = nn.BatchNorm1d(10, momentum=0.999)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        out = self.batch_norm(x)
        return out
