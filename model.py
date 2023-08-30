import torch
import torchvision.models
import torch.nn as nn


class functionLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sqrt(torch.pow(x, 2) + 1) - 1


class fcdLossLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, target):  # target 为0：OK，为1：NG
        normalTerm = logits
        anomalyTerm = torch.log(1 - torch.exp(-normalTerm))

        isGood = 1 - target
        loss = torch.mean(isGood * normalTerm - (1 - isGood) * anomalyTerm)
        return loss


class MyVGG_deep(nn.Module):
    def __init__(self, img_size, num_class=1):
        super(MyVGG_deep, self).__init__()
        net = torchvision.models.vgg16_bn(pretrained=True)
        net.classifier = nn.Sequential()

        self.features = net.features[:44]
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.Conv2d(256, 128, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 64, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, num_class, (1, 1), padding=1),
            # functionLayer(),
            # nn.Sigmoid()

            nn.UpsamplingBilinear2d((img_size, img_size)),
            nn.AvgPool2d((img_size, img_size)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class MyVGG(nn.Module):
    def __init__(self, img_size, num_class=1):
        super(MyVGG, self).__init__()
        net = torchvision.models.vgg16_bn(pretrained=True)
        net.classifier = nn.Sequential()

        self.features = net.features[:24]
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 128, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 64, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, num_class, (1, 1), padding=1),
            # functionLayer(),
            # nn.Sigmoid()

            nn.UpsamplingBilinear2d((img_size, img_size)),
            nn.AvgPool2d((img_size, img_size)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# if __name__ == "__main__":
#     myModel = MyVGG(48, 1)
#     print(myModel.features)
#     print(myModel.classifier)
