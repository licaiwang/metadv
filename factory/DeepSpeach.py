import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=3, padding=1, stride=stride
        )
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm1d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):

        super(ResNet, self).__init__()

        self.relu = nn.ReLU()
        self.inplanes = 80
        self.conv1 = nn.Conv1d(80, 80, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(80)
        self.layer1 = self._make_layer(block, 80, layers[0])

        self.inplanes = 128
        self.conv2 = nn.Conv1d(80, 128, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.layer2 = self._make_layer(block, 128, layers[1])

        self.inplanes = 256
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.layer3 = self._make_layer(block, 256, layers[2])

        self.inplanes = 512
        self.conv4 = nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.layer4 = self._make_layer(block, 512, layers[3])

        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):

        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class DeepSpeaker(nn.Module):
    def __init__(self, num_classes, embedding_size=256):
        super(DeepSpeaker, self).__init__()

        self.embedding_size = embedding_size
        self.model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes)
        self.model.fc = nn.Linear(512, self.embedding_size)
        self.model.classifier = nn.Linear(self.embedding_size, num_classes)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)

        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer2(x)

        x = self.model.conv3(x)
        x = self.model.bn3(x)
        x = self.model.relu(x)
        x = self.model.layer3(x)

        x = self.model.conv4(x)
        x = self.model.bn4(x)
        x = self.model.relu(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x).squeeze(-1)
        embeds = self.model.fc(x)
        norm = embeds.norm(p=2, dim=-1, keepdim=True)
        embeds_normalized = embeds.div(norm)

        predict = self.model.classifier(embeds)
        return predict, embeds_normalized
