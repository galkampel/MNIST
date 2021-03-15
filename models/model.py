import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, img_size=28, in_channels=1, out_channels=64,
                 k=3, s=2, pad=1, out_features=128, num_classes=10, **params):  # BX28X28 -> 14X14X64 -> 7X7X128
        super(ConvNet, self).__init__()
        convs = []
        n_layers = params.get('n_layers', 2)
        # print(f'n_layers = {n_layers}')
        convs.append(self.set_conv_layer(in_channels, out_channels, k, s, pad))
        in_channels = out_channels
        out_channels = in_channels * 2
        for _ in range(n_layers - 1):
            convs.append(
                self.set_conv_layer(in_channels, out_channels, k, s, pad))
            in_channels = out_channels
            out_channels = in_channels * 2

        self.conv = nn.Sequential(* convs)
        feature_size = img_size // (2 ** n_layers)
        # print(f'feature size = {feature_size}')
        # print(f'in_channels = {in_channels}')
        self.mlp = nn.Linear((feature_size ** 2) * in_channels, out_features)

        self.fc = nn.Linear(out_features, num_classes)

    @staticmethod
    def set_conv_layer(in_channels, out_channels, k, s, pad):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, k, s, pad),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return conv_layer

    def forward(self, x, predict_class=False):
        x = self.conv(x)  # (B, 128, 7, 7)
        x = x.flatten(start_dim=1)  # (B, 128*7*7)
        x = self.mlp(x)  # (B, 128)
        if predict_class:
            x = self.fc(x)  # (B, 10)
            x = F.log_softmax(x, dim=1)
        return x


class MLPNet(nn.Module):
    def __init__(self, img_size=28, out_features=128, num_classes=10, **params):  # 784 -> 512 (256) -> 256 (128) -> 128
        super(MLPNet, self).__init__()
        fcs = []
        in_features = img_size ** 2  # 28*28
        hidden_features = params.get('hidden_features', 512)
        i = 1
        while in_features > out_features:
            p = params.get(f'dropout_l{i}', 1.0)
            fcs.append(
                self.set_fc_layer(in_features, hidden_features, p)
            )
            in_features = hidden_features
            hidden_features //= 2
            i += 1
        self.mlp = nn.Sequential(*fcs)
        self.fc = nn.Linear(out_features, num_classes)

    @staticmethod
    def set_fc_layer(in_fetures, out_features, p):
        fc_layer = nn.Sequential(
            nn.Linear(in_fetures, out_features),
            nn.BatchNorm1d(out_features),
            nn.Dropout(p),
            nn.ReLU()
        )
        return fc_layer

    def forward(self, x, predict_class=False):
        x = x.flatten(1)  # (B, 1, 28, 28) -> (B, 784)
        x = self.mlp(x)  # (B, 128)
        if predict_class:
            x = self.fc(x)  # (B, 10)
            x = F.log_softmax(x, dim=1)
        return x
