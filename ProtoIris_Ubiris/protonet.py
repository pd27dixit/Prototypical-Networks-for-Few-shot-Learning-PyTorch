# import torch.nn as nn


# def conv_block(in_channels, out_channels):
#     '''
#     returns a block conv-bn-relu-pool
#     '''
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 3, padding=1),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(),
#         nn.MaxPool2d(2)
#     )


# class ProtoNet(nn.Module):
#     '''
#     Model as described in the reference paper,
#     source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
#     '''
#     def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
#         super(ProtoNet, self).__init__()
#         self.encoder = nn.Sequential(
#             conv_block(x_dim, hid_dim),
#             conv_block(hid_dim, hid_dim),
#             conv_block(hid_dim, hid_dim),
#             conv_block(hid_dim, z_dim),
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         return x.view(x.size(0), -1)


import torch
import torch.nn as nn
import torchvision.models as models

# ----Removes the last FC layer (only keeps feature extraction layers)
class ProtoNet2(nn.Module):
    def __init__(self, z_dim=512):
        super(ProtoNet2, self).__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Modify first convolutional layer to accept 1-channel images
        first_conv_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        first_conv_layer.weight = nn.Parameter(torch.mean(base_model.conv1.weight, dim=1, keepdim=True))  # Convert 3-channel to 1-channel

        base_model.conv1 = first_conv_layer  # Replace original conv layer

        self.encoder = nn.Sequential(*list(base_model.children())[:-1])  # Remove FC layer

        # Global Average Pooling + Fully Connected Layer
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, z_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)  # Feature extraction
        x = self.projection(x)  # Projection to z_dim
        return x

