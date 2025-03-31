import torch.nn as nn


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

"""
def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ProtoNet(nn.Module):
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        # print(f"Input shape: {x.shape}")  # Check input shape Input shape: torch.Size([50, 1, 85, 85])

        for i, layer in enumerate(self.encoder):
            x = layer(x)
            # print(f"After conv_block {i+1}: {x.shape}")  # Track shape after each block
        
        x = x.view(x.size(0), -1)  # Flatten
        # print(f"Final output shape: {x.shape}") #Final output shape: torch.Size([50, 1600])
        
        return x

"""






# import torch
# import torch.nn as nn
# import onnxruntime as ort
# import numpy as np

# class ProtoNet(nn.Module):
#     def __init__(self, onnx_model_path, feature_dim=2048, num_classes=45):
#         super(ProtoNet, self).__init__()
        


#         # Load ONNX model using ONNX Runtime
#         self.ort_session = ort.InferenceSession(onnx_model_path)
#         # print("ONNX Model Input Shape:", self.ort_session.get_inputs()[0].shape)

#         # Fully connected layer for classification
#         self.fc = nn.Linear(feature_dim, num_classes)  #Trainable layer

#     def forward(self, x):
#         # Convert PyTorch Tensor to NumPy for ONNX input
#         x_np = x.cpu().detach().numpy().astype(np.float32)

#         # Run inference through ONNX model
#         ort_inputs = {self.ort_session.get_inputs()[0].name: x_np}
#         ort_outs = self.ort_session.run(None, ort_inputs)

#         # Convert ONNX output to PyTorch Tensor
#         x_torch = torch.from_numpy(ort_outs[0]).to(x.device)

#         # Pass through the fully connected layer
#         return self.fc(x_torch.view(x_torch.size(0), -1))  # 🔥 Now trainable

# # Usage
# onnx_model_path = "/old/home/nishkal/PD/DeepIris_Recog_Drive/ResNet50_Iris.onnx"
# model = ProtoNet(onnx_model_path).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))












import torch
import torch.nn as nn
import torchvision.models as models


# # --- Keeps the entire ResNet50 model, including the FC layer.
# class ProtoNet1(nn.Module):
#     def __init__(self, z_dim=512):
#         super(ProtoNet1, self).__init__()
#         base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
#         # Modify first conv layer to accept 1-channel images
#         first_conv_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         first_conv_layer.weight = nn.Parameter(torch.mean(base_model.conv1.weight, dim=1, keepdim=True))
#         base_model.conv1 = first_conv_layer

#         # Replace the FC layer instead of removing it
#         in_features = base_model.fc.in_features
#         base_model.fc = nn.Linear(in_features, z_dim)  # Adjust FC layer to output z_dim
        
#         self.encoder = base_model  # Keep full ResNet50 with modified first & last layers

#     def forward(self, x):
#         x = self.encoder(x)  # Directly pass through the model (includes the FC layer)
#         return x  # Output is now (batch_size, z_dim)



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
