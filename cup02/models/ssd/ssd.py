import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .vggbase import SSDBackbone
from .anchors import AnchorBox, v2
import sys
base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class SSD(nn.Module):
    def __init__(self, num_classes=20):
        super(SSD, self).__init__()
        self.num_classes = num_classes + 1  # Include background class
        self.phase = 'train'
        self.L2Norm = L2Norm(512, 20)

        # Initialize base VGG16 network as the backbone
        self.backbone = nn.ModuleList(vgg(base['300'], 3))

        # Extra feature layers for multi-scale detection
        self.extra_features = nn.ModuleList([
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        ])

        # Regression and classification layers for bounding boxes and classes
        self.reg_layers = nn.ModuleList([
            nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1),  # conv4_3 feature map
            nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1), # fc7 feature map
            nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),  # Extra layers
            nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)
        ])
        self.cls_layers = nn.ModuleList([
            nn.Conv2d(512, 4 * self.num_classes, kernel_size=3, padding=1),  # conv4_3 feature map
            nn.Conv2d(1024, 6 * self.num_classes, kernel_size=3, padding=1), # fc7 feature map
            nn.Conv2d(512, 6 * self.num_classes, kernel_size=3, padding=1),  # Extra layers
            nn.Conv2d(256, 6 * self.num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * self.num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * self.num_classes, kernel_size=3, padding=1)
        ])

        # Precompute anchors and store them as a buffer (not updated during training)
        self.register_buffer('anchors', torch.from_numpy(AnchorBox(v2)()).float())

        # Initialize weights using Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        features = []
        
        # Forward through base network
        for k in range(23):
            x = self.backbone[k](x)
        block4_3_output = self.L2Norm(x)

        for k in range(23, len(self.backbone)):
            x = self.backbone[k](x)
        backbone_output = x

        features.append(block4_3_output)  # Add conv4_3 feature map
        features.append(backbone_output)  # Add fc7 feature map
        
        # Forward through extra layers
        out = backbone_output
        # print("!!!!", flush=True)
        for i, layer in enumerate(self.extra_features):
            out = F.relu(layer(out), inplace=True)
            # print(f"idx {i}, shape: {out.shape}", flush=True)
            if i % 2 == 1:  # Collect every other output
                features.append(out)
    
        # Prediction (regression and classification) for each feature map
        reg_outputs = []
        cls_outputs = []
        for feature, reg_layer, conf_layer in zip(features, self.reg_layers, self.cls_layers):
            # print(f"feature shape{feature.shape}", flush=True)
            out = reg_layer(feature).permute(0, 2, 3, 1).contiguous()  # Regression
            # print(f"reg shape{out.shape}", flush=True)
            reg_outputs.append(out.view(out.size(0), -1))
            out = conf_layer(feature).permute(0, 2, 3, 1).contiguous()  # Classification
            # print(f"cls shape{out.shape}", flush=True)
            cls_outputs.append(out.view(out.size(0), -1))

        # Concatenate all predictions from different feature maps
        reg_outputs = torch.cat(reg_outputs, dim=1)  # [B, 8732*4]
        reg_outputs = reg_outputs.view(reg_outputs.size(0), -1, 4)  # [B, 8732, 4]
        cls_outputs = torch.cat(cls_outputs, dim=1)  # [B, 8732*21]
        cls_outputs = cls_outputs.view(cls_outputs.size(0), -1, self.num_classes)  # [B, 8732, 21]

        if self.phase == 'eval':
            # Decode boxes and apply softmax to class predictions for evaluation
            decoded_boxes = torch.cat((self.anchors[None, :, :2] + reg_outputs[:, :, :2] * 0.1 * self.anchors[None, :, 2:],
                                       self.anchors[None, :, 2:] * torch.exp(reg_outputs[:, :, 2:] * 0.2)), -1)
            decoded_boxes[:, :, :2] -= decoded_boxes[:, :, 2:] / 2
            decoded_boxes[:, :, 2:] += decoded_boxes[:, :, :2]
            return decoded_boxes, F.softmax(cls_outputs, dim=-1)
        else:
            return reg_outputs, cls_outputs
        

def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers