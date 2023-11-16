import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CIAN(nn.Module):
    def __init__(self, gate_channels=512):
        super(CIAN, self).__init__()
        self.relu = nn.ReLU
        self.norm = nn.BatchNorm2d

        self.fc1 = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels//16),
            self.relu(),
            nn.Linear(gate_channels//16, gate_channels//2)
        )
        self.fc2 = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels//16),
            self.relu(),
            nn.Linear(gate_channels//16, gate_channels//2)
        )

        self.conv = nn.Sequential(
            self.norm(512,),
            self.relu(),
            nn.Conv2d(512, 256, 1, 1, bias=False),

            self.norm(256),
            self.relu(),
            nn.Conv2d(256, 256, 3, 1, bias=False, padding=1),

            self.norm(256),
            self.relu(),
            nn.Conv2d(256, 256, 1, 1, bias=False),

        )


    def forward(self, rgb_bbox_feats, lwir_bbox_feats):
        x = torch.cat([rgb_bbox_feats, lwir_bbox_feats], 1)
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        rgb_scale = torch.sigmoid(self.fc1(avg_pool)).unsqueeze(2).unsqueeze(3).expand_as(rgb_bbox_feats)
        rgb_bbox_feats = rgb_bbox_feats * rgb_scale

        lwir_scale = torch.sigmoid(self.fc2(avg_pool)).unsqueeze(2).unsqueeze(3).expand_as(lwir_bbox_feats)
        lwir_bbox_feats = lwir_bbox_feats * lwir_scale

        x = torch.cat([rgb_bbox_feats, lwir_bbox_feats], 1)
        x = self.conv(x)
        return x


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('/home/zx/myPythonProject/cross-modality-det/cian')
    cian = CIAN()
    x = torch.rand([1, 256, 7, 7])
    y = torch.rand([1, 256, 7, 7])
    print(cian)
    writer.add_graph(cian, [x, y])

    feats = cian(x, y)
    print(feats.shape)