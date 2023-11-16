import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ExpectationResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ExpectationResBlock, self).__init__()
        # share conv
        self.conv1 = conv3x3(in_c, out_c, 1)
        self.conv2 = conv3x3(out_c, out_c, 1)
        # independent bn
        self.bn11 = nn.BatchNorm2d(out_c)
        self.bn12 = nn.BatchNorm2d(out_c)

        self.bn21 = nn.BatchNorm2d(out_c)
        self.bn22 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU(inplace=True)

        #
        self.exp_bn1 = nn.BatchNorm2d(1, affine=False, track_running_stats=False)
        self.exp_bn2 = nn.BatchNorm2d(1, affine=False, track_running_stats=False)


    def channel_expectation(self, x1, x2):
        expect_p1 = torch.sum(torch.softmax(x1, dim=1) * x1, dim=1, keepdim=True)
        expect_p1 = torch.sigmoid(self.exp_bn1(expect_p1))
        expect_p2 = torch.sum(torch.softmax(x2, dim=1) * x2, dim=1, keepdim=True)
        expect_p2 = torch.sigmoid(self.exp_bn2(expect_p2))
        union_expect = expect_p1 + expect_p2 - expect_p1 * expect_p2
        x1 = x1 * union_expect
        x2 = x2 * union_expect
        return x1, x2

    def forward(self, x1, x2):
        out1 = self.conv1(x1)
        out1 = self.bn11(out1)
        out1 = self.relu(out1)
        out1 = self.conv2(out1)
        out1 = self.bn12(out1)

        out2 = self.conv1(x2)
        out2 = self.bn21(out2)
        out2 = self.relu(out2)
        out2 = self.conv2(out2)
        out2 = self.bn22(out2)

        out1, out2 = self.channel_expectation(out1, out2)

        out1 = self.relu(x1 + out1)
        out2 = self.relu(x2 + out2)

        return out1, out2



