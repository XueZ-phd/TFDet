import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch.nn.functional as F
from mmcv.runner import auto_fp16


class zxGlobal2LocalFPN(nn.Module):

    def __init__(self,
                 in_channels=[256, 256, 256],
                 norm_cfg=None,
                 act_cfg=None):
        super(zxGlobal2LocalFPN, self).__init__()

        # down sample conv
        self.downsample_convs = nn.ModuleList()
        for idx in range(len(in_channels)-1):
            self.downsample_convs.append(nn.Sequential(*[ConvModule(in_channels[idx],
                                                                    in_channels[idx], 3, 2, 1,
                                                                    norm_cfg=norm_cfg,
                                                                    act_cfg=act_cfg,
                                                                    inplace=False) for _ in range(len(in_channels)-1-idx)]))
        # out_conv
        self.out_conv = nn.ModuleList()
        for idx in range(len(in_channels)):
            self.out_conv.append(ConvModule(in_channels[idx], in_channels[idx],
                                            3, 1, 1,
                                            norm_cfg=norm_cfg,
                                            act_cfg=act_cfg,
                                            inplace=False))

    @auto_fp16()
    def forward(self, fpn_outs):
        g = fpn_outs[-1]
        dsf = []
        for idx in range(len(fpn_outs) - 1):
            dsf.append(self.downsample_convs[idx](fpn_outs[idx]))
        for idx in range(len(fpn_outs)-1):
            g = g + dsf[idx]
        inters = [None for _ in range(len(fpn_outs)-1)] + [g]
        for idx in range(len(fpn_outs) - 2, -1, -1):
            inters[idx] = fpn_outs[idx] + F.interpolate(inters[idx+1], scale_factor=2, mode='nearest')

        outs = [self.out_conv[i](inters[i]) for i in range(len(fpn_outs))]
        return tuple(outs)








if __name__ == '__main__':
    fpn_outs = [torch.randn(1, 256, 128, 160, requires_grad=True),
                torch.randn(1, 256, 64, 80, requires_grad=True),
                torch.randn(1, 256, 32, 40, requires_grad=True)]
    gl2FPN = zxGlobal2LocalFPN()
    gl2FPN.eval()
    print([x.shape for x in gl2FPN(fpn_outs)])
    torch.onnx.export(gl2FPN, fpn_outs, '../../../runs/debugs/g2lFPN.onnx', verbose=True,
                      opset_version=11,
                      export_params=True,
                      do_constant_folding=True,
                      input_names=['fpn1', 'fpn2', 'fpn3'],
                      output_names=['output1', 'output2', 'output3'],
                      dynamic_axes={'input' : {0 : 'batch_size'},
                                'output' : {0 : 'batch_size'}})



