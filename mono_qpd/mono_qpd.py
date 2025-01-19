import torch
import torch.nn as nn
import torch.nn.functional as F
from mono_qpd.QPDNet.qpd_net import QPDNet
from mono_qpd.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class MonoQPD(nn.Module):
    def __init__(self, args):
        super().__init__()
        qpdnet_args = args['qpdnet']
        da_v2_args = args['da_v2']

        self.qpdnet = QPDNet(qpdnet_args)
        self.da_v2 = DepthAnythingV2(da_v2_args.encoder)
        
    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        enc_features, depth = self.da_v2(image1)
        disp_predictions = self.qpdnet(enc_features, image1, image2)

        return disp_predictions
