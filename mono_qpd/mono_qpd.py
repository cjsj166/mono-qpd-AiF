import torch
import torch.nn as nn
import torch.nn.functional as F
from mono_qpd.QPDNet.qpd_net import QPDNet
from mono_qpd.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
from mono_qpd.feature_converter import PixelShuffleConverter, ConvConverter


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
        else_args = args['else']
        da_v2_args = args['da_v2']

        if else_args.feature_converter == 'pixelshuffle':
            self.feature_converter = PixelShuffleConverter()
        elif else_args.feature_converter == 'conv':
            self.feature_converter = ConvConverter()

        self.da_v2 = DepthAnythingV2(da_v2_args.encoder)
        self.qpdnet = QPDNet(else_args)

    def resize_to_14_multiples(self, image):
        h, w = image.shape[2], image.shape[3]
        new_h = (h // 14) * 14
        new_w = (w // 14) * 14

        resized_image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
        return resized_image
    
    def normalize_image(self, image):
        # Normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
        image = image / image.max()
        image = (image - mean) / std
        return image
        
    def forward(self, image1, image2, AiF_image, iters=12, flow_init=None, test_mode=False):
        h, w = image1.shape[2], image1.shape[3]
        assert h % 224 == 0 and w % 224 == 0, "Image dimensions must be multiples of 224"

        # image1_normalized = self.normalize_image(image1)
        # int_features = self.da_v2(image1_normalized)
        # int_features = int_features[1:]
        # int_features = self.feature_converter(int_features)
        # int_features = int_features[::-1] # Reverse the order of the features

        AiF_normalized = self.normalize_image(AiF_image)
        int_features = self.da_v2(AiF_normalized)
        int_features = int_features[1:]
        int_features = self.feature_converter(int_features)
        int_features = int_features[::-1] # Reverse the order of the features


        if test_mode:
            original_disp, upsampled = self.qpdnet(int_features, image1, image2, iters=iters, test_mode=test_mode, flow_init=None)
            return original_disp, upsampled
        else:
            disp_predictions = self.qpdnet(int_features, image1, image2, iters=iters, test_mode=test_mode, flow_init=None)
            return disp_predictions

