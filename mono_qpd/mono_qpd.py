import torch
import torch.nn as nn
import torch.nn.functional as F
from mono_qpd.QPDNet.qpd_net import QPDNet
from mono_qpd.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
from mono_qpd.feature_converter import FeatureConverter


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

        self.feature_converter = FeatureConverter(qpdnet_args.image_size)

        self.da_v2 = DepthAnythingV2(da_v2_args.encoder)
        self.qpdnet = QPDNet(qpdnet_args)

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
        
    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        image1_resized = self.resize_to_14_multiples(image1)
        image1_resized_normed = self.normalize_image(image1_resized)
        # enc_features, depth = self.da_v2(image1_resized_normed) # Original
        int_features = self.da_v2(image1_resized_normed)
        int_features = int_features[1:]
        int_features = self.feature_converter(int_features)
        int_features = int_features[::-1] # Reverse the order of the features

        if test_mode:
            original_disp, upsampled = self.qpdnet(int_features, image1, image2, iters=iters, test_mode=test_mode, flow_init=None)
            return original_disp, upsampled
        else:
            disp_predictions = self.qpdnet(int_features, image1, image2, iters=iters, test_mode=test_mode, flow_init=None)
            return disp_predictions

        

