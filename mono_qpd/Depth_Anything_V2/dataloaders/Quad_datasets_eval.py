import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import re
import copy
import math
import random
from pathlib import Path
from glob import glob
import os.path as osp
import cv2

from qpd_core.utils import frame_utils
from qpd_core.utils.augmentor import QuadAugmentor, SparseQuadAugmentor
from qpd_core.utils.transforms import RandomBrightness
from depth_anything_v2.util.transform import Resize
# from codes_for_finetunning.transform import Resize, NormalizeImage, PrepareForNet, Crop

class QuadDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None, lrtb='', is_test=False, image_set = 'train', no_gt_disp=False, no_gt_aif=False, resize_size=770):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseQuadAugmentor(**aug_params)
            else:
                self.augmentor = QuadAugmentor(**aug_params)

        self.crop_size = aug_params.pop("crop_size", None) if aug_params is not None else None

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader        

        self.resize = Resize(
                width=resize_size,
                height=resize_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_LINEAR
                )

        
        self.no_gt_disp = no_gt_disp
        self.no_gt_aif = no_gt_aif
        self.is_test = is_test
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.aif_list = []
        self.extra_info = []
        self.lrtb = lrtb
        self.image_set = image_set
        self.randomBright = RandomBrightness()

    def __getitem__(self, index):

        if self.is_test:
            center_img = frame_utils.read_gen(self.image_list[index][0])
            center_img = np.array(center_img).astype(np.uint8)[..., :3]
            center_img = torch.from_numpy(center_img).permute(2, 0, 1).float()
            img_list = []

            for i in range(1,5):
                img = frame_utils.read_gen(self.image_list[index][i])
                img = np.array(img).astype(np.uint8)[..., :3]
                img = torch.from_numpy(img).permute(2, 0, 1).float()
                img_list.append(img)

            imgs = torch.stack(img_list, dim=0)

            return center_img, imgs

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        if not self.no_gt_disp:
            disp = self.disparity_reader(self.disparity_list[index])
            if isinstance(disp, tuple):
                disp, valid = disp
            else:
                valid = disp < 512
            flow = np.stack([-disp, np.zeros_like(disp)], axis=-1)

        img_list = []
        for i in range(0,5):
            
            img = frame_utils.read_gen(self.image_list[index][i])
            img = np.array(img).astype(np.uint8)[..., :3]
            img_list.append(img)

        if not self.no_gt_aif:        
            aif = frame_utils.read_gen(self.aif_list[index])
            aif = np.array(aif).astype(np.uint8)[..., :3]

        # grayscale images
        for i in range(0,5):
            if len(img_list[i].shape) == 2:
                img_list[i] = np.tile(img_list[i][...,None], (1, 1, 3))
            else:
                img_list[i] = img_list[i][..., :3]
        if self.augmentor is not None:
            if not self.no_gt_disp:
                raise NotImplementedError("Augmentation not implemented for ground truth disparity")
            if self.sparse:
                img_list, flow, valid = self.augmentor(img_list, flow, valid)
            else:
                img_list, flow = self.augmentor(img_list, flow)

        if self.crop_size is None:
            h,w,c = img_list[0].shape
            self.crop_size = [h, w]

        lrtb_list = []
        center_img = img_list[0]
        for i in range(1,5):
            lrtb = img_list[i]
            lrtb_list.append(lrtb)

        lrtb_list = np.stack(lrtb_list, axis=-1)

        center_img = torch.from_numpy(center_img).permute(2, 0, 1).float()
        lrtb_list = torch.from_numpy(lrtb_list).permute(3, 2, 0, 1).float()

        if not self.no_gt_aif:
            aif = torch.from_numpy(aif).permute(2, 0, 1).float()
        if not self.no_gt_disp:
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.image_set == 'train':
            lrtb_list = self.randomBright(lrtb_list)
        
        if self.img_pad is not None:
            padH, padW = self.img_pad
            for i in range(0,5):
                img_list[i] = F.pad(img_list[i], [padW]*2 + [padH]*2)

        if not self.no_gt_disp:
            if self.sparse:
                valid = torch.from_numpy(valid)
            else:
                valid = (np.abs(flow[0]) < 512) & (np.abs(flow[1]) < 512)

        if not self.no_gt_disp:
            flow = flow[:1]

        # fitting image to input size

        h, w = center_img.shape[1:]
        scale_factor = 518 / h        

        if self.resize is not None:
            center_img = F.interpolate(center_img[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0] # Hard coding 768, 1024 to 770, 1027
            flow = F.interpolate(flow[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]
            lrtb_list = F.interpolate(lrtb_list, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            
            # flow = self.resize({'image': flow})['image']
            # center_img = self.resize({'image': center_img})['image']
            # resized_left = self.resize({'image': lrtb_list[:, :, :, 0]})['image']
            # resized_right = self.resize({'image': lrtb_list[:, :, :, 1]})['image']
            # lrtb_list = np.stack([resized_left, resized_right], axis=-1)

        h, w = center_img.shape[1:]
        crop_size = h // 14 * 14, w // 14 * 14
        # crop_size = 756, 1022
        h_start = (h - crop_size[0]) // 2
        h_end = h_start + crop_size[0]
        w_start = (w - crop_size[1]) // 2
        w_end = w_start + crop_size[1]

        center_img = center_img[:, h_start:h_end, w_start:w_end]
        lrtb_list = lrtb_list[:, :, h_start:h_end, w_start:w_end]
        flow = flow[:, h_start:h_end, w_start:w_end][0]
        valid = valid[h_start:h_end, w_start:w_end]

        # Normalization
        center_ori = center_img.clone()
        mean=np.array([0.485, 0.456, 0.406])[:, None, None]
        std=np.array([0.229, 0.224, 0.225])[:, None, None]
        center_img = center_img / center_img.max()
        center_img = (center_img - mean) / std

        if not self.no_gt_disp:
            result = self.image_list[index] + [self.disparity_list[index]], center_img, lrtb_list, flow, valid, center_ori
        else:
            result = self.image_list[index], center_img, lrtb_list
            
        return result

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self
        
    def __len__(self):
        return len(self.image_list)

class QPD(QuadDataset):
    def __init__(self, aug_params=None, root='', image_set='train'):
        super(QPD, self).__init__(aug_params, sparse=False, lrtb='', image_set = image_set)
        assert os.path.exists(root)
        imagel_list = sorted(glob(os.path.join(root, image_set+'_l','source', 'seq_*/*.png')))
        imager_list = sorted(glob(os.path.join(root, image_set+'_r','source', 'seq_*/*.png')))
        imaget_list = sorted(glob(os.path.join(root, image_set+'_t','source', 'seq_*/*.png')))
        imageb_list = sorted(glob(os.path.join(root, image_set+'_b','source', 'seq_*/*.png')))
        imagec_list = sorted(glob(os.path.join(root, image_set+'_c','source', 'seq_*/*.png')))
        aif_list = sorted(glob(os.path.join(root, image_set+'_c','target', 'seq_*/*.png')))
        disp_list = sorted(glob(os.path.join(root, image_set+'_c','target_disp', 'seq_*/*.npy')))

        for idx, (imgc, imgl, imgr, imgt, imgb, aif, disp) in enumerate(zip(imagec_list, imagel_list, imager_list, imaget_list, imageb_list, aif_list, disp_list)):
            self.image_list += [ [imgc, imgl, imgr, imgt, imgb] ]
            self.aif_list += [ aif ]
            self.disparity_list += [ disp ]

        self.image_list = self.image_list
        self.disparity_list = self.disparity_list

class Real_QPD(QuadDataset):
    def __init__(self, aug_params=None, root='', image_set='train', no_gt_disp=True, no_gt_aif=True):
        super(Real_QPD, self).__init__(aug_params, sparse=False, lrtb='', image_set = image_set, no_gt_disp=no_gt_disp, no_gt_aif=True)
        assert os.path.exists(root)
        # imagel_list = sorted(glob(os.path.join(root, image_set+'_l','source', 'seq_*/*.png')))
        # imager_list = sorted(glob(os.path.join(root, image_set+'_r','source', 'seq_*/*.png')))
        # imaget_list = sorted(glob(os.path.join(root, image_set+'_t','source', 'seq_*/*.png')))
        # imageb_list = sorted(glob(os.path.join(root, image_set+'_b','source', 'seq_*/*.png')))
        # imagec_list = sorted(glob(os.path.join(root, image_set+'_c','source', 'seq_*/*.png')))
        
        imagel_list = sorted(glob(os.path.join(root, image_set+'_l','source', 'seq_*/*.*')))
        imager_list = sorted(glob(os.path.join(root, image_set+'_r','source', 'seq_*/*.*')))
        imaget_list = sorted(glob(os.path.join(root, image_set+'_t','source', 'seq_*/*.*')))
        imageb_list = sorted(glob(os.path.join(root, image_set+'_b','source', 'seq_*/*.*')))
        imagec_list = sorted(glob(os.path.join(root, image_set+'_c','source', 'seq_*/*.*')))
        

        for idx, (imgc, imgl, imgr, imgt, imgb) in enumerate(zip(imagec_list, imagel_list, imager_list, imaget_list, imageb_list)):
            self.image_list += [ [imgc, imgl, imgr, imgt, imgb] ]

class MDD(QuadDataset):
    def __init__(self, aug_params=None, root='', image_set='train', resize_size = 768):
        super(MDD, self).__init__(aug_params, sparse=False, lrtb='', image_set = image_set, no_gt_aif=True, no_gt_disp=False)
        
        if resize_size is None:
            self.resize_size = None
        else:
            self.resize = Resize(
                width=resize_size,
                height=resize_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_LINEAR
                )


        assert os.path.exists(root)
        imagel_list = sorted(glob(os.path.join(root, image_set+'_l','source', 'seq_*/*.jpg')))
        imager_list = sorted(glob(os.path.join(root, image_set+'_r','source', 'seq_*/*.jpg')))
        imagec_list = sorted(glob(os.path.join(root, image_set+'_c','source', 'seq_*/*.jpg')))
        depth_list = sorted(glob(os.path.join(root, image_set+'_c','target_depth', 'seq_*/*.TIF')))

        for idx, (imgc, imgl, imgr, depth) in enumerate(zip(imagec_list, imagel_list, imager_list, depth_list)):
            self.image_list += [ [imgc, imgl, imgr] ]
            self.disparity_list += [ depth ] # depth


    def __getitem__(self, index):

        if self.is_test:
            center_img = frame_utils.read_gen(self.image_list[index][0])
            center_img = np.array(center_img).astype(np.uint8)[..., :3]
            center_img = torch.from_numpy(center_img).permute(2, 0, 1).float()
            img_list = []

            for i in range(1,3):
                img = frame_utils.read_gen(self.image_list[index][i])
                img = np.array(img).astype(np.uint8)[..., :3]
                img = torch.from_numpy(img).permute(2, 0, 1).float()
                img_list.append(img)

            imgs = torch.stack(img_list, dim=0)

            return center_img, imgs

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        if not self.no_gt_disp:
            depth = self.disparity_reader(self.disparity_list[index])
            valid = depth > 0
            
        img_list = []
        for i in range(0,3):
            
            img = frame_utils.read_gen(self.image_list[index][i])
            img = np.array(img).astype(np.uint8)[..., :3]
            img_list.append(img)

        # grayscale images
        for i in range(0,3):
            if len(img_list[i].shape) == 2:
                img_list[i] = np.tile(img_list[i][...,None], (1, 1, 3))
            else:
                img_list[i] = img_list[i][..., :3]

        if self.crop_size is None:
            h,w,c = img_list[0].shape
            self.crop_size = [h, w]

        lrtb_list = []
        center_img = img_list[0]
        for i in range(1,3):
            lrtb = img_list[i]
            lrtb_list.append(lrtb)

        lrtb_list = np.stack(lrtb_list, axis=-1)
        
        h, w = center_img.shape[:2]

        center_img = torch.from_numpy(center_img).permute(2, 0, 1).float()
        lrtb_list = torch.from_numpy(lrtb_list).permute(3, 2, 0, 1).float()
        depth = torch.from_numpy(depth).unsqueeze(0).float()

        scale_factor = 518 / h
        if self.resize is not None:
            center_img = F.interpolate(center_img[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0] # Hard coding 768, 1024 to 770, 1027
            lrtb_list = F.interpolate(lrtb_list, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            depth = F.interpolate(depth[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]

            
        h, w = center_img.shape[1:]
        crop_size = h // 14 * 14, w // 14 * 14
        # crop_size = 756, 1022
        h_start = (h - crop_size[0]) // 2
        h_end = h_start + crop_size[0]
        w_start = (w - crop_size[1]) // 2
        w_end = w_start + crop_size[1]

        center_img = center_img[:, h_start:h_end, w_start:w_end]
        lrtb_list = lrtb_list[:, :, h_start:h_end, w_start:w_end]
        valid = valid[h_start:h_end, w_start:w_end]
        depth = depth[:, h_start:h_end, w_start:w_end][0]

        # Normalization
        center_ori = center_img.clone()
        mean=np.array([0.485, 0.456, 0.406])[:, None, None]
        std=np.array([0.229, 0.224, 0.225])[:, None, None]
        center_img = center_img / center_img.max()
        center_img = (center_img - mean) / std


        # if self.resize is not None:
        #     depth = self.resize({'image': depth})['image']
        #     center_img = self.resize({'image': center_img})['image']
        #     resized_left = self.resize({'image': lrtb_list[:, :, :, 0]})['image']
        #     resized_right = self.resize({'image': lrtb_list[:, :, :, 1]})['image']
        #     lrtb_list = np.stack([resized_left, resized_right], axis=-1)


        

        # center_img = torch.from_numpy(center_img).permute(2, 0, 1).float()
        # lrtb_list = torch.from_numpy(lrtb_list).permute(3, 2, 0, 1).float()
        # depth = torch.from_numpy(depth).unsqueeze(2).float()

        valid_gt = torch.ones_like(depth).float()
        
        result = self.image_list[index] + [self.disparity_list[index]], center_img, lrtb_list, depth, valid_gt, center_ori
            
        return result

  
def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = None
    
    for dataset_name in args.train_datasets:
        if dataset_name.startswith("QPD"):
            new_dataset = QPD(aug_params, root=args.datasets_path)
        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=True, shuffle=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=True)

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader