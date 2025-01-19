import cv2
import torch
from util import affine_invariant_metrics as ai_metrics
from util.colorize import show_colormap, show_np_colormap
import os

def _eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    mae = torch.mean(torch.abs(diff))

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    ai1, b = ai_metrics.affine_invariant_1(pred.cpu().numpy(), target.cpu().numpy(), target.cpu().numpy() > 0)
    ai2 = ai_metrics.affine_invariant_2(pred.cpu().numpy(), target.cpu().numpy(), target.cpu().numpy() > 0)[0]

    return d1, d2, d3, abs_rel, sq_rel, mae, rmse, rmse_log, log10, silog, ai1, ai2, b


def eval_depth(pred, target):
    d1, d2, d3, abs_rel, sq_rel, mae, rmse, rmse_log, log10, silog, ai1, ai2, b = _eval_depth(pred, target)
    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 'mae': mae.item(),
            'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item(), 'ai1':ai1, 'ai2':ai2, 
            'scaling_factor':b[0], 'shifting_factor':b[1]}

def depth_colormap(pred, valid_mask, save_path, value_range=[0, 0.012], err_percentage=0.4):
    vis_range = [max(value_range[0] - (value_range[1] - value_range[0]) * err_percentage, 0.0), value_range[1] + (value_range[1] - value_range[0]) * err_percentage] # 0~1.0 -> 0~1.4
    err_range = (value_range[1] - value_range[0]) * err_percentage

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    valid_mask = valid_mask.cpu().numpy()
    pred[valid_mask==0] = 0
    pred = pred.cpu().numpy()

    # show_colormap(pred, save_path + '_pred', vis_range, 270, figsize=(12, 10))
    # show_colormap(pred, save_path + '_pred_min_max', [pred[valid_mask].min(), pred[valid_mask].max()], 270, figsize=(12, 10))

    show_np_colormap(pred, save_path + '_pred', vis_range)
    show_np_colormap(pred, save_path + '_pred_min_max', [pred[valid_mask].min(), pred[valid_mask].max()])



def eval_depth_colormap(pred, target, valid_mask, save_path=None):
    d1, d2, d3, abs_rel, sq_rel, mae, rmse, rmse_log, log10, silog, ai1, ai2, b = _eval_depth(pred[valid_mask], target[valid_mask])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        L1_fit = pred * b[0] + b[1]
        mae_2d = torch.abs(pred - target)
        ai1_mae_2d = torch.abs(L1_fit - target)
        valid_mask = valid_mask.cpu().numpy()

        pred[valid_mask==0] = 0
        L1_fit[valid_mask==0] = 0
        mae_2d[valid_mask==0] = 0
        ai1_mae_2d[valid_mask==0] = 0

        L1_fit = L1_fit.cpu().numpy()
        mae_2d = mae_2d.cpu().numpy()
        ai1_mae_2d = ai1_mae_2d.cpu().numpy()
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()


        cv2.imwrite(save_path + '_pred.png', pred * (pred - pred.min()) / (pred.max() - pred.min()) * 255)
    
    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 'mae': mae.item(),
            'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item(), 'ai1':ai1, 'ai2':ai2, 
            'scaling_factor':b[0], 'shifting_factor':b[1]}