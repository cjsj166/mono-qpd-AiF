from evaluation.affine_invariant_metrics import *
import os.path as osp
from datetime import datetime
import numpy as np

class Eval():
    def __init__(self, save_path=''):
        self.ai1s = []
        self.b1s = []
        self.ai2s = []
        self.b2s = []
        self.scs = []
        self.bad_0_1px = []
        self.bad_0_5px = []
        self.bad_1px = []
        self.bad_3px = []
        self.epe = []
        self.rmse = []
        self.filenames = []
        self.color_range = []

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_path = save_path + '_eval_' + timestamp + '.txt'

    def add_colorrange(self, vmin, vmax):
        self.color_range.append((vmin, vmax))

    def add_filename(self, filename):
        self.filenames.append(filename)

    def affine_invariant_1(self, Y, Target, confidence_map=None, irls_iters=5, eps=1e-3):
        ai1, b1 = affine_invariant_1(Y, Target, confidence_map, irls_iters, eps)
        self.ai1s.append(ai1)
        self.b1s.append(b1)
        return self.ai1s[-1], self.b1s[-1]
    
    def affine_invariant_2(self, Y, Target, confidence_map=None, eps=1e-3):
        ai2, b2 = affine_invariant_2(Y, Target, confidence_map, eps)
        self.ai2s.append(ai2)
        self.b2s.append(b2)
        return self.ai2s[-1], self.b2s[-1]

    def spearman_correlation(self, Y, Target):
        self.scs.append(spearman_correlation(Y, Target))
        return self.scs[-1]

    def ai2_bad_pixel_metrics(self, Y, Target):
        ai2, b2 = self.affine_invariant_2(Y, Target)

        self.bad_0_1px.append(self.bad_pixel_metric(Y*b2[0] + b2[1], Target, 0.1))
        self.bad_0_5px.append(self.bad_pixel_metric(Y*b2[0] + b2[1], Target, 0.5))
        self.bad_1px.append(self.bad_pixel_metric(Y*b2[0] + b2[1], Target, 1.0))
        self.bad_3px.append(self.bad_pixel_metric(Y*b2[0] + b2[1], Target, 3.0))

        return self.bad_0_1px[-1], self.bad_0_5px[-1], self.bad_1px[-1], self.bad_3px[-1]
    
    def bad_pixel_metric(self, Y, Target, threshold):
        diff = np.abs(Y - Target)
        bad_pixels = np.sum(diff > threshold)
        total_pixels = np.prod(Y.shape)
        return bad_pixels / total_pixels

    def end_point_error(self, Y, Target):
        # epe = np.sqrt(np.sum((Y - Target) ** 2, axis=-1))
        epe = np.mean(np.abs(Y - Target))
        self.epe.append(np.mean(epe))
        return self.epe[-1]

    def root_mean_squared_error(self, Y, Target):
        rmse = np.sqrt(np.mean((Y - Target) ** 2))
        self.rmse.append(rmse)
        return self.rmse[-1]

    def get_mean_metrics(self):
        dic = {
            'ai1': np.mean(self.ai1s),
            'ai1-scale': np.mean(self.b1s, axis=0)[0],
            'ai1-bias': np.mean(self.b1s, axis=0)[1],
            'ai2': np.mean(self.ai2s),
            'ai2-scale': np.mean(self.b2s, axis=0)[0],
            'ai2-bias': np.mean(self.b2s, axis=0)[1],
            'sc': np.mean(self.scs),
            'bad_0_1px': np.mean(self.bad_0_1px),
            'bad_0_5px': np.mean(self.bad_0_5px),
            'bad_1px': np.mean(self.bad_1px),
            'bad_3px': np.mean(self.bad_3px),
            'epe': np.mean(self.epe),
            'rmse': np.mean(self.rmse)
        }
        return dic
        
    def save_metrics(self):
        with open(self.save_path, "w") as f:
            f.write("filename affine-invariant-1 ai1-scale ai1-bias affine-invariant-2 ai2-scale ai2-bias spearman-coefficient color-range bad<0.1px bad<0.5px bad<1px bad<3px epe rmse\n")
            for filename, ai1, b1, ai2, b2, rng, bad_0_1, bad_0_5, bad_1, bad_3, epe, rmse in zip(self.filenames, self.ai1s, self.b1s, self.ai2s, self.b2s, self.color_range, self.bad_0_1px, self.bad_0_5px, self.bad_1px, self.bad_3px, self.epe, self.rmse):
                f.write(f"{filename} {ai1:.3f} {b1[0]:.3f} {b1[1]:.3f} {ai2:.3f} {b2[0]:.3f} {b2[1]:.3f} {rng[0]:.3f}-{rng[1]:.3f} {bad_0_1:.3f} {bad_0_5:.3f} {bad_1:.3f} {bad_3:.3f} {epe:.3f} {rmse:.3f}\n")
            
            # write mean
            f.write(f"mean {np.mean(self.ai1s):.3f} {np.mean(self.b1s, axis=0)[0]:.3f} {np.mean(self.b1s, axis=0)[1]:.3f} {np.mean(self.ai2s):.3f} {np.mean(self.b2s, axis=0)[0]:.3f} {np.mean(self.b2s, axis=0)[1]:.3f} {np.mean(self.bad_0_1px):.3f} {np.mean(self.bad_0_5px):.3f} {np.mean(self.bad_1px):.3f} {np.mean(self.bad_3px):.3f} {np.mean(self.epe):.3f} {np.mean(self.rmse):.3f}\n")
            f.write("----end----\n")