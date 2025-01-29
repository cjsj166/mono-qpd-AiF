from evaluation.affine_invariant_metrics import *
import os.path as osp
from datetime import datetime

class Eval():
    def __init__(self, save_path=''):
        self.ai1s = []
        self.b1s = []
        self.ai2s = []
        self.b2s = []
        self.scs = []
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
    
    def save_metrics(self):
        with open(self.save_path, "w") as f:
            f.write("filename affine-invariant-1 ai1-scale ai1-bias affine-invariant-2 ai2-scale ai2-bias spearman-coefficient color-range\n")
            for filename, ai1, b1, ai2, b2, sc, rng in zip(self.filenames, self.ai1s, self.b1s, self.ai2s, self.b2s, self.scs, self.color_range):
                f.write(f"{filename} {ai1:.3f} {b1[0]:.3f} {b1[1]:.3f} {ai2:.3f} {b2[0]:.3f} {b2[1]:.3f} {sc:.3f} {rng[0]:.3f}-{rng[1]:.3f}\n")
            
            # write mean
            f.write(f"mean {np.mean(self.ai1s):.3f} {np.mean(self.b1s, axis=0)[0]:.3f} {np.mean(self.b1s, axis=0)[1]:.3f} {np.mean(self.ai2s):.3f} {np.mean(self.b2s, axis=0)[0]:.3f} {np.mean(self.b2s, axis=0)[1]:.3f} {np.mean(self.scs):.3f}\n")
            f.write("----end----\n")
            