import torch
from collections import OrderedDict



if __name__ == '__main__':
    ckpt_path = 'result/checkpoints/20_epoch_30101_Mono-QPD.pth'
    save_path = ckpt_path.replace('.pth', '-fixed.pth')

    state_dict = torch.load(ckpt_path)
    state_dict = state_dict['model_state_dict']

    fixed_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # print(k, v.shape)
        print(k)
        if 'module.' in k:
            k = k.replace('module.', '')
        fixed_state_dict[k] = v

    torch.save(fixed_state_dict, save_path)
            