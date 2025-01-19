import torch
from collections import OrderedDict

if __name__ == '__main__':
    state_dict = torch.load('mono_qpd/QPDNet/checkpoints/checkpoints-CLR-ddp.pth')
    state_dict = state_dict['model_state_dict']

    fixed_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # print(k, v.shape)
        print(k)
        if 'module.' in k:
            k = k.replace('module.', '')
        fixed_state_dict[k] = v

    torch.save(fixed_state_dict, 'mono_qpd/QPDNet/checkpoints/checkpoints-CLR.pth')
            