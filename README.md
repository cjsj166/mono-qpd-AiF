# Disparity Estimation Using a Quad-Pixel Sensor

Source code and synthetic dataset for our paper:

[Disparity Estimation Using a Quad-Pixel Sensor]()<br/>
BMVC 2024<br/>
Zhuofeng Wu, Doehyung Lee, Zihua Liu, Kazunori Yoshizaki, Yusuke Monno, Masatoshi Okutomi<br/>

```
@inproceedings{wu2024qpdnet,
  title={Disparity Estimation Using a Quad-Pixel Sensor},
  author={Zhuofeng Wu, Doehyung Lee, Zihua Liu, Kazunori Yoshizaki, Yusuke Monno, Masatoshi Okutomi},
  booktitle={The 35th British Machine Vision Conference (BMVC)},
  year={2024}
}
```

## Requirements
The code has been tested with PyTorch 1.11 and Cuda 11.3
```Shell
conda env create -f env.yaml
conda activate qpdnet
```
## Our Synthetic Data
Our synthetic dataset was generated by the [recurrent-defocus-deblurring-synth-dual-pixel](https://github.com/Abdullah-Abuolaim/recurrent-defocus-deblurring-synth-dual-pixel).<br/>
The dataset can be downloaded from [QP-data.zip]()


## Training

### Training with full Quad-pixel data (Left-Center-Right and Top-Center-Bottom):
```Shell
python train_quad.py --batch_size 4 --spatial_scale -0.2 0.4 --saturation_range 0 1.4  --mixed_precision  --datasets_path "training data path"
```

### Training with half Quad-pixel data (Left-Center-Right) (equal to Dual-pixel data):
```Shell
python train_quad.py --batch_size 4  --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --mixed_precision  --input_image_num 2 --datasets_path "training data path"
```

## Evaluation
### Checkpoints
Pretrained models (full Quad-pixel data) can be downloaded from [google drive](https://drive.google.com/file/d/1KGXLY_tZAyM9-e7jDV53Fu3e2m4knMNO/view?usp=drive_link).

### Evaluate with full Quad-pixel data (Left-Center-Right and Top-Center-Bottom):
```Shell
python evaluate_quad.py --restore_ckpt "checkpoint path" --mixed_precision  --save_result True --datasets_path "testing data path"
```


### Evaluate with half Quad-pixel data (Left-Center-Right) (equal to Dual-pixel data):
```Shell
python evaluate_quad.py --restore_ckpt "checkpoint path" --mixed_precision  --save_result True --input_image_num 2 --datasets_path "testing data path"
```

## Acknowledgement

This project uses the following open-source projects and data. Please consider citing them if you use related functionalities.


* [RAFT-stereo (Lipson et al., 3DV 2021)](https://github.com/princeton-vl/RAFT-Stereo)
* [Abuolaim et al., ICCV 2021](https://github.com/Abdullah-Abuolaim/recurrent-defocus-deblurring-synth-dual-pixel)

