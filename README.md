# Neural-Style-Transfer-keras

Keras implementation of the paper [Image style transfer using convolutional neural networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.

**IN PROGRESS...**

## Dependencies

* keras
* numpy
* scipy
* matplotlib

## Usage

### 1. Install requirements

```bash
$ pip3 install -r requirements.txt
```

### 2. Generate custom datasets

The `dataset` directory should look like:

```bash
datasets
├── content
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── style_reference
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

Image file name and extension aren't limited.

### 3. Train a model

Run `train.py` and train a model with images you want to synthesize.

```bash
$ python train.py --content_img='cat.jpg'\
				  --style_img='the_scream.jpg'\
				  --img_height=512\
				  --num_iterations=300\
				  --c_weight=0.05\
				  --s_weight=1\
				  --c_layer=5\
				  --s_layer=[1,2,3,4,5]
```

## To do

- [ ] [Image style transfer using convolutional neural networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html)
- [ ] [Controlling perceptual factors in neural style transfer](http://openaccess.thecvf.com/content_cvpr_2017/papers/Gatys_Controlling_Perceptual_Factors_CVPR_2017_paper.pdf)
- [ ] [Preserving color in neural artistic style transfer](https://arxiv.org/abs/1606.05897)

## Author

Daeyoung Kim / [@cyc1am3n](https://github.com/cyc1am3n)