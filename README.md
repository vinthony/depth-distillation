# Defocus Blur Detection via Depth Distillation 

This repo contains the code and results of our ECCV 2020 paper:

<i><b>[Defocus Blur Detection via Depth Distillation](https://arxiv.org/abs/2007.08113)</b></i><br>
[_Xiaodong Cun_](http://vinthony.github.io) and [_Chi-Man Pun_<sup>*</sup>](http://www.cis.umac.mo/~cmpun/) <br>
[University of Macau](http://um.edu.mo/)


[Models](#pretrained-models) | [Results](#results) | [Paper](https://arxiv.org/abs/2007.08113) | [Supp.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580732-supp.pdf) | [Online Demo!(Google CoLab)](https://colab.research.google.com/drive/1a-Un_lZqkEN-mr-SzQh9GLy4qXIJgn0v#scrollTo=Lh2_NGuLaM_c)

![dd](https://user-images.githubusercontent.com/4397546/86791086-c1ac9b80-c09b-11ea-83cf-2f046bafa258.png)


### Results

we provide results on two datasets under different backbone(VGG19,ResNext101), please download from [Google Drive](https://drive.google.com/file/d/13nhzz9qaQ6U0F7Jsu4mLMS8XktZK1-Y_/view?usp=sharing)


### Pretrained Models

* Backbone: [VGG19](https://drive.google.com/file/d/1VigqrPdiIF18VALo92L9WCuASnpzu7qa/view?usp=sharing)
* Backbone: [ResNeXt101](https://drive.google.com/file/d/1HrzFqXSJY2Zxd9qvdKC7_Glxljjd27sf/view?usp=sharing)


### Dependences

* PyTorch
* OpenCV
* scipy
* tqdm
* scikit-learn


### Demos
[Online Demo!(Google CoLab)](https://colab.research.google.com/drive/1a-Un_lZqkEN-mr-SzQh9GLy4qXIJgn0v#scrollTo=Lh2_NGuLaM_c) is recommanded to evaluate the performance of our method.

Also, you can run a local jupyter server to evalute on CPU or GPU.

1. Download the [pretrianed models](#pretrained-models) and [ResNeXt101 backbone](https://drive.google.com/file/d/1o7zQvDef9mAEzbQeHAwMSi9nK_QEhhVZ/view?usp=sharing) and put it to `pretrained`.
2. Download the [DUT500 dataset](https://drive.google.com/file/d/1Qmcu6GDgKhiHVgLxeQg23tfy5I1Xg5Xk/view?usp=sharing) and put it to `dataset`

3. make sure all the path in `paths.py` are correct, the folder may like:

```
depth-distillation/
    - datasets/
        * DUTDefocus/
        * CUHKDefocus/
    - pretrained/
        * res_best.pth
        * vgg_best.pth
        * resnext_101_32x4d.pth
    - models/
    
    other files...
```

4. run the jupyter notebook to evaluate. 