# ShadowMover


This repository contains codes and model for our [paper](https://ieeexplore.ieee.org/document/10049677/):

    ShadowMover: Automatically Projecting Real Shadows onto Virtual Object


## Setup

1.Download the repo

2.Set up dependencies

    pip install -r requirements.txt

## Usage

1.Place input images including rgb of background, shadow of background, 
depth of model and normal of model in the input folders. 
Or use the toy images we provide.

2.Run the shifted shadow estimation model:

    python test.py

3.The shifted shadow maps are written to the folder 

    ./result


## Additional Usage

We also provide the code which transfers the estimated shifted shadow map into real shadows on the virtual object. 
You can find it in the folder

    ./compose

## Get Our Dataset

链接：https://pan.baidu.com/s/1SkzS_sz7e7peDprLfRYa2A?pwd=06wu 
提取码：06wu 
--来自百度网盘超级会员V6的分享
