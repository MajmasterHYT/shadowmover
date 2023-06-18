# ShadowMover


This repository contains code and models for our [paper](https://ieeexplore.ieee.org/document/10049677/):

    ShadowMover: Automatically Projecting Real Shadows onto Virtual Object


## Setup

1.Download the repo

2.Set up dependencies

    pip install -r requirements.txt

## Usage

1.Place input images including rgb of background, shadow of background, 
depth of model and normal of model in the input folders. 
Or use the toy images we provide.

2.Run a shitfed shadow estimation model:

    python test.py

3.The shifted shadow maps are written to the folder 

    ./result


## Additional Usage

We also provide the code which transfers shifted shadow map into real shadow on the virtual object. 
You can find it in the folder

    ./compose


