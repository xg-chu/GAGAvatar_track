<h1 align="center"><b>⚡️ Tracking Framework for GAGA ⚡️</b></h1>
<div align="center"> 
    <b>🚀 Track video 🚀</b> 
    <div align="center"> 
        <b><img src="./demos/track_obama.gif" alt="drawing" width="300"/></b>
    </div>
</div>
<div align="center"> 
    <b>🚅 Track image 🚅</b>
    <div align="center"> 
        <b><img src="./demos/track_monroe.jpg" alt="drawing" width="200"/></b>
    </div>
</div>


## Description
**GAGAvatar Track** is a monocular face tracker built on FLAME. It provides FLAME parameters (including **eyeball pose**) and camera parameters, along with the bounding box and landmarks used during optimization.

## Installation
### Build environment
Comming soon.
<!-- <details>
<summary><span >Install step by step</span></summary>

```
conda create -n track python=3.9
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
pip3 install mediapipe tqdm rich lmdb einops colored ninja av opencv-python scikit-image onnx2torch transformers pykalman face_alignment
```

</details>
<details>

<summary><span style="font-weight: bold;">Install with environment.yml (recommend)</span></summary>

```
conda env create -f environment.yml
```

</details> -->


### Build source
Check the ```build_resources.sh```.


## Fast start
*It takes longer to track the first picture.*

### Track on video:
```
python track_video.py -v ./demos/obama.mp4
```

### Track on one image:
```
python track_image.py -i ./demos/monroe.jpg
```
### Track all images in a lmdb dataset:
```
python track_lmdb.py -l ./demos/lmdb
```
