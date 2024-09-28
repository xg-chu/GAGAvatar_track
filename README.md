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
This environment is a sub-environment of **GAGAvatar**. You can skip this step if you have already built **GAGAvatar**.

```
conda env create -f environment.yml
```

### Build source
Check the ```build_resources.sh```.

*The models are available at https://huggingface.co/xg-chu/GAGAvatar_track.*


## Fast start
*It takes longer to track the first frame.*

#### Track on video(s):
```
python track_video.py -v ./demos/obama.mp4
```

#### Track on image(s):
```
python track_image.py -i ./demos/monroe.jpg
```
#### Track all images in a LMDB dataset:
```
python track_lmdb.py -l ./demos/vfhq_demo
```
