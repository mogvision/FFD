
# FFD: Fast Feature Detector

## Introduction
FFD is a fast scale-invariant feature detector for computer vision tasks. This repo includes the code for keypoint detection from images. Given a pair of images, you can use this repo to extract matching features across the image pair.



* Full paper PDF: [FFD: Fast Feature Detector](https://arxiv.org/pdf/2012.00859.pdf).

* Authors: Morteza Ghahremani, Yonghuai Liu and Bernard Tiddeman



## Dependencies
* Python 3 >= 3.5
* OpenCV >= 3.4 
    (tested on `opencv-python==3.4.11.45` & `opencv-contrib-python==3.4.11.45`)
* NumPy >= 1.18


## Contents
There are two main scripts in this repo:

1. `demo_FFD.py`: runs and shows extracted keypoints from images located in `image/`
2. `match_pairs.py`: reads an image pair from `image/` and matches (SIFT descriptor is used for feature description)

```sh
python3 demo_FFD.py
python3 match_pairs.py
```
P.S. If you get error: "./FFD: Permission denied", please just run `chmod 777 FFD'.


## BibTeX Citation
If you use any ideas from the paper or code from this repo, please consider citing:

```txt
@ARTICLE{9292438,
  author={M. {Ghahremani} and Y. {Liu} and B. {Tiddeman}},
  journal={IEEE Transactions on Image Processing}, 
  title={FFD: Fast Feature Detector}, 
  year={2020},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TIP.2020.3042057}}

```
