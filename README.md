# Deep Networks to Automatically Detect Late activating Regions of the Heart
This repository provides the PyTorch implementation of the paper, [Deep Networks to Automatically Detect Late-activating Regions of the Heart](https://ieeexplore.ieee.org/document/9433796)

## Abstract

This paper presents a novel method to automatically identify late-activating regions of the left ventricle from cine Displacement Encoding with Stimulated Echo (DENSE) MR images. We develop a deep learning framework that identifies late mechanical activation in heart failure patients by detecting the Time to the Onset of circumferential Shortening (TOS). In particular, we build a cascade network performing end-to-end (i) segmentation of the left ventricle to analyze cardiac function, (ii) prediction of TOS based on spatiotemporal circumferential strains computed from displacement maps, and (iii) 3D visualization of delayed activation maps. Our approach results in dramatic savings of manual labors and computational time over traditional optimization-based algorithms. To evaluate the effectiveness of our method, we run tests on cardiac images and compare with recent related works. Experimental results show that the proposed approach provides fast prediction of TOS with improved accuracy.

## Setup
* matplotlib
* torch
* torchvision
* SimpleITK
* yaml
* numpy

## Dataset and pre-trained models
* to be constructed

## Training
* to be constructed
## Testing
* to be constructed

## References
This code is only for research purposes and non-commercial use only, and we request you to cite our research paper if you use it:
[Deep Networks to Automatically Detect Late-activating Regions of the Heart](https://ieeexplore.ieee.org/document/9433796)

```bibtex
@inproceedings{xing2021deep,
  title={Deep networks to automatically detect late-activating regions of the heart},
  author={Xing, Jiarui and Ghadimi, Sona and Abdi, Mohamad and Bilchick, Kenneth C and Epstein, Frederick H and Zhang, Miaomiao},
  booktitle={2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI)},
  pages={1902--1906},
  year={2021},
  organization={IEEE}
}
```
