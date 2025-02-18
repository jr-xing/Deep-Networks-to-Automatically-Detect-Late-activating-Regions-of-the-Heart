# Deep Networks to Automatically Detect Late-Activating Regions of the Heart
![nihms-1977483-f0001](https://github.com/user-attachments/assets/bd6fcc90-82b4-4378-af6c-65503c935586 | width=400)

This repository provides the PyTorch implementation of the paper [Deep Networks to Automatically Detect Late-activating Regions of the Heart](https://ieeexplore.ieee.org/document/9433796), which introduces a novel method for identifying late-activating regions of the left ventricle from cine Displacement Encoding with Stimulated Echo (DENSE) MR images. The proposed deep learning framework detects the Time to the Onset of circumferential Shortening (TOS) to identify late mechanical activation in heart failure patients.

## Abstract
This paper presents a novel method to automatically identify late-activating regions of the left ventricle from cine Displacement Encoding with Stimulated Echo (DENSE) MR images. We develop a deep learning framework that identifies late mechanical activation in heart failure patients by detecting the Time to the Onset of circumferential Shortening (TOS). In particular, we build a cascade network performing end-to-end:

1. **Segmentation** of the left ventricle to analyze cardiac function.
2. **Prediction** of TOS based on spatiotemporal circumferential strains computed from displacement maps.
3. **3D Visualization** of delayed activation maps.

Our approach results in dramatic savings of manual labor and computational time over traditional optimization-based algorithms. Experimental results demonstrate that the proposed approach provides fast prediction of TOS with improved accuracy.

## Setup

To set up the environment, ensure the following dependencies are installed:

- `matplotlib`
- `torch`
- `torchvision`
- `SimpleITK`
- `yaml`
- `numpy`

You can install the required packages using pip:

```bash
pip install matplotlib torch torchvision SimpleITK pyyaml numpy
```

## Dataset and Pre-trained Models

The dataset and pre-trained models will be made available in the future. Please check back for updates.

## Training

To train the model, use the provided training script. The script handles data loading, preprocessing, augmentation, and training the cascaded network.

```bash
python train.py
```

Key steps in the training process include:

1. Loading and preprocessing the data.
2. Performing train-test splits based on patient or slice indices.
3. Applying data augmentation techniques.
4. Training the deep learning model with configurable hyperparameters.

## Inference

To run inference on new data, use the inference script. This script loads a trained model and performs predictions on input MR images.

```bash
python inference.py
```

The inference script outputs:
- Predicted TOS values.
- 3D visualization of delayed activation maps.

## Citation

If you use this code or the associated paper in your research, please cite:

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

## License

This code is provided for **research purposes only** and is intended for **non-commercial use**. Please contact the authors for any commercial licensing inquiries.

## Contact

For questions or feedback, please open an issue on the repository or contact the authors directly.
