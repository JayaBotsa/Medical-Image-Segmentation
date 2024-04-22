# Medical-Image-Segmentation
Detection and Segmentation of Intracranial Aneurysms using UNETR
# Introduction
This repository contains the code and experimental setup for a series of experiments conducted to determine the best model for segmenting aneurysms from a given dataset. Various patch sizes were tested with different algorithms, with the ultimate selection being a universally chosen patch size of 128×128×128 in order to cover most of the image space. The dataset was split into train-validation-test sets with a ratio of 75%-20%-25% to prevent overfitting and introduce independence in the model.

# Experimental Setup
Implementation Details
For both pretraining and fine-tuning, an Adam optimizer was utilized with an initial learning rate varying among different models and pretraining phases, typically ranging from 1e-3 to 1e-5. A constant decay rate of 1e-5 was employed, and a cosine scheduler was utilized for all models. The codebase was implemented using the MONAI library and PyTorch. Multi-GPU training was not utilized due to the capabilities of the available GPUs (Tesla A100 and Tesla V100-SXM2).

# Development Environment
The development environment used for this experiment is outlined in Table 5.1.1.

# Component	Version
Python	3.10.12
MONAI	v1.3
PyTorch	v2.1.0
CUDA	12.2

# Best Model Selection Standard
As the ground truth label is provided for the test dataset, two measures were employed to select the best model checkpoint:

**1. Dice Score:** The model was evaluated on the test dataset, and the Dice score was calculated by taking the average across all cases in the list.
**2. Jaccard Score:** Similarly, the model was evaluated on the test dataset, and the Jaccard score was calculated by averaging across all cases in the list.
Comparatively, the Jaccard score tends to be lower than the Dice score, as it is more restrictive and penalizes instances where there is no overlap between the ground truth and the predicted label.
