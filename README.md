# Medical-Image-Segmentation
Detection and Segmentation of Intracranial Aneurysms using UNETR

# Introduction
Localized enlargements of the arterial blood arteries caused by vascular weakening are known as Cerebral Aneurysms. Aneurysmal Subarachnoid Hemorrhage (aSAH), resulting from the rupture of the blood vessel, is a life-threatening condition with a high percentage of mortality and morbidity. It can lead to cognitive impairment and have long-term effects if not identified in the early stages. Accurate detection of aneurysms is crucial for treatment or rupture avoidance.

In order to address these challenges, early detection of aneurysms and the prevention of rupture are essential tasks for physicians. Developing and assessing a deep learning pipeline dedicated to automatically identifying and segmenting aneurysms can be immensely beneficial. Deep learning models such as Convolutional Neural Networks (CNNs) and Transformer architectures have shown advantages over manual methods, which are time-consuming, labor-intensive, and prone to human errors.

Objective
The aim of this thesis is to build a model capable of automatically detecting and segmenting intracranial aneurysms by following various pre- and post-processing steps. The model should be designed to work with limited data by incorporating different augmentations or utilizing pre-training techniques.

# Pre-training - Model Genesis
Transfer learning is a vital concept in deep learning for medical image analysis, yet adapting 2D solutions to 3D imaging tasks, like those in CT, MRI, and X-ray, often leads to a loss of rich anatomical information and diminished performance. To address this, Models Genesis employ self-supervised learning on unlabeled images, training an encoder-decoder using a sequence of algorithms. The training process involves altering patches with transformations and then teaching the model to restore the original patches. Models Genesis stand out due to their autodidactic nature, eclectic learning approach, scalability, and generic applicability across various tasks. They learn from multiple viewpoints, making them resilient for target tasks, and produce a common visual representation suitable for different applications. The pretraining pipeline includes data augmentation and regularized contrastive loss to learn feature representations of unlabeled data, enhancing the network's ability to reconstruct images and maximize agreement between augmented views. This approach offers a generic view of the process applicable to any application, enabling robust organ or lesion segmentations.

# Experimental Setup
This repository contains the code and experimental setup for a series of experiments conducted to determine the best model for segmenting aneurysms from a given dataset. Various patch sizes were tested with different algorithms, with the ultimate selection being a universally chosen patch size of 128×128×128 in order to cover most of the image space. The dataset was split into train-validation-test sets with a ratio of 75%-20%-25% to prevent overfitting and introduce independence in the model.

# Implementation Details
For both pretraining and fine-tuning, an Adam optimizer was utilized with an initial learning rate varying among different models and pretraining phases, typically ranging from 1e-3 to 1e-5. A constant decay rate of 1e-5 was employed, and a cosine scheduler was utilized for all models. The codebase was implemented using the MONAI library and PyTorch. Multi-GPU training was not utilized due to the capabilities of the available GPUs (Tesla A100 and Tesla V100-SXM2).

# Development Environment
The development environment used for this experiment is outlined in Table 5.1.1.

# Component	Version
- Python	3.10.12
- MONAI	v1.3
- PyTorch	v2.1.0
- CUDA	12.2

# Best Model Selection Standard
As the ground truth label is provided for the test dataset, two measures were employed to select the best model checkpoint:

**1. Dice Score:** The model was evaluated on the test dataset, and the Dice score was calculated by taking the average across all cases in the list.
**2. Jaccard Score:** Similarly, the model was evaluated on the test dataset, and the Jaccard score was calculated by averaging across all cases in the list.
Comparatively, the Jaccard score tends to be lower than the Dice score, as it is more restrictive and penalizes instances where there is no overlap between the ground truth and the predicted label.
