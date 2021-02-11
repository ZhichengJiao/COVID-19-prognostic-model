# COVID-19-prognostic-model
# COVID-19-prognostic-model

## Prognostication of COVID-19 patients presenting to the emergency department

This repository is an easy-to-run prognostication of COVID-19 patients presenting to the emergency department demo.

## Prerequisites
PyTorch 1.3.0

scikit-survival 0.11

efficientnet_pytorch


### Usage
1. Use the **lung_field_segmentation.py** to obtain the lung mask of CXR image. The segmentation model is based on U-net and the parameter setting of this network is included in the file of **unet.py**.

2. Use the **image_based_critical_classification.py** to obtain the *critical vs. non-critical* classification results based on several CXR images of critical and non-critical patients.

2. Use the **visualization_cam.py** to obtain the visualized class activation map (CAM) of a critical patient. The visualization is achieved by CAM based on the pre-trained EfficientNet features and our critical classification network model.

3. Use the **time_to_event_prediction.py** to obtain the time-to-event risk estimation of patients based on their clinical measures or deep learning based image features. The risk prediction is based on the random survival forest model.



### Reference
[1] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.

[2] Tan, Mingxing, and Quoc Le. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." International Conference on Machine Learning. 2019.

[3] Zhou, Bolei, et al. "Learning deep features for discriminative localization." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[4] Ishwaran, Hemant, et al. "Random survival forests." The annals of applied statistics 2.3 (2008): 841-860.
