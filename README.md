# SRCNN_Image_Restoration

## What is SRCNN ?
The SRCNN is a deep convolutional neural network that learns the end-to-end mapping of low-resolution to high-resolution images. Single image super resolution example has been tried to be created with Python-Keras and PyQt5. The goal of super-resolution (SR) is to recover a high-resolution image from a low-resolution input.
For more technical details, you can check out this resource:
https://appgenius.icu/abs/1501.00092)

**SRCNN consists of these stages**

* **Preprocessing & Feature Extraction:** High resolution images are converted into low resolution images. So, Up-scales LR image to desired HR size. Later extracts a set of feature maps from the up-scaled LR image.
* **Non-Linear Mapping:** Then it does mapping of LR and HR properties.
* **Reconstruction:** Produces the HR image from HR patches.

![image](https://miro.medium.com/max/875/1*mZJO-i6ImYyXHorv4H1q_Q.png)

## Requirements
```
pip install keras
pip install opencv-python
pip install PyQt5
```

* opencv -> This is necessary because the SRCNN network was trained on the luminance (Y) channel in the YCrCb color space.
* pyqt5 -> For GUI only
* keras -> For SRCNN. SRCNN training will not be done, 3051crop_weight_200.h5 weight can be used instead. Also thanks to [MarkPrecursor](https://github.com/MarkPrecursor/SRCNN-keras) for training the model. 

## Run
```
python main_GUI.py
```

**Test with any image in the low_resolution_images directory**

![image](https://github.com/fbasatemur/SRCNN_Image_Restoration/blob/master/doc/ss_monarch.jpg)


