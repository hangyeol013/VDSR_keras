# VDSR_keras


A implementation of the original paper ['Accelerating Image Super-Resolution Using Very Deep Convolutional Networks'](https://cv.snu.ac.kr/research/VDSR/)



<center><img width = "800" src="https://user-images.githubusercontent.com/58276840/95052310-aba98800-0729-11eb-9c45-a520eb98b3ed.png"></center>



tensorflow-gpu 2.0.0, keras 2.3.1 based implementation on Python 3.6.9, using Jupyter Notebook.




### Implementation
-------------------------------------------------------
My implementation may have some differences with the original paper:


#### Networks)

- 20 layers where layers except the first and the last are of the same type: 64 filters of ths size 3 x 3 x 64.
(The first layer and the last layer consists of a single filter of size 3 x 3 x 64)
- The network takes an interpolated low-resolution image (to the desired size) as input and predicts image details.
- Zero-padding for all layers


#### Training)

- Loss Function: MSE (Mean Squared Error)
- Optimizer: Adam
- Learning rate: 0.01 (Decreased by a factor of 10 every 20 epochs) 
- Batch size: 64


#### Dataset)

##### Training)
- 291 images (91 images from Yang et al. 200 images from Berkeley Segmentation Dataset) (Training set)
- Another 20 images from the validation set of the BSD500 dataset (Validation set)
- Data augmentation (scaling:0.6, 0.8, Rotation: 90, 180, 270, flip: vertical, horizontal): 35 times more images for training
- Patch size: 41 (Input), 41 (Label)


##### Test)
- Set5, Set14, BSD200, Urban100



### Use
-------------------------------------------------------

You can generate dataset (training sample, test sample) through matlab files in Dataset_Generating directory
- Excute for data augmentation: `VDSR_data_aug.m`
- Excute for making LR images: `Generate_LR.m`
- Excute for training sample: `VDSR_generate_train.m`
- Excute for test sample: `VDSR_generate_test.m`


I also uploaded the trained weight files.

With VDSR_main.ipynb file and weight file in 'weight', you can test the network for all scales (x2, x3, x4).
