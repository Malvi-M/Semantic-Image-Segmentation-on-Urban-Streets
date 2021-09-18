# Semantic-Image-Segmentation-on-Urban-Streets

In this project, we have performed Semantic Image Segmentation on Urban Streets Dataset. Semantic Segmentation is one of the core element of complex system of Autonomous Driving. 

## Description

### Dataset Used

1. [Cityscapes Image Pairs Dataset](https://www.kaggle.com/dansbecker/cityscapes-image-pairs)<br/>
   This Cityscapes dataset is taken from Kaggle, which consists 2975 training images and 500 validation images.Cityscapes data contains labeled videos taken from vehicles driven in Germany. This version is a processed subsample created as part of the Pix2Pix paper. The dataset has still images from the original videos, and the semantic segmentation labels are shown in images alongside the original image.

### Libraries Used

* Numpy
* Pandas
* Matplotlib
* Sklearn
* Keras Tensorflow

### **Helper Functions**
For preprocessing the dataset and defining the model, we have defined several helper functions -


* `LoadImage` - Loads a single image and its corresponding segmentation map, also allows simple Data Augmentation options such as flipping and rotating 
    * **Arguements** :
        * `name` - Name of the image file
        * `path` - Path to the image directory
        * `rotation` - Angle by which the image will rotate for Data Augmentation
        * `flip` - True/False
        * `size` - size of image
    * **Returns** - A tuple of 2 numpy arrays (image and segmentation map)



---


* `ColorToClass` - Converts the discrete color representation (output of the color clustering) to a 13-dimensional class representation used for training our model
    * **Arguements** :
        * `seg` - Segmentation mask after clustering (width, height, 3) **(RGB)**
    * **Returns** - Categorical segmentation map (width, height, classes) 



---


* `LayersToRGBImage` - Converts the layer representation (categorical arrays) to a color representation for visualization 
    * **Arguements** :
        * `img` - Categorical segmented map (height, width, 13)
    * **Returns** - Colored segmentation map (width, height, 3) **(RGB)**


---


* `DataGenerator` - creates batches of e.g. 10 raw-segmented image pairs at a time, also uses image augmentation and randomly flips and rotates the images to increase the effective size of the dataset and returns data in form of batches  
    * **Arguements** :
        * `path` - location or path of the image directory
        * `batch_size` - size of each batch
        * `maxangle` -  angle to rotate image
    * **Returns** - Tuple of `batch_size` number of images and segmentation maps


---







* `fcn` - Creates the FCN model
    * **Arguements** :
        * `vgg` - VGG16 pretrained model
        * `classes` - Number of categories or unique pixel values (13)
        * `fcn8` - Set True to use FCN-8s model
        * `fcn16` - Set True to use FCN-16s model
    * **Returns** - FCN model
    * **Note** - If both `fcn8` and `fcn16` arguements are set to False, it returns FCN-32s model by default




---










## Results



## Contact Info

📧 [E-Mail](malvipatel1999@gmail.com) <br/>
🤝 [LinkedIn](https://www.linkedin.com/in/malvi-m)
