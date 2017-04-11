# German Traffic Signs Classifier

## Data Summary
Each image has a 32x32x3 dimensions. Training set has 34,799 images, validation set has 4,410 images and testing set has 12,630 images. There are 43 different types of traffic sign images (So, our output layer should have 43 neurons). Each class (type) has different number of samples. Sample traffic sign image looks like below:

![Image of Sample](https://github.com/suji0131/German_Traffic_Signs_Classifier/blob/master/extras/sample.png)

## Data Preprocessing
Data set was augmented using slight translations, rotations, zooming and any combination thereof. A 34,800 dataset was augmented to 1,96,000 training data set. To accelerate training every component in the image matrix is normalized (by dividing each value by 255) and its average is made 0 (simply by subtracting the average from each component). 
```
x_train = x_train/255
x_t_t = np.average(x_train, axis=0)
x_train = x_train - x_t_t
```

## Architecture
###### Overall Architecture

![Image of Architecture](https://github.com/suji0131/German_Traffic_Signs_Classifier/blob/master/extras/Architecture.png)

Architecture consists of two convolution layers, an Inception module (more on this later), max pool layers and fully connected neural network. First convolution layer has a filter size of 5,5,3,8 and relu activation function. The output of first convolution is then fed to a max pool layer of kernel size 2x2 and a stride of 2x2. This is then fed to another convolution layer with filter size of 3,3,8,16 stride of 1 and relu activation function. This is then forwarded to another max pool layer of kernel size 2x2 and stride 2x2. This is used as an input to inception module. Inception module out put is then fed to max pool layer and then to a fully connected neural network that has three layers, except for output layer (which has sigmoid activation) other layers has relu activation. Input and output dimension for each layer is depicted in below image: 

###### Inception Module

![Image of Inception](https://github.com/suji0131/German_Traffic_Signs_Classifier/blob/master/extras/Inception.png)

An inception module has convolutions connected in parallel way. Input is fed to four different parallel lanes. First lane has 1x1 filter size convolution, second lane has 1x1 convolution and a 3x3 convolution, third lane has 1x1 convolution and 5x5 convolution and finally fourth layer has max pooling layer and a 1x1 convolution. Every convolution has relu activation, stride of 1 in every dimension and padding is set as SAME. Output from each lane is concatenated along last axis, -1. Concatenating means stacking every output tensors along a particular axis and care must be taken to keep dimensions in every other axis same.
(Inception is first implemented in GoogLeNet you can read more about it [here](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf))

## Model Training
Adam optimizer was used to minimize the cross entropy loss. A learning rate of 0.001, batch size of 9000 is used for a total of 65 epochs. Nine thousand random integers are generated between 0 and 1,96,000 and data at these indices are used as a batch. Training was done on google cloud with 2 vCPUs with 13GB memory and a nVidia Tesla K80 GPU. Training time for each epoch is approximately 22 seconds. 

## Results
Validation accuracy of 99 percent was achieved and trained model has top five accuracy of 98 percent on test set.  
