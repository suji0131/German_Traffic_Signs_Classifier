# German Traffic Signs Classifier

## Data Summary
Each image has a 32x32x3 dimensions. Training set has 34,799 images, validation set has 4,410 images and testing set has 12,630 images. There are 43 different types of traffic sign images (So, our output layer should have 43 neurons). Each class (type) has different number of samples. Sample traffic sign image looks like below: (insert a sample image and no of samples for each class graph)
![Image of Sample](https://github.com/suji0131/German_Traffic_Signs_Classifier/blob/master/extras/sample.png)

## Data Preprocessing
Data set was augmented using slight translations, rotations, zooming and any combination there off. A 34,800 dataset was augmented to 1,96,000 training data set. To accelerate training every component in the image matrix is normalized (by dividing each value by 255) and its average is made 0 (simply by subtracting the average from each component). 
```
x_train = x_train/255
x_t_t = np.average(x_train, axis=0)
x_train = x_train - x_t_t
```

## Architecture
Architecture consists of two convolution layers, an Inception module (more on this later), max pool layers and fully connected neural network. First convolution layer has a filter size of 5,5,3,8 and relu activation function. The output of first convolution is then fed to a max pool layer of kernel size 2x2 and a stride of 2x2. This is then fed to another convolution layer with filter size of 3,3,8,16 stride of 1 and relu activation function. This is then forwarded to another max pool layer of kernel size 2x2 and stride 2x2. This is used as an input to inception module. Inception module out put is then fed to max pool layer and then to a fully connected neural network that has three layers, except for output layer (which has sigmoid activation) other layers has relu activation. Input and output dimension for each layer is depicted in below image: 
![Image of Architecture](https://github.com/suji0131/German_Traffic_Signs_Classifier/blob/master/extras/Architecture.png)
