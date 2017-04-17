# German Traffic Signs Classifier

## Data Summary
Each image has a 32x32x3 dimensions. Training set has 34,799 images, validation set has 4,410 images and testing set has 12,630 images. There are 43 different types of traffic sign images (So, our output layer should have 43 neurons). Each class (type) has different number of samples. Sample traffic sign image looks like below:

![Image of Sample](https://github.com/suji0131/German_Traffic_Signs_Classifier/blob/master/extras/sample.png)

## Data Preprocessing
Data set was augmented using slight translations, rotations, zooming and any combination thereof. A 34,800 dataset was augmented to 1,96,000 training data set. To accelerate training every component in the image matrix is normalized (by dividing each value by 255) and its average is made 0 (simply by subtracting the average from each component). 
```
x_train = x_train/255 #normalization
x_t_t = np.average(x_train, axis=0)
x_train = x_train - x_t_t #making mean=0
```

## Architecture
###### Overall Architecture

![Image of Architecture](https://github.com/suji0131/German_Traffic_Signs_Classifier/blob/master/extras/Architecture.png)

Architecture consists of two convolution layers, an Inception module (more on this later), max pool layers and fully connected neural network. First convolution layer has a filter size of 5,5,3,8 and relu activation function. The output of first convolution is then fed to a max pool layer of kernel size 2x2 and a stride of 2x2. This is then fed to another convolution layer with filter size of 3,3,8,16 stride of 1 and relu activation function. This is then forwarded to another max pool layer of kernel size 2x2 and stride 2x2. This is used as an input to inception module. Inception module out put is then fed to max pool layer and then to a fully connected neural network that has three layers, except for output layer (which has sigmoid activation) other layers has relu activation. Input and output dimension for each layer is depicted in above image: 

###### Inception Module

![Image of Inception](https://github.com/suji0131/German_Traffic_Signs_Classifier/blob/master/extras/Inception.png)

An inception module has convolutions connected in parallel way. Input is fed to four different parallel lanes. First lane has 1x1 filter size convolution, second lane has 1x1 convolution and a 3x3 convolution, third lane has 1x1 convolution and 5x5 convolution and finally fourth layer has max pooling layer and a 1x1 convolution. Every convolution has relu activation, stride of 1 in every dimension and padding is set as SAME. Output from each lane is concatenated along last axis, -1. Concatenating means stacking every output tensors along a particular axis and care must be taken to keep dimensions in every other axis same.
(Inception is first implemented in GoogLeNet you can read more about it [here](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf))
```
#concatenates the four parallel layers of inception module 
#each layers passed should have same dimens except for last axis(-1)
def concat_lay(lay_1, lay_2, lay_3, lay_4, axis=-1):
    lay_1 = tf.unstack(lay_1, axis=axis) #this creates a list unpacked along last axis(-1)
    
    lay_1.extend(tf.unstack(lay_2, axis=axis)) #extending the previous list
    lay_1.extend(tf.unstack(lay_3, axis=axis))
    lay_1.extend(tf.unstack(lay_4, axis=axis))
    
    return tf.stack(lay_1, axis=axis) #re stacking the layers along the last axis(-1)

#inception modules takes x, weights, bias etc.
#pad_wt is a param for padding to be done for max_pool layer
def inception(x, iweights, ibias, pad_wt,strides=1, padding = 'SAME', k=2):
    #1-1 filter conv layer
    lay_1 = conv_(x, iweights['icnw1'],ibias['icnb1'], strides=strides, padding = padding)
    
    #1-1 filter conv layer and then 3-3 conv layer
    lay_2 = conv_(x, iweights['icnw2'],ibias['icnb2'], strides=strides, padding = padding)
    lay_2 = conv_(lay_2, iweights['icnw23'],ibias['icnb23'], strides=strides, padding = padding)
    
    #1-1 filter conv layer and then 5-5 conv layer
    lay_3 = conv_(x, iweights['icnw3'],ibias['icnb3'], strides=strides, padding = padding)
    lay_3 = conv_(lay_3, iweights['icnw35'],ibias['icnb35'], strides=strides, padding = padding)
    
    #max_pool layer and then a 1-1 conv layer
    lay_4 = maxpool_(x, k=k, padding=padding)
    lay_4 = conv_(lay_4, iweights['icnwm1'],ibias['icnbm1'], strides=strides, padding = padding)
    
    #concatenating the layers
    return concat_lay(lay_1, lay_2, lay_3, lay_4)
```

## Model Training
Adam optimizer was used to minimize the cross entropy loss. A learning rate of 0.001, batch size of 9000 is used for a total of 65 epochs. Nine thousand random integers are generated between 0 and 1,96,000 and data at these indices are used as a batch. Training was done on google cloud with 2 vCPUs with 13GB memory and a nVidia Tesla K80 GPU. Training time for each epoch is approximately 22 seconds. 

## Results and Analysis
Validation accuracy of 99 percent was achieved and trained model has top five accuracy of 98 percent on test set. Zoomed in version of the loss (y-axis) vs iteration (x-axis) looks like below. Both training loss and validation loss are smooth which implies that learning rate is good. Training loss didn't increase after steadily decreasing so it can be inferred that there is no over-fitting.  
  
![Image of Loss](https://github.com/suji0131/German_Traffic_Signs_Classifier/blob/master/extras/zoom_graph.png)

Note: 1 epoch = 21 iterations

###### Testing on real images
![Image of Loss](https://github.com/suji0131/German_Traffic_Signs_Classifier/blob/master/New_imgs/50_limit_2.jpg) ![Image of Loss](https://github.com/suji0131/German_Traffic_Signs_Classifier/blob/master/New_imgs/end_32.jpg) ![Image of Loss](https://github.com/suji0131/German_Traffic_Signs_Classifier/blob/master/New_imgs/keeprt_38.jpg) ![Image of Loss](https://github.com/suji0131/German_Traffic_Signs_Classifier/blob/master/New_imgs/priority_12.jpg) ![Image of Loss](https://github.com/suji0131/German_Traffic_Signs_Classifier/blob/master/New_imgs/road_25.jpg)

Five German traffic sign images (displayed above) are pulled from the web and converted as data. All five signs are centered and compressed to 32x32 pixel size. There is a good chance most will be classified correctly expect for speed limit 50 sign as the data set has nine speed limit signs that look similiar except for numbers. When model was run on these new images four of them are correctly classified. Error rate is less for classes which have atleast six thousand data points. To further improve the model data points for every class should be augmented to atleast five thousand threshold.
