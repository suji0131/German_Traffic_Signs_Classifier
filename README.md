# German Traffic Signs Classifier

## Data Preprocessing

Data set was augmented using slight translations, rotations, zooming and any combination there off. A 34,800 dataset was augmented to 1,96,000 training data set. Each image has a 32x32x3 dimensions. To accelerate training every component in the image matrix is normalized (by dividing each value by 255) and its average is made 0 (simply by subtracting the average from each component). 

```
x_train = x_train/255
x_t_t = np.average(x_train, axis=0)
x_train = x_train - x_t_t
```
