
# coding: utf-8

# In[1]:

# Load pickled data
import pickle
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle

training_file = 'train_aug.p'
validation_file= 'valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
x_train, y_train = train['features'], train['labels']
x_valid, y_valid = valid['features'], valid['labels']
x_test, y_test = test['features'], test['labels']


# In[2]:

assert len(x_train) == len(y_train)
assert len(x_valid) == len(y_valid)
assert len(x_test) == len(y_test)


# In[3]:

#Number of training examples
n_train = 34799

#Number of validation examples.
n_valid = 4410

#Number of testing examples.
n_test = 12630

#Shape of an traffic sign image
image_shape = np.shape(x_train[0])

#Unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))


# In[4]:

#each labels count in the training set
count_ = np.bincount(y_train)
unq_ = np.unique(y_train)


# In[6]:

#Preprocess the data here. Preprocessing steps include normalization
print(np.shape(x_train))
x_train = x_train/255
x_t_t = np.average(x_train, axis=0)
print(np.shape(x_t_t))
x_train = x_train - x_t_t
print(np.shape(x_train))

print(np.shape(x_valid))
x_valid = x_valid/255
x_t_v = np.average(x_valid, axis=0)
x_valid = x_valid - x_t_v
print(np.shape(x_valid))

x_test = x_test/255
x_t_te = np.average(x_test, axis=0)
x_test = x_test - x_t_te

# In[12]:

#model parameters
learning_rate = 0.001
epochs = 25
batch_size = 9000


# In[8]:

#weights and bias
#first convolution weight and bias
cnw_1 = tf.Variable(tf.truncated_normal([5,5,3,8], mean=0, stddev=0.1)) #stride 1
cnb_1 = tf.Variable(tf.zeros([8]))

#second convolution weight and bias
cnw_2 = tf.Variable(tf.truncated_normal([3,3,8,16], mean=0, stddev=0.1)) #stride 1
cnb_2 = tf.Variable(tf.zeros([16]))

#weights and bias for first inception module
weights_1 = {'icnw1': tf.Variable(tf.truncated_normal([1,1,16,32])),
          'icnw2': tf.Variable(tf.truncated_normal([1,1,16,20])),
          'icnw23': tf.Variable(tf.truncated_normal([3,3,20,24])),
          'icnw3': tf.Variable(tf.truncated_normal([1,1,16,20])),
          'icnw35': tf.Variable(tf.truncated_normal([5,5,20,24])),
          'icnwm1': tf.Variable(tf.truncated_normal([1,1,16,32]))}

bias_1 = {'icnb1': tf.Variable(tf.zeros([32])),
          'icnb2': tf.Variable(tf.zeros([20])),
          'icnb23': tf.Variable(tf.zeros([24])),
          'icnb3': tf.Variable(tf.zeros([20])),
          'icnb35': tf.Variable(tf.zeros([24])),
          'icnbm1': tf.Variable(tf.zeros([32]))}

#fully connected weights and bias
nnwts_1 = tf.Variable(tf.truncated_normal([1008, 86], mean=0, stddev=0.1))
nnb_1 = tf.Variable(tf.zeros([86]))

nnwts_2 = tf.Variable(tf.truncated_normal([86, 43], mean=0, stddev=0.1))
nnb_2 = tf.Variable(tf.zeros([43]))

# In[9]:

#NN layerincluding relu activation
def NN_lay(x, wts, bias):
    x = tf.add(tf.matmul(x, wts), bias)
    return tf.nn.relu(x)

#does a convolution and a relu activation
def conv_(x, wts, bias, strides=1, padding='VALID'):
    x = tf.nn.conv2d(x, wts, [1,strides,strides,1], padding)
    x = tf.nn.bias_add(x, bias)
    return tf.nn.relu(x)  

#maxpooling layer
def maxpool_(x, k=2, stride=1,padding='SAME'):
    return tf.nn.max_pool(x, [1,k,k,1], [1,stride,stride,1], 'SAME')    

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
    
    #a is for padding the lay_4 to bring its dimens to shape
    #a = np.array([[0,0],[pad_wt,pad_wt],[pad_wt,pad_wt],[0,0]])
    #a = tf.constant(a, dtype=tf.int32)
    #lay_4 = tf.pad(lay_4, a) #padded max_pool layer
    
    #concatenating the layers
    return concat_lay(lay_1, lay_2, lay_3, lay_4)


# In[11]:

x = tf.placeholder(tf.float32, [None,32,32,3])
y_ = tf.placeholder(tf.int32, [None])
y = tf.one_hot(y_, 43)

#first layer, a convolution layer
#inp: 32x32x3 to out: 28x28x8
logits = conv_(x, cnw_1, cnb_1, strides=1, padding='VALID')
#inp: 28x28x8 to out: 14x14x8
logits = tf.nn.max_pool(logits, [1,2,2,1], [1,2,2,1], 'VALID')

#second layer, a convolution layer
#inp: 14x14x8 to out: 12x12x16
logits = conv_(logits, cnw_2, cnb_2, strides=1, padding='VALID')
#inp: 12x12x16 to out: 6x6x16
logits = tf.nn.max_pool(logits, [1,2,2,1], [1,2,2,1], 'VALID')

#third layer, a inception module
#inp: 6x6x16 to out: 6x6x112
logits = inception(logits, weights_1, bias_1, 1, k=3)
logits = tf.nn.max_pool(logits, [1,2,2,1], [1,2,2,1], 'VALID')

#flatten the tensor: 3*3*112 = 1008
logits = flatten(logits)

#fully connected layer 1: 1008 to 86
logits = NN_lay(logits, nnwts_1, nnb_1)

#fully connected layer 2: 86 to 43
logits = tf.add(tf.matmul(logits, nnwts_2), nnb_2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf. global_variables_initializer()


# In[13]:

with tf.Session() as sess:
    sess.run(init)
    num_ex = len(x_train)
    
    no_of_batches = int(len(x_train)/batch_size)
    x_t, y_t = shuffle(x_train, y_train)
    for epoch in range(epochs):
        strt_tym = time.time()
        for offset in range(no_of_batches):
            idx = np.random.randint(0, high=len(x_t), size=batch_size)
            batch_x, batch_y = x_t[idx], y_t[idx]
            sess.run(optimizer, feed_dict={x:batch_x, y_:batch_y})
            loss = sess.run(cost, feed_dict={x:batch_x, y_:batch_y})
            val_acc = sess.run(accuracy, feed_dict={x: x_valid, y_: y_valid})
        end_tym = time.time()
        print('no of epoch: ',epoch)
        print('loss: ',loss)
        print('accuracy: ', val_acc)
        print('time in mins for an epoch: ', (end_tym-strt_tym)/60)
        print('/*****************************************************************/')
    
    print('test accuracy: ', sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))



