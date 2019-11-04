
# coding: utf-8

# In[1]:


from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# In[2]:


import os
import numpy as np
import tensorflow as tf
import argparse
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# In[3]:


# PARAMETER
VALIDATION_SIZE=406
max_iter= 300   
num_epoch=350
batch_size=123


# In[4]:


# READ THE DATASET
name=["data_batch_1","data_batch_2","data_batch_3","data_batch_4","data_batch_5","test_batch"]
dataset=np.empty(shape=(60000,32,32,3))
labels=np.empty(shape=(60000,1))
ind=0

# itero sui batch
for j in name:
    filename='./dataset/'+j
    df=unpickle(filename)
    data=df[b'data']
    lab=(df[b'labels'])
    
    # itero sui 10000 elementi di ogni batch
    for i in range(0,len(data)):
        # scompatto canali
        R=(data[i][0:1024]/255).reshape((32,32))
        G=(data[i][1024:2048]/255).reshape((32,32))
        B=(data[i][2048:]/255).reshape((32,32))
        img = np.dstack((R, G, B))
        # inserisco in posizione corretta
        indice=(ind*10000)+i
        dataset[indice]=img
        labels[indice]=lab[i]
    # aggiorno l'indice dei batch
    ind=ind+1


# In[5]:


dataset.shape


# In[7]:


labels.shape


# In[8]:


# LABEL ENCODER
enc = OneHotEncoder(sparse=False)
labelsOnehot=enc.fit_transform(labels)
print(labels[1])
print(labelsOnehot[1])


# In[9]:


# split data into training & validation
validation_images = dataset[:VALIDATION_SIZE,:,:,:]
validation_labels = labelsOnehot[:VALIDATION_SIZE,:]

test_images=dataset[-10000:,:,:,:]
test_labels=labelsOnehot[-10000:,:]

train_images = dataset[VALIDATION_SIZE:-10000,:,:,:]
train_labels = labelsOnehot[VALIDATION_SIZE:-10000,:]


print(train_images.shape)
print(validation_images.shape)
print(test_images.shape)


# In[10]:


epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]
def next_batch(batch_size):
    
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
        # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]


# In[11]:


# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# In[12]:


# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# In[13]:


# max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# In[14]:


# input image
x = tf.placeholder('float', shape=[None, 32, 32, 3]) # image_size
# labels
y_ = tf.placeholder('float', shape=[None, 10]) #labels_count


# In[15]:


# First convolutional layer - maps one grayscale image to 32 feature maps.
with tf.name_scope('conv1'):
  W_conv1 = weight_variable([5, 5, 3, 16])
  b_conv1 = bias_variable([16])
  h_conv1 = tf.nn.tanh(conv2d(x, W_conv1) + b_conv1) #ORIGINARIAMENTE RELU 

# Pooling layer - downsamples by 2X.
with tf.name_scope('pool1'):
  h_pool1 = max_pool_2x2(h_conv1)

# Residual Block   
with tf.name_scope('conv2'):
  W_conv2 = weight_variable([3, 3, 16, 16])
  b_conv2 = bias_variable([16])
with tf.name_scope('conv3'):
  W_conv3 = weight_variable([3, 3, 16, 4])
  b_conv3 = bias_variable([4])

  W_fc1 = weight_variable([8 * 8 * 16, 500])
  b_fc1 = bias_variable([500])
    
  W_fc2 = weight_variable([500, 50])
  b_fc2 = bias_variable([50])

  W_fc3 = weight_variable([500, 10])
  b_fc3 = bias_variable([10])

# FRPN loop
def convergence(pooling, stato, stato_vecchio, k, y_con, c):


      h_conv2 = tf.nn.tanh(conv2d(h_pool1, W_conv2) + b_conv2) 

  # CONCATENATE STATE OND POOLING OUTPUT
      h_concat=tf.concat(axis=3,values=[pooling,stato])
  
  # CONVOLUTION=STATE
      h_conv3 = tf.nn.tanh(conv2d(h_conv2, W_conv3) + b_conv3)
      h_conv3r=tf.reshape(h_conv3,[batch_size,16,16,4])
    
  # RESHAPE TO HAVE A VECTOR
      h_pool2_flat = tf.reshape(h_conv3, [-1, 16*16*4])
  
  # FIRST FULLY CONNECTED
      h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  
  # SOFTMAX 
      y_con = tf.reshape(tf.nn.softmax(tf.matmul(h_fc1, W_fc3) + b_fc3),[batch_size,10])

  # ENTROPY
      c=tf.abs(tf.cast(tf.reduce_sum(tf.multiply(y_con,tf.log(y_con))),dtype=tf.float32))
  
  # INCREMENT COUNTER
      k=k+1
      return pooling, h_conv3r, stato, k, y_con, c
def condition(a, stato, stato_vecchio, k, y_con, c):
  # EVALUATE ENTROPY
   outDistance = tf.abs(tf.cast(tf.reduce_sum(tf.multiply(y_con,tf.log(y_con))),dtype=tf.float32))
  
  # CONDITION
   c1 = tf.greater(outDistance, 0.5)
   c2 = tf.less(k, max_iter) 
   c3 = tf.equal(k,0)
   con1=tf.logical_and(c1,c2)
   return tf.logical_or(con1,c3)

  # INITIALIZE THE NUMBER OF STEP
k=tf.constant(0,name="k")

ccc=tf.Variable(tf.zeros(shape=()), dtype=tf.float32)
  
  # INITIALIZE STATE, PREVIOUS STATE AND OUTPUT
state  = tf.Variable(tf.random_normal([batch_size,16, 16, 4], mean=0, stddev=1))
state_old = tf.Variable(tf.random_normal([batch_size,16, 16, 4], mean=0, stddev=1))
y_conv=tf.Variable(tf.ones([batch_size,10]))


  # WHILE LOOP FOR THE RECURSION
res, st, old_st, num, y_conv, c = tf.while_loop(condition, convergence, [h_pool1, state, state_old, k, y_conv,ccc])



with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), 1))
    
with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1),tf.argmax(y_ ,1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.histogram('accuracy', accuracy)
         
with tf.Session() as sess:
    prediction = []
    loss_ = []
    cicli = []
    entropy= []
    valid=[]
    sess.run(tf.global_variables_initializer())
    acc_val=0
    for i in range(0,num_epoch):
        total_batch = int(len(train_images)/batch_size)
        train_accuracy=0
        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(batch_size)  
            _, acc, loss, num_cicle= sess.run([train_step, accuracy, cross_entropy, num], feed_dict={x:batch_xs, y_:batch_ys})
            prediction.append(acc)
            loss_.append(loss)
            cicli.append(num_cicle)
            train_accuracy+=acc 
        print(str(i)+"/"+str(total_batch)+" train_accuracy="+str(train_accuracy/i)+" num_cicle="+str(num_cicle)+" loss="+str(loss)+'\r')
        
        # Evaluate on validation set
        total_batch = int(len(validation_images)/batch_size)
        acc=0
        for i in range(total_batch):
            batch_xs, batch_ys = validation_images[0:batch_size], validation_labels[0:batch_size]
            acc+=accuracy.eval(feed_dict={x:batch_xs, y_:batch_ys})
        print("validation_accuracy:",acc/total_batch)
        valid.append(acc/total_batch)
        
            
    # evaluate on test_set
    predic=[]
    cicle_test=[]
    total_batch = int(len(test_images)/batch_size)
    print(total_batch)
    acc=0
    for i in range(total_batch):
        batch_xs, batch_ys = test_images[0:batch_size], test_labels[0:batch_size]
        accu, pr, num_cicle= sess.run([accuracy, correct_prediction, num], feed_dict={x:batch_xs, y_:batch_ys})
        acc+=accu
        predic.append(pr)
        cicle_test.append(num_cicle)
    print("test_accuracy:",acc/total_batch)

## DATI RELATIVI AL TRAINING SET
d1=pd.DataFrame(prediction)
d2=pd.DataFrame(loss_)
d3=pd.DataFrame(cicli)
d4=pd.DataFrame(entropy)
d5=pd.DataFrame(valid)
d1.to_csv("accuracyres.csv", index=False)
d2.to_csv("lossres.csv", index=False)
d3.to_csv("ciclires.csv", index=False)
d4.to_csv("entropyres.csv", index=False)
d5.to_csv("cicli_testres.csv", index=False)
d11=pd.DataFrame(predic)
d22=pd.DataFrame(cicle_test)
d11.to_csv("predictionres.csv", index=False)
d22.to_csv("cicli_testres.csv", index=False)
d23.to_csv("validation.csv", index=False)
    

