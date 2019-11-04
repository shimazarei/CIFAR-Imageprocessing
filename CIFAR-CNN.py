
# coding: utf-8

# In[1]:


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# In[2]:


VALIDATION_SIZE=5000
num_epoch=10
batch_size=2


# In[3]:


#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from skimage.io import imsave
import tensorflow as tf
import tempfile
from sklearn.preprocessing import OneHotEncoder


# In[4]:


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
        R=data[i][0:1024].reshape((32,32))
        G=data[i][1024:2048].reshape((32,32))
        B=data[i][2048:].reshape((32,32))
        img = np.dstack((R, G, B))
        # inserisco in posizione corretta
        indice=(ind*10000)+i
        dataset[indice]=img
        labels[indice]=lab[i]
    # aggiorno l'indice dei batch
    ind=ind+1


# In[5]:


dataset.shape


# In[6]:


plt.axis('off')
plt.imshow(dataset[0])


# In[7]:


enc = OneHotEncoder(sparse=False)
labelsOnehot=enc.fit_transform(labels)
print(labels[1])
print(labelsOnehot[1])


# In[8]:


# split data into training & validation
validation_images = dataset[:VALIDATION_SIZE]
validation_labels = labelsOnehot[:VALIDATION_SIZE]

test_images=dataset[-10000:]
test_labels=labelsOnehot[-10000:]

train_images = dataset[VALIDATION_SIZE:-10000]
train_labels = labelsOnehot[VALIDATION_SIZE:-10000]


print('train_images({0[0]},{0[1]})'.format(train_images.shape))
print('validation_images({0[0]},{0[1]})'.format(validation_images.shape))
print('test_images({0[0]},{0[1]})'.format(test_images.shape))


# In[9]:


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
    if index_in_epoch > num_examples:
        epochs_completed += 1
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch   
    return train_images[start:end], train_labels[start:end]    



# In[10]:


print(labels[1])
print(labelsOnehot[1])


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


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# In[14]:


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

# Second convolutional layer -- maps 32 feature maps to 64.
with tf.name_scope('conv2'):
  W_conv2 = weight_variable([5, 5, 16, 4])
  b_conv2 = bias_variable([4])
  h_conv2 = tf.nn.tanh(conv2d(h_pool1, W_conv2) + b_conv2)



# Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
# is down to 7x7x64 feature maps -- maps this to 1024 features.
with tf.name_scope('fc1'):
  W_fc1 = weight_variable([16 * 16 * 4, 500])
  b_fc1 = bias_variable([500])

  h_pool2_flat = tf.reshape(h_conv2, [-1, 16*16*4])
  h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# Map the 1024 features to 10 classes, one for each digit
with tf.name_scope('fc2'):
  W_fc2 = weight_variable([500, 10])
  b_fc2 = bias_variable([10])
  y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2


# In[ ]:


with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
    cross_entropy = tf.reduce_mean(cross_entropy)

with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

graph_location = tempfile.mkdtemp()
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for j in range(0,num_epoch):
        total_batch = int(len(train_images)/batch_size)
        train_accuracy=0
        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(batch_size)
            _, Accuracy, crossEntropy = sess.run([train_step,accuracy,cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
        
            train_accuracy+= Accuracy
        print(str(i)+" acc="+str(train_accuracy/i)+" loss="+str(crossEntropy)+'\r')
        # Evaluate on validation set
        total_batch = int(len(validation_images)/batch_size)
        acc=0
        for i in range(total_batch):
            batch_xs, batch_ys = validation_images[i].reshape([1,32,32,3]), validation_labels[i].reshape([1,10])
            acc+=accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
        print("validation_accuracy:",acc/total_batch)
    
    # evaluate on test_set
    total_batch = int(len(test_images)/batch_size)
    acc=0
    for i in range(total_batch):
        batch_xs, batch_ys = test_images[i].reshape([1,32,32,3]), test_labels[i].reshape([1,10])
        acc+=accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
    print("test_accuracy:",acc/total_batch)

