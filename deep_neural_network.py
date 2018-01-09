#THIS IS A DEEP NEURAL NETWORK MODEL WITH 4 HIDDEN LAYERS
#TRAINING DATA CONSIST OF 55,000 SAMPLES (IMAGES) AND THE TEST DATA CONSIST OF 10,000 SAMPLES
#LABELS ARE FROM 0 TO 9

#WE USE 2 IN-BUILT TENSORFLOW FUNCTIONS FOR OPTIMIZATION AND PREDICTION, WHICH IS ADAM OPTIMIZER AND SOFTMAX FUNCTION RESPECTIVELY
#RELU FUNCTION IS USED OVER THE HIDDEN LAYERS

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#ONE HOT ENCODER IS USED TO CONVERT THE LABELS FROM NUMERICAL FORM TO BINARY
#FOR EXAMPLE - DIGIT 2 IS CONVERTED TO 0010000000 - WHERE 1 CORRESPONDS TO THE INDEX VALUE OF THE DIGIT
mnist = input_data.read_data_sets('/home/roshan/Desktop/Tensorflow-Bootcamp-master/03-Convolutional-Neural-Networks/MNIST_data/', one_hot=True)

#PLACEHOLDERS
#SHAPE = [None, 784] - 'None' SINCE WE DO NOT INPUT THE ENTIRE DATASET AT ONCE. THE MODEL TRAVERSES THROUGH A BATCHSIZE OF 100 OR 200 SAMPLES INSTEAD
#pkeep - USED FOR ASSIGNING DROPOUT
x = tf.placeholder(tf.float32, shape=[None, 784])
y_true = tf.placeholder(tf.float32, shape=[None, 10])
pkeep = tf.placeholder(tf.float32)

#VARIABLES
w1 = tf.Variable(tf.truncated_normal([784,200], stddev=0.1))
b1 = tf.Variable(tf.zeros([200]))

w2 = tf.Variable(tf.truncated_normal([200,100], stddev=0.1))
b2 = tf.Variable(tf.zeros([100]))

w3 = tf.Variable(tf.truncated_normal([100,60], stddev=0.1))
b3 = tf.Variable(tf.zeros([60]))

w4 = tf.Variable(tf.truncated_normal([60,30], stddev=0.1))
b4 = tf.Variable(tf.zeros([30]))

w5 = tf.Variable(tf.truncated_normal([30,10], stddev=0.1))
b5 = tf.Variable(tf.zeros([10]))

#FLATTENING THE IMAGE 
x = tf.reshape(x,[-1,784])

#GRAPH
#HIDDEN LAYERS WITH RELU FUNCTION AND DROP OUT ON THE 3RD AND 4TH LAYER
y1 = tf.nn.relu(tf.matmul(x,w1) + b1)
y2 = tf.nn.relu(tf.matmul(y1,w2) + b2)

y3 = tf.nn.relu(tf.matmul(y2,w3) + b3) 
yf1 = tf.nn.dropout(y3, pkeep) # applied dropout

y4 = tf.nn.relu(tf.matmul(yf1,w4) + b4)
yf = tf.nn.dropout(y4, pkeep) #applied dropout

#OUTPUT
y = tf.nn.softmax(tf.matmul(yf,w5) + b5)

#OPTIMIZATION - USING GRADIENT DESCENT
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
train = optimizer.minimize(cross_entropy)

#INITIALIZING ALL VARIABLES
init = tf.global_variables_initializer()

#RUNNING THE MODEL
steps = 10001

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(steps):
    
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict ={x:batch_x, y_true:batch_y, pkeep: 0.75})
    
        if i%1000 == 0:
            print ('ON STEP : {}'.format(i))
            print 'ACCURACY'
            matches = tf.equal(tf.argmax(y_true,1), tf.argmax(y,1))
            
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))
            
            print(sess.run(acc, feed_dict ={x:mnist.test.images, y_true:mnist.test.labels, pkeep: 1}))
            print ('\n')

#ACCURACY ON TEST DATA - 0.9724

#THEREFORE BY ADDING FEW LAYERS AND APPLYING DROPOUT, OUR ACCURACY SIGNIFICANTLY INCREASED FROM 92%(SIMPLE NEURAL NETWORK) TO 97%

#THE PROBLEM WITH THIS MODEL IS THAT WE HAVE CHANGED A 2-DIMENSIONAL IMAGE TO 1-DIMENSIONAL.
#BY DOING THIS WE HAVE DESTROYED SHAPE INFORMATION

#THIS PROBLEM COULD BE RECTIFIED BY USING CONVOLUTED NEURAL NETWORKS
#CHECK OUT 'conv_neural_network.py'
