#THIS IS A CONVOLUTED NEURAL NETWORK MODEL WITH 4 HIDDEN LAYERS
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
w1 = tf.Variable(tf.truncated_normal([5,5,1,4], stddev=0.1))
b1 = tf.Variable(tf.ones([4]))

w2 = tf.Variable(tf.truncated_normal([4,4,4,8], stddev=0.1))
b2 = tf.Variable(tf.ones([8]))

w3 = tf.Variable(tf.truncated_normal([4,4,8,12], stddev=0.1))
b3 = tf.Variable(tf.ones([12]))

w4 = tf.Variable(tf.truncated_normal([7*7*12, 200], stddev=0.1))
b4 = tf.Variable(tf.ones([200]))

w5 = tf.Variable(tf.truncated_normal([200,10], stddev=0.1))
b5 = tf.Variable(tf.ones([10]))

#RESHAPING THE IMAGING INTO 28*28 PIXELS WITH 1 OUTPUT
x_image = tf.reshape(x, [-1,28,28,1])

#GRAPH
#HIDDEN LAYERS WITH RELU FUNCTION AND DROP OUT ON THE 3RD AND 4TH LAYER
y1 = tf.nn.relu(tf.nn.conv2d(x_image,w1, strides=[1,1,1,1], padding='SAME') + b1)
y2 = tf.nn.relu(tf.nn.conv2d(y1,w2, strides=[1,2,2,1], padding='SAME') + b2)

#Adding dropout
y3 = tf.nn.relu(tf.nn.conv2d(y2,w3, strides=[1,2,2,1], padding='SAME') + b3)
yf1 = tf.nn.dropout(y3, pkeep)

#Flattening the image
yy = tf.reshape(yf1, [-1,7*7*12])

#Adding dropout
y4 = tf.nn.relu(tf.matmul(yy, w4) + b4)
yf = tf.nn.dropout(y4, pkeep)

#OUTPUT
y = tf.nn.softmax(tf.matmul(yf, w5) + b5)

#OPTIMIZATION - USING GRADIENT DESCENT
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

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

#ACCURACY ON TEST DATA - 98.67%

#THEREFORE BY APPLYING CNN AND NOT FLATTENING THE IMAGE OUR ACCURACY INCREASED FROM 97.24% TO 98.76%

#THIS WAS MY FIRST NEURAL NETWORK PROJECT.
#THANK YOU FOR TAKING THE TIME. I HOPE THIS HAS BEEN OF SOME HELP

