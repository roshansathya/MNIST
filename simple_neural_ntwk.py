#THIS IS A SIMPLE NEURAL NETWORK MODEL WITH NO HIDDEN LAYERS
#TRAINING DATA CONSIST OF 55,000 SAMPLES (IMAGES) AND THE TEST DATA CONSIST OF 10,000 SAMPLES
#LABELS ARE FROM 0 TO 9

#WE USE 2 IN-BUILT TENSORFLOW FUNCTIONS FOR OPTIMIZATION AND PREDICTION, WHICH IS GRADIENT DESCENT AND SOFTMAX FUNCTION RESPECTIVELY

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#ONE HOT ENCODER IS USED TO CONVERT THE LABELS FROM NUMERICAL FORM TO BINARY
#FOR EXAMPLE - DIGIT 2 IS CONVERTED TO 0010000000 - WHERE 1 CORRESPONDS TO THE INDEX VALUE OF THE DIGIT
mnist = input_data.read_data_sets('/home/roshan/Desktop/Tensorflow-Bootcamp-master/03-Convolutional-Neural-Networks/MNIST_data/', one_hot=True)

#PLACEHOLDERS
#SHAPE = [None, 784] - 'None' SINCE WE DO NOT INPUT THE ENTIRE DATASET AT ONCE. THE MODEL TRAVERSES THROUGH A BATCHSIZE OF 100 OR 200 SAMPLES INSTEAD
x = tf.placeholder(tf.float32, shape=[None, 784])
y_true = tf.placeholder(tf.float32, shape=[None, 10])

#VARIBLES
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#GRAPH - USING SOFTMAX FUNCTON
y_pred = tf.nn.softmax(tf.matmul(x,w) + b)

#OPTIMIZATION - USING GRADIENT DESCENT
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

#INITIALIZING ALL VARIABLES
init = tf.global_variables_initializer()

#RUNNING THE MODEL
with tf.Session() as sess:
    sess.run(init)
    
    for steps in range(5000):
        label_x, label_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict = {x:label_x, y_true:label_y})
        
    is_correct = tf.equal(tf.argmax(y_true,1), tf.argmax(y_pred,1))
    acc = tf.reduce_mean(tf.cast(is_correct,tf.float32))
    
    print sess.run(acc,feed_dict ={x:mnist.test.images, y_true:mnist.test.labels})

#ACCURACY ON TEST DATA - 0.9215

#WE CAN INCREASE THE ACCURACY BY ADDING HIDDEN LAYERS FOR ABSTRACTION
#CHECK OUT 'deep_neural_network.py'