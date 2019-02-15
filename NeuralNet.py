import ImageHeader

import tensorflow as tf
import numpy as np
import re

class NeuralNet(object):
    imgHdr = ImageHeader.ImageHeader()
    x_train = np.empty(1)

    y_train = np.empty(1)


    x_test = np.empty(1)
    y_test = np.empty(1)

    def __init__(self, imageHdr):
        self.imgHdr = imageHdr

        self.x_train = np.empty((self.imgHdr.maxImages, self.imgHdr.imgHeight * self.imgHdr.imgWidth))
        self.y_train = np.empty(shape=(self.imgHdr.maxImages, 15))

        self.x_test = np.empty((self.imgHdr.maxImages, self.imgHdr.imgHeight * self.imgHdr.imgWidth))
        self.y_test = np.empty(shape=(self.imgHdr.maxImages, 15))

    def getTrainImg(self, index, filelabel, fileimg):

        # have encode the labels into one-hot arrays
        regex = re.compile(r'\d+')
        targetNum = regex.search(filelabel)
        num = int(targetNum.group())
        temp_encoder = np.zeros((1,15))
        temp_encoder[0, num-1] = 1
        print("train encoder: ", temp_encoder)
        self.y_train[index-1] = temp_encoder


        #temp_train = np.empty(self.imgHdr.imgWidth *self.imgHdr.imgHeight)
        temp_train = fileimg
        self.x_train[index-1] = temp_train.flatten()
        print("train shape: ", self.x_train.shape)

        return self.x_train, self.y_train




    def getTestImg(self, index, filelabel, fileimg):

        regex = re.compile(r'\d+')
        targetNum = regex.search(filelabel)
        num = int(targetNum.group())
        #have encode the labels into one-hot arrays
        temp_encoder = np.zeros((1,15))
        temp_encoder[0, num-1] = 1
        print("test encoder: ", temp_encoder)
        self.y_test[index-1] = temp_encoder


        #temp_test  = np.empty(self.imgHdr.imgHeight*self.imgHdr.imgWidth)
        temp_test = fileimg
        self.x_test[index-1] = temp_test.flatten()
        print("test shape: ", self.x_test.shape)

        return self.x_test, self.y_test

    def trainMNIST(self):
        sess = tf.Session()
        x_inputs = tf.placeholder(tf.float32, shape = [None, self.imgHdr.imgWidth * self.imgHdr.imgHeight])
        y_outputs = tf.placeholder(tf.float32, shape = [None, 15])
        inputPop = self.imgHdr.imgWidth * self.imgHdr.imgHeight
        Weights = tf.Variable(tf.zeros([inputPop, 15]))
        bias  = tf.Variable(tf.zeros([15]))
        y_prediction = tf.nn.softmax(tf.matmul(x_inputs, Weights) + bias)


        #settings
        x_train = self.x_train
        y_train = self.y_train
        x_test = self.x_test
        y_test = self.y_test



        lr = 0.0001
        train_steps = 2500

        init = tf.global_variables_initializer()
        sess.run(init)

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_outputs * tf.log(y_prediction), reduction_indices=[1]))
        training = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(y_outputs, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        for i in range(train_steps + 1):
            thePrediction = sess.run(y_prediction, feed_dict={x_inputs: x_train})
            print("y_prediction: ", str(thePrediction))
            sess.run(training, feed_dict={x_inputs: x_train, y_outputs: y_train})
            if i % 100 == 0:
                print('Training Step:' + str(i) + '  Accuracy =  ' +
                      str(sess.run(accuracy, feed_dict={x_inputs: x_test, y_outputs: y_test})) + '  Loss = ' +
                      str(sess.run(cross_entropy, {x_inputs: x_train, y_outputs: y_train})))