import argparse
import os
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def main(_):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    #   x will be used to funnel the mnist dataset into the neural network
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])

    #   W = weights, b = bias, y probability
    W = tf.Variable(tf.zeros(shape=[784, 10]))
    b = tf.Variable(tf.zeros(shape=[10]))
    y = tf.matmul(x, W) + b

    # y_ is actually y' and is the true distribution
    y_ = tf.placeholder(dtype=tf.float64, shape=[None, 10])

    #   cross entropy puts a number to how inaccurate our
    #   prediction is at describing the truth
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                logits=y)
    )

    train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    #   train it!
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    # compare a the actual output with the expected output
    correct_prediction = tf.equal(tf.argmax(y, 1),
                                  tf.argmax(y_, 1))

    #   how accurate was the comparison
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    run = sess.run(accuracy,feed_dict={x: mnist.test.images,y_: mnist.test.labels})
    print('This MNIST iteration was {}% accurate'.format(round(run * 100, 2)))


if __name__ == '__main__':
    #   gets rid of a pesky warning
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
