"""Script for tensorboard visualization of a node that
provides no usefull results"""
import tensorflow as tf

a = 2
b = 3

x = tf.add(a, b)
y = tf.multiply(a, b)

useless = tf.multiply(a, x, name="useless")

z = tf.pow(y, x)

with tf.Session() as sess:
	writer = tf.summary.FileWriter('./graph/02/useless_nodes', sess.graph)
	z = sess.run(z)

writer.close()