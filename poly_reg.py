import tensorflow as tf
import numpy as np

trX = np.linspace(0, 1, 101)
trY = np.sin(trX)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

c1 = tf.Variable(tf.random_normal([1]))
c2 = tf.Variable(tf.random_normal([1]))
c3 = tf.Variable(tf.random_normal([1]))

Y_pred = X*c1 + tf.pow(X, 2)*c2 + tf.pow(X, 3)*c3

loss = tf.reduce_sum(tf.square(Y - Y_pred)) / 100
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	epochs = 100
	for epoch in xrange(epochs):
		for (x, y) in zip(trX, trY):
			sess.run(optimizer, feed_dict={X: x, Y: y})
		print "Epoch {}, Loss: {}".format(epoch, sess.run(loss, feed_dict={X: trX, Y: trY}))
