import tensorflow as tf
import numpy as np

trX = np.linspace(0, 1, 101)
trY = np.sin(trX)

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

c1 = tf.Variable(tf.random_normal([1]), name="c1")
c2 = tf.Variable(tf.random_normal([1]), name="c2")
c3 = tf.Variable(tf.random_normal([1]), name="c3")

Y_pred = X*c1 + tf.pow(X, 2)*c2 + tf.pow(X, 3)*c3

loss = tf.reduce_sum(tf.square(Y - Y_pred), name='loss') / 100
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('graph/poly', sess.graph)
	epochs = 100
	for epoch in xrange(epochs):
		for (x, y) in zip(trX, trY):
			sess.run(optimizer, feed_dict={X: x, Y: y})
		print "Epoch {}, Loss: {}".format(epoch, sess.run(loss, feed_dict={X: trX, Y: trY}))
	writer.close()
