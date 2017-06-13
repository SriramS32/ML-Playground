# w*X + b, (Y - Y_p)^2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

DATA_FILE = 'data/fire_theft.xls'

book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

w = tf.Variable(0.0, name="w")
b = tf.Variable(0.0, name="b")

Y_p = X*w + b

loss = tf.square(Y - Y_p, name='loss')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	writer = tf.summary.FileWriter('./graph/03/lin_reg', sess.graph)
	
	for i in range(100):
		total_loss = 0
		for x, y in data:
			_, loss_on_run = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
			total_loss += loss_on_run
		print "Epoch {}: {}".format(i, total_loss/n_samples)
	writer.close()

	w_v, b_v = sess.run([w, b])
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X*w_v + b_v, 'r', label='Predicted data')
plt.legend()
plt.show()