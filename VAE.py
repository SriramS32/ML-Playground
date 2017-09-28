# Coded in 599 class, Deep learning and applications

import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
import numpy as np
import matplotlib
matplotlib.user('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
num_sample = mnist.train.num_examples
input_dim = 784
w = h = 28

class VariationalAutoencoder(object):
    def __init__(self, learning_rate=1e-4, batch_size=100, n_z=5):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z
        
        # build the network
        self.build()
        
        # launch as session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
    # build the network
    def build(self):
        # input
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, input_dim])
        
        # encoder
        # slim.fc(input, outputdim, scope, act_fn)
        f1 = fc(self.x, 512, scope='enc_fc1', activation_fn=tf.nn.elu)
        f2 = fc(f1, 384, scope='enc_fc2', activation_fn=tf.nn.elu)
        f3 = fc(f2, 256, scope='enc_fc3', activation_fn=tf.nn.elu)
        
        self.z_mu = fc(f3, self.n_z, scope='enc_fc4_mu', activation_fn=None)
        # log (sigma^2)
        self.z_log_sigma_sq = fc(f3, self.n_z, scope='enc_fc4_sigma', activation_fn=None)
        # N(z_mu, z_sigma)
        eps = tf.random_normal(shape=tf.shape(self.z_log_sigma_sq), 
                               mean=0, stddev=1, dtype=tf.float32) # Unigaussian
        
        self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq))*eps # Reversing to get back sigma
        # decoder
        g1 = fc(self.z, 256, scope='dec_fc1', activation_fn=tf.nn.elu)
        g2 = fc(g1, 384, scope='dec_fc2', activation_fn=tf.nn.elu)
        g3 = fc(g2, 512, scope='dec_fc3', activation_fn=tf.nn.elu)
        self.x_hat = fc(g3, input_dim, scope='dec_fc4', activation_fn=tf.sigmoid) # sigmoid b/c onehot encoded
        
        # losses
        # reconstruction loss
        # x <-> x_hat
        # H(x, x_hat) = - \Sigma x * log(x_hat) + (1-x)*log(1-x_hat)
        epsilon = 1e-10 # to prevent log(0)
        recon_loss = -tf.reduce_sum(
            self.x * tf.log(self.x_hat + epsilon) + (1-self.x)*tf.log(1-self.x_hat + epsilon),
            axis=1
        )
        
        # latent loss
        # KL divergence: measure the difference between two distributions
        # the latent distribution and N(0, 1)
        latent_loss = -0.5*tf.reduce_sum(
            1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq),
            axis=1
        )
        
        # total loss
        self.total_loss = tf.reduce_mean(recon_loss + latent_loss)
        
        # optimizer
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)
        
    
    # execute a forward and a backward pass
    # report the loss
    def run_single_step(self, x):
        _, loss = self.sess.run([self.train_op, self.total_loss], feed_dict={self.x: x})
        return loss
        
    
    # reconstruction
    # x -> x_hat
    def reconstructor(self, x):
        self.sess.run(self.x_hat, feed_dict={self.x: x})
    
    # generation
    # z -> x_hat
    def generator(self, z):
        return self.sess.run(self.x_hat, feed_dict={self.z: z})
        
    # transformation
    # x -> z
    def transformer(self, x):
        return self.sess.run(self.z, feed_dict={self.x: x})
        

def trainer(learning_rate=1e-4, batch_size=100, num_epoch=100, n_z=10):
    
    # model
    model = VariationalAutoencoder(learning_rate=learning_rate, batch_size=batch_size, n_z=n_z)
    
    # training loop
    for epoch in range(num_epoch):
        for iter in range(num_sample // batch_size):
            # obtain a mini-batch
            # tuple: images, labels
            batch = mnist.train.next_batch(batch_size)
            
            # train: execute a forward and a backward pass
            loss = model.run_single_step(batch[0]) # only want images not labels
        print ('[Epoch {}] Loss: {}'.format(epoch, loss))
    print('Done!')
    return model

# Train the model
model = trainer(learning_rate=1e-4,  batch_size=100, num_epoch=100, n_z=5)

# Test the trained model: reconstruction
batch = mnist.test.next_batch(100)
x_reconstructed = model.reconstructor(batch[0])

n = np.sqrt(model.batch_size).astype(np.int32)
I_reconstructed = np.empty((h*n, 2*w*n))
for i in range(n):
    for j in range(n):
        x = np.concatenate(
            (x_reconstructed[i*n+j, :].reshape(h, w),
             batch[0][i*n+j, :].reshape(h, w)),
            axis=1
        )
        I_reconstructed[i*h:(i+1)*h, j*2*w:(j+1)*2*w] = x

fig = plt.figure()
plt.imshow(I_reconstructed, cmap='gray')
plt.savefig('I_reconstructed.png')
plt.close(fig)

# Test the trained model: generation
# Sample noise vectors from N(0, 1)
z = np.random.normal(size=[model.batch_size, model.n_z])
x_generated = model.generator(z)

n = np.sqrt(model.batch_size).astype(np.int32)
I_generated = np.empty((h*n, w*n))
for i in range(n):
    for j in range(n):
        I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = x_generated[i*n+j, :].reshape(28, 28)

fig = plt.figure()
plt.imshow(I_generated, cmap='gray')
plt.savefig('I_generated.png')
plt.close(fig)

tf.reset_default_graph()
# Train the model with 2d latent space
model_2d = trainer(learning_rate=1e-4,  batch_size=100, num_epoch=50, n_z=2)

# Test the trained model: transformation
batch = mnist.test.next_batch(3000)
z = model_2d.transformer(batch[0])
fig = plt.figure()
plt.scatter(z[:, 0], z[:, 1], c=np.argmax(batch[1], 1))
plt.colorbar()
plt.grid()
plt.savefig('I_transformed.png')
plt.close(fig)

# Test the trained model: transformation
n = 20
x = np.linspace(-2, 2, n)
y = np.linspace(-2, 2, n)

I_latent = np.empty((h*n, w*n))
for i, yi in enumerate(x):
    for j, xi in enumerate(y):
        z = np.array([[xi, yi]]*model_2d.batch_size)
        x_hat = model_2d.generator(z)
        I_latent[(n-i-1)*28:(n-i)*28, j*28:(j+1)*28] = x_hat[0].reshape(28, 28)

fig = plt.figure()
plt.imshow(I_latent, cmap="gray")
plt.savefig('I_latent.png')
plt.close(fig)

