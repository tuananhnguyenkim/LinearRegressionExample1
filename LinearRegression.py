import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#insert X and Y
X_train = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
Y_train = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

#initializer X sample number
n_sample = X_train.shape[0]

# Create placeholder X and Y
X = tf.placeholder(tf.float32, name = 'X')
Y = tf.placeholder(tf.float32, name = 'Y')

#Linear Model : y = w * x + b so create and define variable w and b and set value is 0.0
w = tf.get_variable('weight', initializer =  tf.constant(0.0))
b = tf.get_variable('bias', initializer =  tf.constant(0.0))

Y_predicted = w * X + b

#loss function 
loss = tf.square(Y - Y_predicted, name = 'loss')

#minimizer loss function
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

#Watch in tensorboard by this code :
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

writer.close()

with tf.Session() as sess:
	#initializer the necessary variable , in this case w and b
	sess.run(tf.global_variables_initializer())

	#Set epoch is 100
	for i in range(100):
		total_loss = 0
		for x, y in zip(X_train, Y_train):
			#Session execute opimizer and fetch values of loss
			_, _loss = sess.run([optimizer,loss], feed_dict = {X: x, Y: y})
			total_loss += _loss
			print('Epoch{0}: {1}'.format(i,total_loss/ n_sample))
			w_out, b_out = sess.run([w, b])
			Y_predicted = X_train * w_out + b_out
#compare raw value and predicted value
for i, j in zip(Y_predicted, Y_train):
	print(i, '|' ,j)
