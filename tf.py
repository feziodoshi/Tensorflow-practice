import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('data/',one_hot=True)

##making the placeholders
X=tf.placeholder(tf.float32,[None,784]) ## if you are giving an image like this you will have to flatten the image later
##if the image is being taken in the shape of 28*28 then use tf.reshape(X,[-1,784]) in order to flatten it

Y_=tf.placeholder(tf.float32,[None,10])

##making the variables and initializing as zeros
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10])) ## this will automatically get broadcasted to all the examples

##Hyperparameters
alpha=0.005
drop_train=0.1
drop_test=0.5
total_epochs=10
batch_size=100

#################################OPS##########################

##op to initialize
init=tf.initialize_all_variables()

##op to compute activation(since the image is given in a 28*28 format, we will have to flatten it)
Y=tf.nn.softmax(tf.add(tf.matmul(X,W),b))

##op to calculate cost
cross_entropy_cost=-tf.reduce_sum(Y_*tf.log(Y))


##the main crux of tensorflow is here
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.005)
training=optimizer.minimize(cross_entropy_cost)


##final statistics
##is_correct=tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))
##accuracy=tf.reduce_mean(tf.cast(is_correct,tf.float32))
corr=tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))
accuracy=tf.reduce_mean(tf.cast(corr,"float"))
##################################################################

print("Computational Graph is ready")

print("Starting the session")

sess=tf.Session()
sess.run(init)

print("Training started")

for epoch in range(total_epochs):
	avg_cost=0.0
	total_num_batches=int(mnist.train.num_examples/batch_size)
	for batch in range(total_num_batches):
		batch_x,batch_y=mnist.train.next_batch(batch_size)
		feed = {X: batch_x, Y_: batch_y}##, dropout_keep_prob: drop_train}
		sess.run(training,feed_dict=feed)
		##calculate cost after training on this feed on this feed itself with a higher dropout
		feed={X:batch_x,Y_:batch_y}##,dropout_keep_prob:drop_test}
		avg_cost=sess.run(cross_entropy_cost,feed_dict=feed)
		##acc=sess.run(accuracy,feed_dict=feed)
	avg_cost=avg_cost/total_num_batches
	feed={X:mnist.test.images ,Y_:mnist.test.labels}
	acc=sess.run(accuracy,feed_dict=feed)
	print("Epoch:%d  cost:%f  accuracy after this epoch:%f" %(epoch,avg_cost,acc))

print("Optimization is Done")





