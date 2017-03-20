import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('data/',one_hot=True)



##Hyperparameters
alpha=0.001
std_dev=0.5
input_dim=784
hidden1_dim=512
hidden2_dim=256
hidden3_dim=128	
output_dim=10
total_epochs=50
batch_size=100
num_batches=int(mnist.train.num_examples/batch_size)
drop_train_keep=0.6
drop_test_keep=1.0

##placeholders
X=tf.placeholder(tf.float32,[None,input_dim])
Y_=tf.placeholder(tf.float32,[None,output_dim])
dropout_keep_prob=tf.placeholder(tf.float32)

##Variables
W={
	"input_hidden1":tf.Variable(tf.random_normal([input_dim,hidden1_dim],stddev=std_dev)),
	"hidden1_hidden2":tf.Variable(tf.random_normal([hidden1_dim,hidden2_dim],stddev=std_dev)),
	"hidden2_hidden3":tf.Variable(tf.random_normal([hidden2_dim,hidden3_dim],stddev=std_dev)),
	"hidden3_output":tf.Variable(tf.random_normal([hidden3_dim,output_dim],stddev=std_dev))
}

b={
	"hidden1":tf.Variable(tf.random_normal([hidden1_dim],stddev=std_dev)),
	"hidden2":tf.Variable(tf.random_normal([hidden2_dim],stddev=std_dev)),
	"hidden3":tf.Variable(tf.random_normal([hidden3_dim],stddev=std_dev)),
	"output":tf.Variable(tf.random_normal([output_dim],stddev=std_dev))
}

def multilayer(x,w,b,keep_prob):
	hidden1_output=tf.nn.sigmoid(tf.add(tf.matmul(x,w["input_hidden1"]),b["hidden1"]))
	hidden1_output=tf.nn.dropout(hidden1_output,keep_prob)
	hidden2_output=tf.nn.sigmoid(tf.add(tf.matmul(hidden1_output,w["hidden1_hidden2"]),b["hidden2"]))
	hidden2_output=tf.nn.dropout(hidden2_output,keep_prob)
	hidden3_output=tf.nn.sigmoid(tf.add(tf.matmul(hidden2_output,w["hidden2_hidden3"]),b["hidden3"]))
	output=tf.nn.softmax(tf.add(tf.matmul(hidden3_output,w["hidden3_output"]),b["output"]))
	return output


##Ops
init=tf.initialize_all_variables()
Y=multilayer(X,W,b,dropout_keep_prob)
cross_entropy_cost=-tf.reduce_sum(Y_*tf.log(Y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=alpha)

train_step=optimizer.minimize(cross_entropy_cost)   ###################          Beauty of Tensorflow                 ################

corr=tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))
accuracy=tf.reduce_mean(tf.cast(corr,tf.float32))

print("Computational Graph is ready")

##Start Session and initialize
print("Staring Session")
sess=tf.Session()
sess.run(init)

writer=tf.summary.FileWriter("/tmp/mnist_demo/1")
print("done")
writer.add_graph(sess.graph)
#Training
print("Starting Training")
for epoch in range(10):
	avg_cost_epoch=0.0
	for batch in range(num_batches):
		batch_x,batch_y=mnist.train.next_batch(batch_size)
		feed={X:batch_x,Y_:batch_y,dropout_keep_prob:drop_train_keep}
		sess.run(train_step,feed_dict=feed)
		avg_cost_epoch+=sess.run(cross_entropy_cost,feed_dict=feed)
	
	avg_cost_epoch=avg_cost_epoch/num_batches
	feed={X:mnist.test.images,Y_:mnist.test.labels,dropout_keep_prob:drop_test_keep}
	acc=sess.run(accuracy,feed_dict=feed)
	print("epoch=%d      Error:%f    Accuracy=%f" %(epoch,avg_cost_epoch,acc))



	##size of the gap between training loss and testing loss will tell us how much overfitting is done. Thus we can regularize it to prevent overfitting. T
	##Dropout=shootout, it keeps the ones you specify and replaces the other ones by zeros
	##Thus the weights and biases will not be updated for that iteration
	##if you want something to change based on training and testing, then use placeholders

	##sigmoid->relu->decay(noise reduces)->dropout(noise comes back)