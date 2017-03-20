import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('data/',one_hot=True)



##Hyperparameters
alpha=0.001
std_dev=0.5
input_dim=784
hidden1_dim=250
hidden2_dim=100
hidden3_dim=20	
output_dim=10
total_epochs=50
batch_size=100
num_batches=int(mnist.train.num_examples/batch_size)
drop_train_keep=0.6
drop_test_keep=1.0

##placeholders
X=tf.placeholder(tf.float32,[None,input_dim],name='x')
Y_=tf.placeholder(tf.float32,[None,output_dim],name='labels')
dropout_keep_prob=tf.placeholder(tf.float32,name='dropout')

##Variables
W={
	"input_hidden1":tf.Variable(tf.random_normal([input_dim,hidden1_dim],stddev=std_dev),name='W1'),
	"hidden1_hidden2":tf.Variable(tf.random_normal([hidden1_dim,hidden2_dim],stddev=std_dev),name='W2'),
	"hidden2_hidden3":tf.Variable(tf.random_normal([hidden2_dim,hidden3_dim],stddev=std_dev),name='W3'),
	"hidden3_output":tf.Variable(tf.random_normal([hidden3_dim,output_dim],stddev=std_dev),name='W4')
}

b={
	"hidden1":tf.Variable(tf.random_normal([hidden1_dim],stddev=std_dev),name='B1'),
	"hidden2":tf.Variable(tf.random_normal([hidden2_dim],stddev=std_dev),name='B2'),
	"hidden3":tf.Variable(tf.random_normal([hidden3_dim],stddev=std_dev),name='B3'),
	"output":tf.Variable(tf.random_normal([output_dim],stddev=std_dev),name='B4')
}

def multilayer(x,w,b,keep_prob,name='fc'):
	if(name!="readout"):
		with(tf.name_scope(name)):
			output=tf.nn.sigmoid(tf.add(tf.matmul(x,w),b))
			output=tf.nn.dropout(output,keep_prob)
			return output
	elif(name=="readout"):
		with(tf.name_scope(name)):
			output=tf.nn.softmax(tf.add(tf.matmul(x,w),b))
			# output=tf.nn.dropout(output,keep_prob)
			return output


##Ops
init=tf.initialize_all_variables()
hidden1_output=multilayer(X,W["input_hidden1"],b["hidden1"],dropout_keep_prob,"fc1")
hidden2_output=multilayer(hidden1_output,W["hidden1_hidden2"],b["hidden2"],dropout_keep_prob,"fc2")
hidden3_output=multilayer(hidden2_output,W["hidden2_hidden3"],b["hidden3"],dropout_keep_prob,"fc3")
Y=multilayer(hidden3_output,W["hidden3_output"],b["output"],dropout_keep_prob,"readout")

with tf.name_scope("xent"):
	cross_entropy_cost=-tf.reduce_sum(Y_*tf.log(Y))

with tf.name_scope("TRAIN"):
	train_step=tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(cross_entropy_cost)   ###################          Beauty of Tensorflow                 ################

with tf.name_scope("Accuracy"):
	corr=tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))
	accuracy=tf.reduce_mean(tf.cast(corr,tf.float32))

tf.summary.scalar("cross entropy",cross_entropy_cost)
tf.summary.scalar("accuracy",accuracy)
tf.summary.histogram("w1",W["input_hidden1"])
tf.summary.histogram("w2",W["hidden1_hidden2"])
tf.summary.histogram("w3",W["hidden2_hidden3"])
tf.summary.histogram("w4",W["hidden3_output"])

merged_summary=tf.summary.merge_all()
print("Computational Graph is ready")

##Start Session and initialize
print("Staring Session")
sess=tf.Session()
sess.run(init)

writer=tf.summary.FileWriter("/tmp/mnist_demo/2")
print("done")
writer.add_graph(sess.graph)

#Training
print("Starting Training")
for epoch in range(10):
	avg_cost_epoch=0.0
	for batch in range(num_batches):
		batch_x,batch_y=mnist.train.next_batch(batch_size)
		feed={X:batch_x,Y_:batch_y,dropout_keep_prob:drop_train_keep}
		s,t,c=sess.run([merged_summary,train_step,cross_entropy_cost],feed_dict=feed)
		avg_cost_epoch+=c
		if(batch%100==0):
			writer.add_summary(s,batch)
	
	avg_cost_epoch=avg_cost_epoch/num_batches
	feed={X:mnist.test.images,Y_:mnist.test.labels,dropout_keep_prob:drop_test_keep}
	acc=sess.run(accuracy,feed_dict=feed)
	print("epoch=%d      Error:%f    Accuracy=%f" %(epoch,avg_cost_epoch,acc))



	##size of the gap between training loss and testing loss will tell us how much overfitting is done. Thus we can regularize it to prevent overfitting. T
	##Dropout=shootout, it keeps the ones you specify and replaces the other ones by zeros
	##Thus the weights and biases will not be updated for that iteration
	##if you want something to change based on training and testing, then use placeholders

	##sigmoid->relu->decay(noise reduces)->dropout(noise comes back)