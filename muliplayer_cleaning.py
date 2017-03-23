import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('data/',one_hot=True)


def train(alp=0.01,hidden1_dim_args=512,hidden2_dim_args=256,hidden3_dim_args=128,drop_args=0.6,dir='1'):
	##Hyperparameters
	alpha=alp
	std_dev=0.5
	input_dim=784
	hidden1_dim=hidden1_dim_args
	hidden2_dim=hidden2_dim_args
	hidden3_dim=hidden3_dim_args
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



	def multilayer(x,inp,outp,keep_prob,name='fc'):
		w=tf.Variable(tf.random_normal([inp,outp],stddev=std_dev),name='W')
		b=tf.Variable(tf.random_normal([outp],stddev=std_dev),name='B')
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
		tf.summary.histogram("w",w)
		tf.summary.histogram("b",b)



	##Ops
	init=tf.initialize_all_variables()
	hidden1_output=multilayer(X,input_dim,hidden1_dim,dropout_keep_prob,"fc1")
	hidden2_output=multilayer(hidden1_output,hidden1_dim,hidden2_dim,dropout_keep_prob,"fc2")
	hidden3_output=multilayer(hidden2_output,hidden2_dim,hidden3_dim,dropout_keep_prob,"fc3")
	Y=multilayer(hidden3_output,hidden3_dim,output_dim,dropout_keep_prob,"readout")

	with tf.name_scope("xent"):
		cross_entropy_cost=-tf.reduce_sum(Y_*tf.log(Y))

	with tf.name_scope("TRAIN"):
		train_step=tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(cross_entropy_cost)   ###################          Beauty of Tensorflow                 ################

	with tf.name_scope("Accuracy"):
		corr=tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))
		train_accuracy=tf.reduce_mean(tf.cast(corr,tf.float32))
		test_accuracy=tf.reduce_mean(tf.cast(corr,tf.float32))

	tf.summary.scalar("cross entropy",cross_entropy_cost)
	tf.summary.scalar("train accuracy",train_accuracy)
	# tf.summary.scalar("test accuracy",test_accuracy)
	
	merged_summary=tf.summary.merge_all()
	print("Computational Graph is ready")

	##Start Session and initialize
	print("Staring Session")
	sess=tf.Session()
	sess.run(init)

	writer=tf.summary.FileWriter("/tmp/mnist_demo/1")
	print("done")
	# writer.add_graph(sess.graph)

	#Training
	print("Starting Training")
	for epoch in range(5000):
		avg_cost_epoch=0.0
		batch_x,batch_y=mnist.train.next_batch(batch_size)
		feed={X:batch_x,Y_:batch_y,dropout_keep_prob:drop_train_keep}
		s,t,c=sess.run([merged_summary,train_step],feed_dict=feed)
		avg_cost_epoch+=c
		if(batch%10==0):
			writer.add_summary(s,epoch)
		feed={X:mnist.test.images,Y_:mnist.test.labels,dropout_keep_prob:drop_test_keep}
		# acc=sess.run(accuracy,feed_dict=feed)
		# print("epoch=%d      Error:%f    Accuracy=%f" %(epoch,avg_cost_epoch,acc))

train()


		##size of the gap between training loss and testing loss will tell us how much overfitting is done. Thus we can regularize it to prevent overfitting. T
		##Dropout=shootout, it keeps the ones you specify and replaces the other ones by zeros
		##Thus the weights and biases will not be updated for that iteration
		##if you want something to change based on training and testing, then use placeholders

		##sigmoid->relu->decay(noise reduces)->dropout(noise comes back)