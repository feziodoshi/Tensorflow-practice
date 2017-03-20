import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('data/',one_hot=True)


##	Hyperparameters

input_dim=28
input_channels=1
output_dim=10
learning_rate=0.05
total_epoch=20
batch_size=100
num_batches=int(mnist.train.num_examples/batch_size)
dropout_keep_train=0.6
dropout_keep_test=0.1



trainX,tranY,testX,testY=mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels
trainX=trainX.reshape(-1,input_dim,input_dim,input_channels)
testX=testX.reshape(-1,input_dim,input_dim,input_channels)

	

##creating placeholders
X=tf.placeholder(tf.float32,[None,input_dim,input_dim,input_channels])
Y_=tf.placeholder(tf.float32,[None,output_dim])
dropout_keep=tf.placeholder(tf.float32)


##creating weights
W={
	"input_hidden1":tf.Variable(tf.zeros([5,5,1,32])),
	"hidden1_hidden2":tf.Variable(tf.zeros([5,5,32,64])),
	"hidden2_hidden3":tf.Variable(tf.zeros([5,5,64,128])),
	"hidden3_hidden4_fully":tf.Variable(tf.zeros([4*4*128,64])),
	"hidden4_output_fully":tf.Variable(tf.zeros([64,10]))
}

# ##creting ops
def conv_model(x,w,keep_prob=0.5):
	hidden1_output=tf.nn.relu(tf.nn.conv2d(x,w['input_hidden1'],strides=[1,1,1,1],padding='SAME'))	##input has shape:[-1,28,28,1]
	hidden1_output=tf.nn.max_pool(hidden1_output,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')  ##hidden1 has shape:[-1,14,14,32]
	hidden1_output=tf.nn.dropout_keep(hidden1_output,keep_prob)

	hidden2_output=tf.nn.relu(tf.nn.conv2d(hidden1_output,w['hidden1_hidden2'],strides=[1,1,1,1],padding='SAME'))
	hidden2_output=tf.nn.max_pool(hidden2_output,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')	##hidden2 has shape:[-1,7,7,64]
	hidden2_output=tf.nn.dropout_keep(hidden2_output,keep_prob)

	hidden3_output=tf.nn.relu(tf.nn.conv2d(hidden2_output,w['hidden2_hidden3'],strides=[1,1,1,1],padding='SAME'))
	hidden3_output=tf.nn.max_pool(hidden3_output,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') 	##hidden3 has shape:[-1,4,4,128]
	hidden3_output=tf.nn.dropout_keep(hidden3_output,keep_prob)
	##now flatten this
	hidden3_output=tf.reshape(hidden3_output,[-1,w['hidden3_hidden4_fully'].get_shape().as_list()[0]])	##now the flattened shape:[-1,4*4*128]

	hidden4_output=tf.nn.relu(tf.matmul(hidden3_output,w['hidden3_hidden4_fully']))

	output=tf.nn.softmax(tf.matmul(hidden4_output,w['hidden4_output_fully']))

	return output

init=tf.initialize_all_variables()
Y=conv_model(X,W)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y,labels=Y_))
optimizer=tf.train.GradientDescentOptimizer(learning_rate)
train_step=optimizer.minimize(cost)

corr=tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))
accuracy=tf.reduce_mean(tf.cast(corr,tf.float32))

print("Computational Graph is ready")

print("Starting session")
sess=tf.Session()
sess.run(init)
print("Session Started and initialized")
##training

for epoch in range(total_epoch):
	avg_cost=0.0
	for batch in range(num_batches):
		batch_x,batch_y=mnist.train.next_batch(batch_size)
		batch_x=batch_x.reshape(-1,input_dim,input_dim,input_channels)
		feed={X:batch_x,Y_:batch_y,dropout_keep:dropout_keep_train}
		sess.run(train_step,feed_dict=feed)
		cost_batch=sess.run(cost,feed_dict=feed)
		avg_cost+=cost_batch
	avg_cost=avg_cost/num_batches
	feed={X:testX,Y_:testY}
	test_cost,acc=sess.run([cost,accuracy],feed_dict=feed)
	print("Epoch:  %d 	 	TrainingCost:  %f 	 	TestingCost:  %f 	Accuracy:  %f",%(epoch,avg_cost,test_cost,acc))





##























## here understand the concept of paraeter sharing, convolution, depth slice, depth column
##ask raghav about slicing a numpy array

## when padding is there as 'same' output is same , then just check the stride