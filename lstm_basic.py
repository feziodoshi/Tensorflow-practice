import tensorflow as tf
import numpy as np
from random import shuffle
import random
import pickle

file=open('my_data.pickle','rb')
NUM_EX=1000000
num_hidden=24
output_dim=21
alpha=0.05
nb_epoch=10000
batch_size=50
def get_data():
	train_input = ['{0:020b}'.format(i) for i in range(2**20)]
	shuffle(train_input)

	# train_input = [map(int,i) for i in train_input]
	ti=[]

	for i in train_input:
		temp=[]
		for j in i:
			temp.append([j])
		ti.append(np.array(temp))

	train_input=ti

	train_output=[]

	for i in train_input:
		temp=[0]*21
		count=0
		for j in i:
			if j[0]==1:
				count+=1
		temp[count]=1
		train_output.append(temp)

	# print(len(train_input))
	# print(len(train_output))
	test_input=train_input[NUM_EX:]
	train_input=train_input[:NUM_EX]
	test_output=train_output[NUM_EX:]
	train_output=train_output[:NUM_EX]

	return train_input,train_output,test_input,test_output

# train_input,train_output,test_input,test_output=get_data()
# type(train_input)
# pickle.dump([train_input,train_output,test_input,test_output],file)
train_input,train_output,test_input,test_output=pickle.load(file)
print("DATA Ready")

#############################################################
print("MAKING COMPUTATION GRAPH")

###################making ops

##placeholder
with tf.name_scope("Inputs"):
	x=tf.placeholder(tf.float32,[None,20,1],name='input')
	y=tf.placeholder(tf.float32,[None,output_dim],name='targets')

##cell state
with tf.name_scope("RNNS"):
	cell=tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
	val,state=tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
	##val will be of the order: batch,sequence,output@sequence

	##preprocessing the val
	val=tf.transpose(val,[1,0,2])
	print(val.get_shape())
	##removing the sequence wala part
	last=tf.gather(val,int(val.get_shape()[0]-1),name='hiddenstatesforbatches')
	print(last.get_shape())

with tf.name_scope("feed_forward"):
	##creating weights and biases for the feedforward net
	weight=tf.Variable(tf.random_normal([num_hidden,output_dim]),name='W')
	bias=tf.Variable(tf.random_normal([output_dim]),name='bias')
	tf.summary.histogram('weights',weight)
	tf.summary.histogram('bias',bias)

	##final prediction
	y_=tf.nn.softmax(tf.add(tf.matmul(last,weight),bias),name='prediction')

##loss
with tf.name_scope('errors'):
	cross_entropy=-tf.reduce_sum(y*tf.log(y_))
with tf.name_scope('TRAINING'):
	train_step=tf.train.AdamOptimizer(learning_rate=alpha).minimize(cross_entropy)

##calculating accuracy
with tf.name_scope('accuracy'):
	correct=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
	accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

tf.summary.scalar('cost',cross_entropy)
tf.summary.scalar('accuracy',accuracy)
with tf.variable_scope("RNN/LSTMCell") as vs:
	my_var=[v for v in tf.trainable_variables()
			if v.name.startswith(vs.name)]
for i in my_var:
	tf.summary.histogram(str(i.name),i)

merged_summary=tf.summary.merge_all()

train_writer=tf.summary.FileWriter('/tmp/rnn_paper/count/train')
test_writer=tf.summary.FileWriter('/tmp/rnn_paper/count/test')
init_op=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init_op)
train_writer.add_graph(sess.graph)
test_writer.add_graph(sess.graph)

# x=tf.get_collection(tf.	GraphKeys.VARIABLES,scope='RNNS')
# my_var=tf.trainable_variables()

ptr=0
for i in range(nb_epoch):
	if(ptr+batch_size<=len(train_input)):
		batch_x=train_input[ptr:ptr+batch_size]
		batch_y=train_output[ptr:ptr+batch_size]
		ptr+=batch_size
		m,t=sess.run([merged_summary,train_step],feed_dict={x:batch_x,y:batch_y})
		train_writer.add_summary(m,i)
	if(i%10==0):
		##calculating the accuracy over test set
		m,acc=sess.run([merged_summary,accuracy],feed_dict={x:test_input[:10000],y:test_output[:10000]})
		test_writer.add_summary(m,i)
		print("Accuracy= ",acc,"epoch",i)
		# print(i)

# print(var.get_name for var in my_var)	
