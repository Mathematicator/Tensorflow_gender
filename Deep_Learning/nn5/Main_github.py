import tensorflow as tf
import numpy as np
import DataSets as ds
import Layers

def get_dict(database,IsTrainingMode):
	xs,ys = database.next_training_batch()
	return {x:xs,y_desired:ys,ITM:IsTrainingMode}

LoadModel = False
KeepProb_Dropout = 0.95

experiment_name = '10k_Dr%.3f'%KeepProb_Dropout
#train = ds.DataSet('../DataBases/data_1k.bin','../DataBases/gender_1k.bin',1000)
train = ds.DataSet('../DataBases/data_10k.bin','../DataBases/gender_10k.bin',10000)
#train = ds.DataSet('../DataBases/data_100k.bin','../DataBases/gender_100k.bin',100000)
test = ds.DataSet('../DataBases/data_test10k.bin','../DataBases/gender_test10k.bin',10000)

with tf.name_scope('input'):
	x = tf.placeholder(tf.float32, [None, train.dim],name='x')
	y_desired = tf.placeholder(tf.float32, [None, 2],name='y_desired')
	ITM = tf.placeholder("bool", name='Is_Training_Mode')

with tf.name_scope('CNN'):
	t = Layers.unflat(x,48,48,1)
	nbfilter = 16
	for k in range(4):
		for i in range(2):
			t = Layers.conv(t,nbfilter,3,1,ITM,'conv_%d_%d'%(nbfilter,i),KeepProb_Dropout)
		t = Layers.maxpool(t,2,'pool')
		nbfilter *= 2
	
	t = Layers.flat(t)
	t = Layers.fc(t,50,ITM,'fc_1',KeepProb_Dropout)
	y = Layers.fc(t,2,ITM,'fc_2',KP_dropout=1.0,act=tf.nn.log_softmax)

with tf.name_scope('cross_entropy'):
	diff = y_desired * y 
	with tf.name_scope('total'):
		cross_entropy = -tf.reduce_mean(diff)
	tf.scalar_summary('cross entropy', cross_entropy)	
	
with tf.name_scope('accuracy'):
	with tf.name_scope('correct_prediction'):
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_desired, 1))
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.scalar_summary('accuracy', accuracy)	

with tf.name_scope('learning_rate'):
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(1e-3,global_step,1000, 0.75, staircase=True)


with tf.name_scope('learning_rate'):
    tf.scalar_summary('learning_rate', learning_rate)

#train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)
merged = tf.merge_all_summaries()

Acc_Train = tf.placeholder("float", name='Acc_Train');
Acc_Test = tf.placeholder("float", name='Acc_Test');
MeanAcc_summary = tf.merge_summary([tf.scalar_summary('Acc_Train', Acc_Train),tf.scalar_summary('Acc_Test', Acc_Test)])


print("-----------------------------------------------------"
print("-----------",experiment_name
print("-----------------------------------------------------"

sess = tf.Session()	
sess.run(tf.initialize_all_variables())
writer = tf.train.SummaryWriter(experiment_name, sess.graph)
saver = tf.train.Saver()
if LoadModel:
	saver.restore(sess, "./model.ckpt")

nbIt = 10000
for it in range(nbIt):
	trainDict = get_dict(train,IsTrainingMode=True)					
	sess.run(train_step, feed_dict=trainDict)
	
	if it%100 == 0:
		acc,ce = sess.run([accuracy,cross_entropy], feed_dict=trainDict)
		print("it= %6d - cross_entropy= %f - acc= %f" % (it,ce,acc )
		summary_merged = sess.run(merged, feed_dict=trainDict)
		writer.add_summary(summary_merged, it)	
				
	if it%1000 == 0:
		Acc_Train_value = train.mean_accuracy(sess,accuracy,x,y_desired,ITM)
		Acc_Test_value = test.mean_accuracy(sess,accuracy,x,y_desired,ITM)
		print("mean accuracy train = %f  test = %f" % (Acc_Train_value,Acc_Test_value )
		summary_acc = sess.run(MeanAcc_summary, feed_dict={Acc_Train:Acc_Train_value,Acc_Test:Acc_Test_value})
		writer.add_summary(summary_acc, it)
		
writer.close()
if not LoadModel:
	saver.save(sess, "./model.ckpt")
sess.close()