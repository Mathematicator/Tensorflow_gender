import Layers
import DataSets as ds
import numpy as np
import tensorflow as tf

#############################################################################
# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
################################################################################
################################################################################
################################################################################

#define a get_dict function to extract next training batch in training mode
def get_dict(database,IsTrainingMode):
	xs,ys = database.NextTrainingBatch()
	return {x:xs,y_desired:ys,ITM:IsTrainingMode}

#Loading model is false 
LoadModel = False
#??
KeepProb_Dropout = 0.9
#we give a nem to the expirement KeepProb_Dropout
experiment_name = '10k_Dr%.3f'%KeepProb_Dropout
#train = ds.DataSet('../DataBases/data_1k.bin','../DataBases/gender_1k.bin',1000)
train = ds.DataSet('D:/bdr/Documents/TP 3 tensor flow/Deep_Learning_Cours/Deep_Learning_Cours/DataBases/data_10k.bin','D:/bdr/Documents/TP 3 tensor flow/Deep_Learning_Cours/Deep_Learning_Cours/DataBases/gender_10k.bin',10000)
#train = ds.DataSet('../DataBases/data_100k.bin','../DataBases/gender_100k.bin',100000)
test = ds.DataSet('D:/bdr/Documents/TP 3 tensor flow/Deep_Learning_Cours/Deep_Learning_Cours/DataBases/data_test10k.bin','D:/bdr/Documents/TP 3 tensor flow/Deep_Learning_Cours/Deep_Learning_Cours/DataBases/gender_test10k.bin',10000)

#we give to tf our x as input and y as output 
with tf.name_scope('input'):
	x = tf.placeholder(tf.float32, [None, train.dim],name='x')
	y_desired = tf.placeholder(tf.float32, [None, 2],name='y_desired')
	ITM = tf.placeholder("bool", name='Is_Training_Mode')

#we unflat our images to apply the filters "nbfilters=3" in traing mode 
with tf.name_scope('CNN'):
	t = Layers.unflat(x,48,48,1)
	nbfilter = 3
	for k in range(4):
		for i in range(2):
			t = Layers.conv(t,nbfilter,3,1,ITM,'conv_%d_%d'%(nbfilter,i),KeepProb_Dropout)
		t = Layers.maxpool(t,2,'pool')
		nbfilter *= 2
	#after we flat our image 
	t = Layers.flat(t)
	#t = Layers.fc(t,50,ITM,'fc_1',KeepProb_Dropout)
	y = Layers.fc(t,2,ITM,'fc_2',KP_dropout=1.0,act=tf.nn.log_softmax)

with tf.name_scope('cross_entropy'):
	diff = y_desired * y 
	with tf.name_scope('total'):
		cross_entropy = -tf.reduce_mean(diff)
	tf.summary.scalar('cross entropy', cross_entropy)	
	
with tf.name_scope('accuracy'):
	with tf.name_scope('correct_prediction'):
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_desired, 1))
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)	

with tf.name_scope('learning_rate'):
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(1e-3,global_step,1000, 0.75, staircase=True)


with tf.name_scope('learning_rate'):
    tf.summary.scalar('learning_rate', learning_rate)

#train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)
merged = tf.summary.merge_all()

Acc_Train = tf.placeholder("float", name='Acc_Train');
Acc_Test = tf.placeholder("float", name='Acc_Test');
MeanAcc_summary = tf.summary.merge([tf.summary.scalar('Acc_Train', Acc_Train),tf.summary.scalar('Acc_Test', Acc_Test)])


print ("-----------",experiment_name)
print ("-----------------------------------------------------")
print ("-----------------------------------------------------")


sess = tf.Session()	
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter(experiment_name, sess.graph)
saver = tf.train.Saver()
if LoadModel:
	saver.restore(sess, "./model.ckpt")

nbIt = 5000
for it in range(nbIt):
	trainDict = get_dict(train,IsTrainingMode=True)					
	sess.run(train_step, feed_dict=trainDict)
	
	if it%10 == 0:
		acc,ce,lr = sess.run([accuracy,cross_entropy,learning_rate], feed_dict=trainDict)
		print ("it= %6d - rate= %f - cross_entropy= %f - acc= %f" % (it,lr,ce,acc ))
		summary_merged = sess.run(merged, feed_dict=trainDict)
		writer.add_summary(summary_merged, it)	
				
	if it%100 == 50:
		Acc_Train_value = train.mean_accuracy(sess,accuracy,x,y_desired,ITM)
		Acc_Test_value = test.mean_accuracy(sess,accuracy,x,y_desired,ITM)
		print ("mean accuracy train = %f  test = %f" % (Acc_Train_value,Acc_Test_value ))
		summary_acc = sess.run(MeanAcc_summary, feed_dict={Acc_Train:Acc_Train_value,Acc_Test:Acc_Test_value})
		writer.add_summary(summary_acc, it)
		
writer.close()
if not LoadModel:
	saver.save(sess, "./model.ckpt")
sess.close()
