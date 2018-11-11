
import tensorflow as tf
import numpy as np

# nombre d images
nbdata = 1000
trainDataFile = 'D:/bdr/Documents/TP 3 tensor flow/Deep_Learning_Cours/Deep_Learning_Cours/DataBases/data_1k.bin'
LabelFile = 'D:/bdr/Documents/TP 3 tensor flow/Deep_Learning_Cours/Deep_Learning_Cours/DataBases/gender_1k.bin'

# taille des images 48*48 pixels en niveau de gris
dim = 2304
#we open the wanted data in two color "red" "blue"
f = open(trainDataFile, 'rb')
data = np.empty([nbdata, dim], dtype=np.float32)
for i in range(nbdata):
	data[i,:] = np.fromfile(f, dtype=np.uint8, count=dim).astype(np.float32)
f.close()
#we construct our data matrix "data"
#we open our LabelFile 
f = open(LabelFile, 'rb')
label = np.empty([nbdata, 2], dtype=np.float32)
for i in range(nbdata):
	label[i,:] = np.fromfile(f, dtype=np.float32, count=2)
f.close()


def fc_layer(tensor, input_dim, output_dim): 
	print ('tensor  ',tensor.get_shape()   )
	Winit = tf.truncated_normal([input_dim, output_dim], stddev=0.1)
	W = tf.Variable(Winit)
	print ('Winit  ',Winit.get_shape())
	print ('W      ',W.get_shape())
	Binit = tf.constant(0.0, shape=[output_dim])
	B = tf.Variable(Binit)
	print ('Binit  ',Binit.get_shape())
	print ('B      ',B.get_shape())
	tensor = tf.matmul(tensor, W) + B
	print ('tensor  ',tensor.get_shape())
	print ('--------')
	return tensor
	

x = tf.placeholder(tf.float32, [None, dim])
y_desired = tf.placeholder(tf.float32, [None, 2])

layer1 = fc_layer(x,dim,50)
sigmo = tf.nn.sigmoid(layer1)
y = fc_layer(sigmo,50,2)

loss = tf.reduce_sum(tf.square(y - y_desired))
train_step = tf.train.GradientDescentOptimizer(1e-5).minimize(loss)

sess = tf.Session()	
sess.run(tf.global_variables_initializer())
curPos = 0
batchSize = 256
nbIt = 1000000
for it in range(nbIt):
	if curPos + batchSize > nbdata:
		curPos = 0
	trainDict = {x:data[curPos:curPos+batchSize,:],y_desired:label[curPos:curPos+batchSize,:]}
		
	curPos += batchSize

	sess.run(train_step, feed_dict=trainDict)
	if it%1000 == 0:
		print ("it= %6d - loss= %f" % (it, sess.run(loss, feed_dict=trainDict)))

sess.close()
