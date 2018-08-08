import tensorflow as tf 
import numpy as np 
import cv2
import pickle

def get_training_data(batch_size, groundtruth_files, max_time):

	training_set_in = np.zeros((0, max_time, 2))
	training_set_out = np.zeros((0, max_time, 2))
	stop_sequence = []
	frame_sizes = [(640, 480), (352, 288), (241,193)]

	for i in range(batch_size):

		#groundtruth_files is a list of addresses of the files
		groundtruth_file = open(groundtruth_files[i], 'r')

		training_data_in = np.zeros((0, 2))
		training_data_out = np.zeros((0, 2))

		lines = groundtruth_file.read().splitlines()
		u_f, v_f = (0,0)
		u_f1, v_f1 = (0,0)
		for j in range(len(lines) - 1):
			u,v,m,n = lines[j].split(',')
			new_u, new_v, new_m, new_n = lines[j+1].split(',')
			u_f1 = float(new_u) - float(u); v_f1 = float(new_v) - float(v)
			training_data_in = np.append(training_data_in, np.array([[u_f, v_f]]), axis = 0)
			training_data_out = np.append(training_data_out, np.array([[u_f1, v_f1]]), axis = 0)
			u_f, v_f = (u_f1, v_f1)

		stop_sequence.append(len(lines))

		for j in range(max_time - len(lines) + 1):
			training_data_in = np.append(training_data_in, np.array([[-1,-1]]), axis = 0)
			training_data_out = np.append(training_data_out, np.array([[0,0]]), axis = 0)

		training_set_in = np.append(training_set_in, training_data_in.reshape((1,max_time,2)), axis = 0)
		training_set_out = np.append(training_set_out, training_data_out.reshape((1,max_time,2)), axis = 0)
	return (training_set_in, training_set_out, stop_sequence, frame_sizes)

def get_groundtruth_files():
	import glob
	files = glob.glob('./coords/*.txt')
	files.sort()
	return files


max_time = 2000
num_size = 128
batch_size = 10
groundtruth_files = get_groundtruth_files()
x_in, y_out, stop_sequence, frame_sizes = get_training_data(batch_size, groundtruth_files, max_time) 

x = tf.placeholder(tf.float32, [batch_size, max_time, 2], name='x')
input_labels = tf.placeholder(tf.float32, [batch_size, max_time, 2], name='input_labels')
#seq_length = tf.constant(stop_sequence)
seq_length = tf.constant(stop_sequence)

list_x = tf.split(x, batch_size)

W1 = tf.get_variable('W1', [2, num_size], initializer = tf.initializers.random_normal())
b1 = tf.get_variable('b1', [1, num_size], initializer = tf.initializers.zeros(dtype=tf.float32))
rnn_input = []

for xi in list_x:
	xi = tf.reshape(xi, [max_time, 2])
	yi = tf.matmul(xi, W1) + b1
	rnn_input.append(yi)
rnn_input = tf.stack(rnn_input)


#rnn_input shape needs to be [batch_size, max_time, num_size]
cell = tf.nn.rnn_cell.LSTMCell(num_size, state_is_tuple=True)
cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
#lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_size), tf.nn.rnn_cell.LSTMCell(num_size)]
#lstm_cells = [tf.nn.rnn_cell.DropoutWrapper(lstm_cells, output_keep_prob=0.5)]
u_and_v_cell = tf.nn.rnn_cell.MultiRNNCell([cell]*2)
init_state = u_and_v_cell.zero_state(batch_size, tf.float32)

rnn_outputs, final_state = tf.nn.dynamic_rnn(u_and_v_cell, rnn_input, initial_state=init_state, sequence_length=seq_length)

W2 = tf.get_variable('W2', [num_size, 2], initializer = tf.initializers.random_normal())
b2 = tf.get_variable('b2', [1, 2], initializer = tf.initializers.zeros(dtype=tf.float32))

rnn_outputs_list = tf.split(rnn_outputs, batch_size)
output_coords = []
for i in range(len(rnn_outputs_list)):
	o = rnn_outputs_list[i]
	o = tf.reshape(o, [max_time, num_size])
	oi = tf.matmul(o, W2) + b2
	oix = tf.split(oi, max_time)
	oix = [oix[j] for j in range(stop_sequence[i])]
	for j in range(max_time - stop_sequence[i]):
		oix.append(tf.zeros(dtype = tf.float32, shape = [1,2]))
	oix = tf.reshape(tf.stack(oix), shape = [max_time, 2])	
	output_coords.append(oix) 
output_coords = tf.stack(output_coords)
print(output_coords.get_shape())

loss = 0
coord_loss=[0,0]
for i in range(batch_size):
	coord_loss[0] += tf.reduce_mean(tf.square(input_labels[i, 0:stop_sequence[i], 0] - output_coords[i , 0:stop_sequence[i], 0]))
	coord_loss[1] += tf.reduce_mean(tf.square(input_labels[i, 0:stop_sequence[i], 1] - output_coords[i , 0:stop_sequence[i], 1]))


coord_loss[0]/=batch_size
coord_loss[1]/=batch_size

loss = (coord_loss[0]+coord_loss[1])/2.

tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W1)
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W2)
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, b1)
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, b2)
for var in u_and_v_cell.variables:
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, var)
scl = tf.placeholder(tf.float32, shape = [])
regularizer = tf.contrib.layers.l2_regularizer(scale=scl)
reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
loss += reg_term

lr = tf.placeholder(tf.float32, shape = [])
optimizer = tf.train.RMSPropOptimizer(lr).minimize(loss)

saver = tf.train.Saver()

epochs = 1000
prev_loss_val = 0
loss_val = 0
learning_rate = 1e-4
scale = 1e-2
err = 1e-5
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver.restore(sess, "/tmp/model.ckpt")
	#learning_rate, scale, err = pickle.load(open('hyp.pickle', 'rb'))
	print("Model restored.")
	print(x_in.shape)
	feed_dict = {x: x_in, input_labels: y_out, seq_length: stop_sequence, lr: learning_rate, scl: scale}
	for epoch in range(epochs):
		prev_loss_val = loss_val
		err_dist, loss_val, _o, out_vec = sess.run([coord_loss,loss, optimizer, output_coords], feed_dict)
		'''
		if prev_loss_val - loss_val < err and prev_loss_val != 0 and learning_rate >= 1e-5:
			learning_rate /= 2.
			scale/=10.
			feed_dict = {x: x_in, input_labels: y_out, lr: learning_rate, scl: scale}
			err/=10.
		'''
		train_accuracy = 0
		diff = ((out_vec - y_out)**2)
		for j in range(batch_size):
			truth_vec = diff[j]
			count = 0.
			for i in range(stop_sequence[j]):
				#print(truth_vec)
				if truth_vec[i][0] + truth_vec[i][1] <= 25.:
					count += 1
			train_accuracy += (count*1./stop_sequence[j]) * 100 
		train_accuracy /= batch_size
		#print(out_vec[0])
		print('Epoch: ', epoch, '\tloss: ', round(loss_val, 2) ,'\tdist_err: ', round((err_dist[0] + err_dist[1])**0.5, 2) ,'\ttrain_accuracy: ', round(train_accuracy,2), '\t learning rate', learning_rate)
	#out_vec = sess.run([output_coords], feed_dict)
	
	save_path = saver.save(sess, "/tmp/model.ckpt")
	meta_graph_def = tf.train.export_meta_graph(filename='/tmp/graph_lstm.meta')
	print('Model saved in ', save_path)