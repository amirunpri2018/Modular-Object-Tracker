import tensorflow as tf 
import numpy as np 
import cv2

batch_size = 1
max_time = 2000
num_size = 128

#test_data creation here (x and input_label)
test_file_path = './coords/10.txt'
frame_size = (640,480)
test_file = open(test_file_path, 'r')
stop_sequence = []
test_data_in = np.zeros((0, 2))
test_data_out = np.zeros((0, 2))
lines = test_file.read().splitlines()

u_f, v_f = (0,0)
u_f1, v_f1 = (0,0)
for j in range(len(lines) - 1):
	u, v, m, n = lines[j].split(',')
	new_u, new_v, new_m, new_n = lines[j + 1].split(',')
	u_f1 = float(new_u) - float(u);
	v_f1 = float(new_v) - float(v)
	test_data_in = np.append(test_data_in, np.array([[u_f, v_f]]), axis=0)
	test_data_out = np.append(test_data_out, np.array([[u_f1, v_f1]]), axis=0)
	u_f, v_f = (u_f1, v_f1)
stop_sequence.append(len(lines))

for j in range(max_time - len(lines) + 1):
	test_data_in = np.append(test_data_in, np.array([[-1,-1]]), axis = 0)
	test_data_out = np.append(test_data_out, np.array([[0,0]]), axis = 0)



x = tf.placeholder(tf.float32, [batch_size, max_time, 2], name='x')
input_labels = tf.placeholder(tf.float32, [batch_size, max_time, 2], name='input_labels')
seq_length = tf.placeholder(tf.int32, [batch_size], name='seq_length')

W1 = tf.get_variable('W1', [2, num_size], initializer = tf.initializers.random_normal())
b1 = tf.get_variable('b1', [1, num_size], initializer = tf.initializers.zeros(dtype=tf.float32))

x = tf.reshape(x, [max_time, 2])
rnn_input = tf.matmul(x, W1) + b1
rnn_input = tf.reshape(rnn_input, [1, max_time, num_size])

#rnn_input shape needs to be [batch_size, max_time, num_size]
cell = tf.nn.rnn_cell.LSTMCell(num_size, state_is_tuple=True)
cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
#lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_size), tf.nn.rnn_cell.LSTMCell(num_size)]
#lstm_cells = [tf.nn.rnn_cell.DropoutWrapper(lstm_cells, output_keep_prob=0.5)]
u_and_v_cell = tf.nn.rnn_cell.MultiRNNCell([cell]*2)
init_state = u_and_v_cell.zero_state(batch_size, tf.float32)

print('here')
print(u_and_v_cell.variables)
print('here2')

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
output_coords = tf.reshape(output_coords, [max_time,2])

saver = tf.train.Saver()



with tf.Session() as sess:

	saver.restore(sess, "/tmp/model.ckpt")
	print("Model restored.")
	feed_dict = {x: test_data_in, seq_length: stop_sequence}
	out_vec = sess.run([output_coords], feed_dict = feed_dict)
	test_accuracy = 0
	diff = ((out_vec - test_data_out)**2)
	err_dist = np.average(diff[0], axis = 0)
	#print diff
	for j in range(batch_size):
		truth_vec = diff[j]
		count = 0.
		for i in range(stop_sequence[j]):
			if truth_vec[i][0] + truth_vec[i][1] <= 25.:
				count += 1
		test_accuracy += (count*1./stop_sequence[j]) * 100 
	test_accuracy /= batch_size
	print('Test_accuracy: ', round(test_accuracy,2), 'err_dist: ', (err_dist[0] + err_dist[1])**0.5)