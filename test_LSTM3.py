import tensorflow as tf
import numpy as np
import cv2
import glob
def predict(frame, frames_passed, u, v, lstm_state):
    batch_size = 1
    max_time = 2000
    num_size = 128

    #test_data creation here (x and input_label)
    #test_file_path = './coords/10.txt'

    frame_size = frame.shape[:2]

    #test_file = open(test_file_path, 'r')
    #stop_sequence = []


    #test_data_in = np.zeros((0, 2))
    test_data_in = np.array(np.array(u,v))
    center_coords = []
    #test_data_out = np.zeros((0, 2))


    #lines = test_file.read().splitlines()


    #test_data_in = np.append(test_data_in, np.array([[u,v]]))
    #test_data_out = np.append(test_data_out, np.array([[u_prev,v_prev]]))


    #u_f, v_f = (u,v)
    #u_f1, v_f1 = (u_prev, v_prev)
    #for j in range(len(lines) - 1):
    #	u, v, m, n = lines[j].split(',')
    #	new_u, new_v, new_m, new_n = lines[j + 1].split(',')
    #	u_f1 = float(new_u) - float(u);
    #	v_f1 = float(new_v) - float(v)
    #	test_data_in = np.append(test_data_in, np.array([[u_f, v_f]]), axis=0)
    #	center_coords.append( (int(float(u) + float(m)/2), int(float(v) + float(n)/2)) )
    #	test_data_out = np.append(test_data_out, np.array([[u_f1, v_f1]]), axis=0)
    #	u_f, v_f = (u_f1, v_f1)
    #stop_sequence.append(len(lines))

    #for j in range(max_time - len(lines) + 1):
    #	test_data_in = np.append(test_data_in, np.array([[-1,-1]]), axis = 0)
    #	test_data_out = np.append(test_data_out, np.array([[0,0]]), axis = 0)

    stop_sequence = [1]

    x = tf.placeholder(tf.float32, [batch_size, max_time, 2], name='x')
    #input_labels = tf.placeholder(tf.float32, [batch_size, max_time, 2], name='input_labels')
    seq_length = tf.placeholder(tf.int32, [batch_size], name='seq_length')

    W1 = tf.get_variable('W1', [2, num_size], initializer = tf.initializers.random_normal())
    b1 = tf.get_variable('b1', [1, num_size], initializer = tf.initializers.zeros(dtype=tf.float32))

    x = tf.reshape(x, [max_time, 2])
    rnn_input = tf.matmul(x, W1) + b1
    rnn_input = tf.reshape(rnn_input, [1, max_time, num_size])

    lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_size), tf.nn.rnn_cell.LSTMCell(num_size)]
    print('\n')
    print(lstm_cells[0].variables)
    print('\n')
    u_and_v_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)


    #init_state = u_and_v_cell.zero_state(batch_size, tf.float32)
    if frames_passed == 0:
        init_state = u_and_v_cell.zero_state(batch_size, tf.float32)
    else:
        init_state = lstm_state

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
    print('graph ', output_coords)
    output_coords = tf.reshape(output_coords, [max_time,2])
    print('graph2 ', output_coords)


    saver = tf.train.Saver()


    out_vec = 0
    with tf.Session() as sess:
        W1.initializer.run()
        b1.initializer.run()
        for var in u_and_v_cell.variables:
            print(var, ' hi')
            var.initializer.run()
        W2.initializer.run()
        b2.initializer.run()
        saver.restore(sess, "/tmp/model.ckpt")
        print("Model restored.")
        print(output_coords)
        print('feede_dict')
        feed_dict = {x: test_data_in, seq_length: stop_sequence}
        #feed_dict = {x: test_data_in}

        print(feed_dict)

        out_vec = sess.run([output_coords], feed_dict = feed_dict)
        out_vec = out_vec[0]

        #list_of_imgs = glob.glob('./Tiger1/img/*.jpg')
        #list_of_imgs.sort()
        #k = 0
        #for img in list_of_imgs:
        #	frame =  cv2.imread(img)
        #	frame = cv2.circle(frame, center_coords[k], 2, (0, 0, 255), -1)
        #	frame = cv2.circle(frame, center_coords[k+1], 2, (0,255,0), -1)
        #	frame = cv2.circle(frame, (int(center_coords[k][0] + out_vec[k][0]),int(center_coords[k][1] + out_vec[k][1])), 2, (255, 0, 0), -1)
        #	frame = cv2.circle(frame, center_coords[k+1], 12, (0,0,0), 1)

        #	cv2.imshow('frame', frame)
        #	cv2.waitKey(0)
        #	k+=1
        #cv2.destroyAllWindows()


    return out_vec, final_state
