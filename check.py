import numpy as np
import time
import cv2
from particle_filter_using_lstm_regression import *

cap = cv2.VideoCapture('C:\\Users\\karth\\Pictures\\Camera Roll\\hand.mp4')
#cap = cv2.VideoCapture('../ps6/input/pedestrians.avi')
#cap = cv2.VideoCapture('./hand.mp4')
ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

w,h = frame.shape[:2]

#Tweak params:
N = 100
sig_d = 20
sig_mse = 10
#karthi hand
u = 520
v = 375
m = 200
n = 220
#romney hand
#u = 535
#v = 375
#m = 70
#n = 120
#white dressed lady
#u = 211
#v = 36
#m = 100
#n = 293
#face
#u = 320
#v = 175
#m = 103
#n = 129
start = time.clock()
prev_t = start
init_center = [u+m/2, v+n/2]

beta = 0.1
alpha = 0.2
temp = frame[v:v+n,u:u+m]
cv2.imshow('temp',temp)
cv2.waitKey(0)

count = 0

#S = part_filt_variable_window(N, temp, w, h, sig_d, sig_mse, beta, alpha= alpha)
S = part_filt(N, temp, w, h, sig_d, sig_mse, init_center)

while True:
	print ('frame no.: ', count)
	count = count + 1
	ret, frame = cap.read()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	if ret:
		prev_t = start
		start = time.clock()
		S.sample(frame)
		#S.update_temp(frame)

		#for i in xrange(S.num):
		#	cv2.circle(frame,(int(S.xt_1[i].u),int(S.xt_1[i].v)), 2, (0,0,255), -1)
		
		#u = np.sum([S.xt_1[i].u*S.xt_1[i].wt for i in xrange(S.num)])
		#v = np.sum([S.xt_1[i].v*S.xt_1[i].wt for i in xrange(S.num)])
		u = S.xt_1[0].u
		v = S.xt_1[0].v		
		#u = np.average([S.xt_1[i].u for i in xrange(1)])
		#v = np.average([S.xt_1[i].v for i in xrange(1)])
		n = S.n
		m = S.m
		cv2.rectangle(frame, (int(u-m/2),int(v-n/2)), (int(u+m/2), int(v+n/2)), (200,200,0), 2)
		
		cv2.imshow('frame',frame)
		print ('time for frame', start - prev_t)


		#cv2.imshow('temp', S.temp)
		cv2.waitKey(0)

	else:
		break
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
