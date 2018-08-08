import numpy as np
import time
import cv2
from particle_filter_using_lstm_regression import *

#cap = cv2.VideoCapture('../ps6/input/pres_debate.avi')
#cap = cv2.VideoCapture('../ps6/input/pedestrians.avi')
#cap = cv2.VideoCapture('./hand.mp4')
#cap = cv2.VideoCapture('../hand.mp4')
#ret, frame = cap.read()
count = 1
frame = cv2.imread('./BlurBody/BlurBody/img/000'+str(count)+'.jpg')
#frame = cv2.imread('../Suv/img/000'+str(count)+'.jpg')
#frame = cv2.imread('../Dancer2/img/000'+str(count)+'.jpg')
#frame = cv2.imread('../Bolt2/img/000'+str(count)+'.jpg')
#frame = cv2.imread('/Jumping/img/000'+str(count)+'.jpg')
#frame = cv2.imread('./Man/img/000'+str(count)+'.jpg')

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#
#count = 100
#

w,h = frame.shape[:2]

#Tweak params:
N = 100
sig_d = 10
sig_mse = 10
#Jumping
#u = 145
#v = 95
#m = 36
#n = 50
#Man
#u = 69
#v = 48
#m = 26
#n = 39
#Bolt2
#u = 269
#v = 75
#m = 34
#n = 64
#Dancer2
#u = 150
#v = 53
#m = 40
#n = 148
#Suv
#u = 142
#v = 125
#m = 91
#n = 40
#BlurBody
u = 400
v = 48
m = 87
n = 319
#karthi hand
#u = 520
#v = 375
#m = 200
#n = 220
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

#count = 0
ret = True

#S = part_filt_variable_window(N, temp, w, h, sig_d, sig_mse, beta, alpha= alpha)
S = part_filt(N, temp, w, h, sig_d, sig_mse, init_center)

while True:
	print('frame no.: ', count)
	count = count + 1
	#ret, frame = cap.read()
	if count<10:
		frame = cv2.imread('./BlurBody/BlurBody/img/000'+str(count)+'.jpg')
		#frame = cv2.imread('../Suv/img/000'+str(count)+'.jpg')
		#frame = cv2.imread('../Dancer2/img/000'+str(count)+'.jpg')
		#frame = cv2.imread('../Bolt2/img/000'+str(count)+'.jpg')
		#frame = cv2.imread('/Jumping/img/000'+str(count)+'.jpg')
		#frame = cv2.imread('./Man/img/000'+str(count)+'.jpg')

	elif count>=10 and count < 100:
		frame = cv2.imread('./BlurBody/BlurBody/img/00'+str(count)+'.jpg')
		#frame = cv2.imread('../Suv/img/00'+str(count)+'.jpg')
		#frame = cv2.imread('../Dancer2/img/00'+str(count)+'.jpg')
		#frame = cv2.imread('../Bolt2/img/00'+str(count)+'.jpg')
		#frame = cv2.imread('/Jumping/img/00'+str(count)+'.jpg')
		#frame = cv2.imread('./Man/img/00'+str(count)+'.jpg')

	else:
		frame = cv2.imread('./BlurBody/BlurBody/img/0'+str(count)+'.jpg')
		#frame = cv2.imread('../Suv/img/0'+str(count)+'.jpg')
		#frame = cv2.imread('../Dancer2/img/0'+str(count)+'.jpg')
		#frame = cv2.imread('../Bolt2/img/0'+str(count)+'.jpg')
		#frame = cv2.imread('/Jumping/img/0'+str(count)+'.jpg')
		#frame = cv2.imread('./Man/img/0'+str(count)+'.jpg')

	frame1 = np.copy(frame)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	if count<=334:
		ret = True
	else:
		ret = False

	if ret:
		prev_t = start
		start = time.clock()

		S.sample(frame)
		#S.update_temp(frame)

		for i in range(S.num):
			cv2.circle(frame1,(int(S.xt_1[i].u),int(S.xt_1[i].v)), 2, (0,0,min(255, S.xt_1[i].wt*2550)), -1)

		#u = np.sum([S.xt_1[i].u*S.xt_1[i].wt for i in xrange(S.num)])
		#v = np.sum([S.xt_1[i].v*S.xt_1[i].wt for i in xrange(S.num)])
		u = S.xt_1[0].u
		v = S.xt_1[0].v
		#u = np.average([S.xt_1[i].u for i in xrange(1)])
		#v = np.average([S.xt_1[i].v for i in xrange(1)])
		n = S.n
		m = S.m
		cv2.rectangle(frame1, (int(u-m/2),int(v-n/2)), (int(u+m/2), int(v+n/2)), (200,200,0), 2)

		cv2.imshow('frame',frame1)
		print ('time for frame', start - prev_t)
		print()

		#cv2.imshow('temp', S.temp)
		cv2.waitKey(0)

	else:
		break
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()
