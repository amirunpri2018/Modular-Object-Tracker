from test_LSTM3 import *
import numpy as np
import cv2
import time
from mergeSort import *


# Particle Filter class
class part_filt:

    # COnstructor
    def __init__(self, num, temp, w, h, sig_d, sig_mse, init_center, sigma_wm=1, ff=0.9, n_0=6, k=10, alpha=0.8):
        self.num = num
        self.n, self.m = temp.shape[:2]
        self.frames_passed = 0
        self.forget_factor = ff
        self.mu_data = temp.reshape((temp.size, 1))

        # number of eigenvectors
        self.k = k

        self.sub_s = np.zeros((self.n * self.m, 0))
        self.sigma_svd = np.zeros(0)

        self.xt = []
        self.xt_1 = []

        self.n_0 = n_0
        self.prev_us = np.zeros((0, 1))
        self.prev_vs = np.zeros((0, 1))
        self.t_poly_weights = np.zeros((4, 2))
        self.t_matrix = np.zeros((0, 4))

        self.sig_mse = sig_mse
        self.sig_d = sig_d
        self.alpha = alpha
        self.sigma_wm = sigma_wm

        self.lstm_state = 0

        self.weightmask = np.ones(self.mu_data.shape)
        for j in range(self.num):
            x = particle(init_center[0], init_center[1], 1.0 / self.num)
            self.xt_1.append(x)

    # Sample only first 'n' particles with highest weight
    # This solves the issue. LOL :P
    def sample(self, frame):

        self.num = len(self.xt_1)
        total_p = 0
        i = 0
        eta = 0.0

        # Store the mean of the ten best particles
        # nmlz = np.sum([self.xt_1[i].wt for i in range(self.num)])
        # self.mean_best_ten = (np.sum([self.xt_1[i].u*self.xt_1[i].wt for i in range(self.num)])/nmlz,
        #					  np.sum([self.xt_1[i].v*self.xt_1[i].wt for i in range(self.num)])/nmlz)
        nmlz = np.sum([self.xt_1[k].wt for k in range(10)])
        self.mean_best_ten = (np.sum([self.xt_1[k].u * self.xt_1[k].wt for k in range(10)]) / nmlz,
                              np.sum([self.xt_1[k].v * self.xt_1[k].wt for k in range(10)]) / nmlz)

        #self.regress()
        #u_t_plus_1 = self.get_new_u()
        #v_t_plus_1 = self.get_new_v()
        #self.print_vel()

        u_t_plus_1, v_t_plus_1 = self.pred_pos(frame)

        n_vel = int(self.alpha * self.num)  # n particles with the velocity

        cv2.imshow('mean best 10 ',
                   frame[int(self.mean_best_ten[1] - self.n / 2):int(self.mean_best_ten[1] + self.n / 2),
                   int(self.mean_best_ten[0] - self.m / 2): int(self.mean_best_ten[0] + self.m / 2)])
        #
        # print 'i < n_vel'
        while i < n_vel:
            p = int(round(self.xt_1[i].wt * n_vel, 0))
            total_p += p
            # print 'i ', i, ' p ', p
            if (total_p < n_vel):
                # Create Gaussian Noise
                # delt_u = np.random.normal(u_t_plus_1 - self.xt_1[0].u, self.sig_d, p)
                delt_u = np.random.normal(u_t_plus_1 - self.mean_best_ten[0], self.sig_d, p)
                # delt_v = np.random.normal(v_t_plus_1 - self.xt_1[0].v, self.sig_d, p)
                delt_v = np.random.normal(v_t_plus_1 - self.mean_best_ten[1], self.sig_d, p)
                j = 0
                while j < p:
                    new_u = self.xt_1[i].u + delt_u[j]
                    new_v = self.xt_1[i].v + delt_v[j]
                    new_wt = self.pzt(frame, new_u, new_v)
                    eta += new_wt
                    self.xt.append(particle(new_u, new_v, new_wt))
                    j += 1
                    # print '\tj ', j
            else:
                # Create Gaussian noise
                # delt_u = np.random.normal(u_t_plus_1 - self.xt_1[0].u, self.sig_d, n_vel - total_p + p)
                delt_u = np.random.normal(u_t_plus_1 - self.mean_best_ten[0], self.sig_d, n_vel - total_p + p)
                # delt_v = np.random.normal(v_t_plus_1 - self.xt_1[0].v, self.sig_d, n_vel - total_p + p)
                delt_v = np.random.normal(v_t_plus_1 - self.mean_best_ten[1], self.sig_d, n_vel - total_p + p)
                j = 0
                while j < n_vel - total_p + p:
                    new_u = self.xt_1[i].u + delt_u[j]
                    new_v = self.xt_1[i].v + delt_v[j]
                    new_wt = self.pzt(frame, new_u, new_v)
                    eta += new_wt
                    self.xt.append(particle(new_u, new_v, new_wt))
                    j += 1
                    # print '\tj ', j
                break
            i += 1
        # If target of 'n' particles has not been reached, add particles with higher weights
        # print 'total_p < n_vel'
        if (total_p < n_vel):
            delt_u = np.random.normal(u_t_plus_1 - self.mean_best_ten[0], self.sig_d, n_vel - total_p)
            # delt_u = np.random.normal(u_t_plus_1 - self.xt_1[0].u, self.sig_d, n_vel - total_p)
            delt_v = np.random.normal(v_t_plus_1 - self.mean_best_ten[1], self.sig_d, n_vel - total_p)
            # delt_v = np.random.normal(v_t_plus_1 - self.xt_1[0].v, self.sig_d, n_vel - total_p)
            j = 0
            while j < n_vel - total_p:
                # Create Gaussian noise
                new_u = self.xt_1[j].u + delt_u[j]
                new_v = self.xt_1[j].v + delt_v[j]
                new_wt = self.pzt(frame, new_u, new_v)
                eta += new_wt
                self.xt.append(particle(new_u, new_v, new_wt))
                j += 1
                # print '\tj ', j

        # Sample some amount of particles near the original place with noise too...
        # because vel can get haywire at times
        # CURRENT RATIO is 80:20
        n_wo_vel = self.num - n_vel
        delt_u = np.random.normal(0, self.sig_d, n_wo_vel)
        delt_v = np.random.normal(0, self.sig_d, n_wo_vel)
        i = 0
        # print 'i < n_wo_vel '
        while i < n_wo_vel:
            new_u = self.mean_best_ten[0] + delt_u[i]
            new_v = self.mean_best_ten[1] + delt_v[i]
            new_wt = self.pzt(frame, new_u, new_v)
            eta += new_wt
            self.xt.append(particle(new_u, new_v, new_wt))
            i += 1
            # print 'i ', i

        i = 0
        while i < self.num:
            self.xt[i].wt /= eta
            i += 1
            ##print 'i ', i

        # Merge sort, to sort particles by weight
        self.sort_by_weight()
        self.xt_1 = self.xt
        self.xt = []

        # self.weight_mask(frame)
        start = time.clock()
        self.update_temp(frame)
        self.disp_eig()
        self.frames_passed = self.frames_passed + 1
        # print 'sample function time ',time.clock() - start

    # Calculate P(Zt|Xt)
    def pzt(self, frame, u, v):
        h, w = frame.shape[:2]
        # start = time.clock()
        # Boundary Condtitions... :P
        if (u <= w - self.m / 2 and u >= self.m / 2 and v >= self.n / 2 and v <= h - self.n / 2):
            # All these if conditions to make sure we have same sized images to subtract

            if (self.n % 2 == 0 and self.m % 2 == 0):
                img2 = frame[int(v - self.n / 2): int(v + self.n / 2), int(u - self.m / 2): int(u + self.m / 2)]
            elif (self.n % 2 == 0 and self.m % 2 != 0):
                img2 = frame[int(v - self.n / 2): int(v + self.n / 2), int(u - self.m / 2): int(u + self.m / 2) + 1]
            elif (self.n % 2 != 0 and self.m % 2 == 0):
                img2 = frame[int(v - self.n / 2): int(v + self.n / 2) + 1, int(u - self.m / 2): int(u + self.m / 2)]
            else:
                img2 = frame[int(v - self.n / 2): int(v + self.n / 2) + 1, int(u - self.m / 2): int(u + self.m / 2) + 1]

                ##
                # img2 = frame[int(v):int(v+self.n), int(u):int(u+self.m)]
                ##
            img2 = img2.flatten()
            img2 = img2.reshape((img2.size, 1))

            # Real stuff happens here
            err = self.MSE(img2)
            weight = np.exp(-err / (2 * self.sig_mse ** 2))
            # weight = err
            ##print 'err wt', err,' ',weight
            ##print 'pzt time ', time.clock() - start

            return weight
        else:
            return 0

        # Mean Squared Error
        '''
    def MSE(self,img2):
        z = img2 - self.mu_data
        p = np.dot(self.sub_s, np.dot(self.sub_s.T,z))
        return np.sum((z-p)**2)/z.size
        '''
        # MSE with robust error norm

    def MSE(self, img2):
        z = img2 - self.mu_data
        p = np.dot(self.sub_s, np.dot(self.sub_s.T, z))
        l = (z - p) ** 2
        # m = (l > ((10**2)*np.ones(l.shape))).astype(int)
        err = np.sum((l.astype(float) / (l + (38 ** 2) * 3)))
        return err

    def sort_by_weight(self):
        mergeSort(self.xt, 0, int(self.num) - 1)

    def update_temp(self, frame):
        # u = self.xt_1[0].u
        # v = self.xt_1[0].v
        #
        nmlz = np.sum([self.xt_1[i].wt for i in range(10)])
        u = np.sum([self.xt_1[i].u * self.xt_1[i].wt for i in range(10)]) / nmlz
        v = np.sum([self.xt_1[i].v * self.xt_1[i].wt for i in range(10)]) / nmlz
        #

        if (self.n % 2 == 0 and self.m % 2 == 0):
            img2 = frame[int(v - self.n / 2): int(v + self.n / 2), int(u - self.m / 2): int(u + self.m / 2)]
        elif (self.n % 2 == 0 and self.m % 2 != 0):
            img2 = frame[int(v - self.n / 2): int(v + self.n / 2), int(u - self.m / 2): int(u + self.m / 2) + 1]
        elif (self.n % 2 != 0 and self.m % 2 == 0):
            img2 = frame[int(v - self.n / 2): int(v + self.n / 2) + 1, int(u - self.m / 2): int(u + self.m / 2)]
        else:
            img2 = frame[int(v - self.n / 2): int(v + self.n / 2) + 1, int(u - self.m / 2): int(u + self.m / 2) + 1]

            ##
            # img2 = frame[int(v):int(v+self.n), int(u):int(u+self.m)]
            #
            # cv2.imshow('img2', img2)
            ##
        img2 = img2.reshape((img2.size, 1))

        B = img2
        factor = (self.frames_passed * 1.0 / (self.frames_passed + 1)) ** 0.5

        B_hat = np.append(np.zeros((img2.size, 1)), (img2 - self.mu_data) * factor, axis=1)

        self.mu_data = (self.mu_data * (self.frames_passed) * self.forget_factor + img2) * 1. / (
                    (self.frames_passed) * self.forget_factor + 1)

        U_sigma = self.forget_factor * np.dot(self.sub_s,
                                              np.diag(self.sigma_svd))  # Matrix multiplication of U and Sigma
        QR_mat = np.append(U_sigma, B_hat, axis=1)  # This is the matrix whose QR factors we want

        U_B_tild, R = np.linalg.qr(QR_mat)

        U_tild, sig_tild, vh_tild = np.linalg.svd(R)

        U_new = np.dot(U_B_tild, U_tild)

        if (sig_tild.size > self.k):
            self.sigma_svd = sig_tild[0:self.k]
            self.sub_s = U_new[:, 0:self.k]
        else:
            j = 0  # iterator
            while j < self.sub_s.shape[1]:
                self.sub_s[:, j] = U_new[:, j]
                self.sigma_svd[j] = sig_tild[j]
                j = j + 1
            self.sub_s = np.append(self.sub_s, U_new[:, j].reshape((self.sub_s.shape[0], 1)), axis=1)
            self.sigma_svd = np.append(self.sigma_svd, sig_tild[j])

    def disp_eig(self):
        for i in range(self.sub_s.shape[1]):
            sub_s = self.sub_s[:, i].reshape(self.mu_data.shape)
            temp = sub_s  # + self.mu_data)/255.0
            # temp = (self.sub_s[:,i])
            disp = temp.reshape(self.n, self.m)
            # cv2.imshow('disp2', disp)
            disp = cv2.normalize(disp, 0, 255, cv2.NORM_MINMAX)
            # stack = np.dstack((stack,disp))
            # cv2.imshow('disp', disp)
            # cv2.imshow('mean', self.mu_data.reshape((self.n, self.m))/255.0)
            # cv2.waitKey(0)

    # Occlusion handling
    def weight_mask(self, frame):
        u = np.sum([self.xt_1[i].u * self.xt_1[i].wt for i in range(self.num)])
        v = np.sum([self.xt_1[i].v * self.xt_1[i].wt for i in range(self.num)])
        It = 0
        D = np.zeros(self.weightmask.shape)
        # need to make It as a mn cross k matrix
        if (self.n % 2 == 0 and self.m % 2 == 0):
            It = frame[int(v - self.n / 2): int(v + self.n / 2), int(u - self.m / 2): int(u + self.m / 2)]
        elif (self.n % 2 == 0 and self.m % 2 != 0):
            It = frame[int(v - self.n / 2): int(v + self.n / 2), int(u - self.m / 2): int(u + self.m / 2) + 1]
        elif (self.n % 2 != 0 and self.m % 2 == 0):
            It = frame[int(v - self.n / 2): int(v + self.n / 2) + 1, int(u - self.m / 2): int(u + self.m / 2)]
        else:
            It = frame[int(v - self.n / 2): int(v + self.n / 2) + 1, int(u - self.m / 2): int(u + self.m / 2) + 1]
            It = It.flatten()
            prod = It - np.matmul(np.matmul(self.sub_s, self.sub_s.T), It)
            # prod = prod.flatten()
            for i in range(prod.size):
                D[i] = prod[i] * self.weightmask[i]
                self.weightmask[i] = np.exp(-1 * D[i] ** 2 / self.sigma_wm ** 2)

    # some kind of cubic regression in temporal domain,
    # predicts next point given the motion history
    # needs slight tweaks, slightly unstable model

    # OPEN TO SUGGESTIONS!!!! :P
    # Run and see
    '''
    def regress(self):
        print('u(t) = ', self.mean_best_ten[0])
        print('v(t) = ', self.mean_best_ten[1])

        self.prev_us = np.append(self.prev_us, np.ones((1, 1)) * self.mean_best_ten[0], axis=0)
        self.prev_vs = np.append(self.prev_vs, np.ones((1, 1)) * self.mean_best_ten[1], axis=0)

        if (self.frames_passed >= self.n_0):
            self.prev_us = np.delete(self.prev_us, 0, axis=0)
            self.prev_vs = np.delete(self.prev_vs, 0, axis=0)

        if self.frames_passed == 0:
            t = np.zeros((1, 4))
            t[0, 0] = 1.
            self.t_matrix = np.append(self.t_matrix, t, axis=0)
            self.t_poly_weights[:, 0] = np.array([self.prev_us[0], 0, 0, 0])
            self.t_poly_weights[:, 1] = np.array([self.prev_vs[0], 0, 0, 0])

        elif self.frames_passed == 1:
            t = np.array([(self.frames_passed ** i) for i in range(4)]).reshape((1, 4))
            self.t_matrix = np.append(self.t_matrix, t, axis=0)
            self.t_poly_weights[:, 0] = np.array([self.prev_us[0], self.prev_us[1] - self.prev_us[0], 0, 0])
            self.t_poly_weights[:, 1] = np.array([self.prev_vs[0], self.prev_vs[1] - self.prev_vs[0], 0, 0])

        elif self.frames_passed == 2:
            t = np.array([(self.frames_passed ** i) for i in range(4)]).reshape((1, 4))
            self.t_matrix = np.append(self.t_matrix, t, axis=0)

            self.t_poly_weights[:, 0] = np.array([self.prev_us[0],
                                                  (-self.prev_us[2] + 4 * self.prev_us[1] - 3 * self.prev_us[0]) / 2.0,
                                                  (self.prev_us[2] - 2 * self.prev_us[1] + self.prev_us[0]) / 2.0, 0])

            self.t_poly_weights[:, 1] = np.array([self.prev_vs[0],
                                                  (-self.prev_vs[2] + 4 * self.prev_vs[1] - 3 * self.prev_vs[0]) / 2.0,
                                                  (self.prev_vs[2] - 2 * self.prev_vs[1] + self.prev_vs[0]) / 2.0, 0])

        else:
            if self.frames_passed < self.n_0:
                t = np.array([(self.frames_passed ** i) for i in range(4)]).reshape((1, 4))
                self.t_matrix = np.append(self.t_matrix, t, axis=0)
            ata = np.dot(self.t_matrix.T, self.t_matrix)
            self.t_poly_weights[:, 0] = np.dot(np.linalg.inv(ata), np.dot(self.t_matrix.T, self.prev_us)).reshape((4,))
            self.t_poly_weights[:, 1] = np.dot(np.linalg.inv(ata), np.dot(self.t_matrix.T, self.prev_vs)).reshape((4,))

    '''
    def pred_pos(self, frame):
        print ('pred_pos_called')
        pred_u, pred_v, self.lstm_state = predict(frame, self.frames_passed, self.mean_best_ten[0], self.mean_best_ten[1], self.lstm_state)
        return pred_u, pred_v

    def get_new_u(self):
        if (self.frames_passed >= self.n_0):
            tplusone = np.array([((self.n_0 + 0.5) ** i) for i in range(4)])
        else:
            tplusone = np.array([((self.frames_passed - 0.5) ** i) for i in range(4)])
        new_u = np.dot(tplusone, self.t_poly_weights[:, 0].reshape((4, 1)))[0]
        print('u(t+1) = ', new_u)
        return new_u

    def get_new_v(self):
        if (self.frames_passed >= self.n_0):
            tplusone = np.array([((self.n_0) ** i) for i in range(4)])
        else:
            tplusone = np.array([((self.frames_passed - 0.5) ** i) for i in range(4)])

        new_v = np.dot(tplusone, self.t_poly_weights[:, 1].reshape((4, 1)))[0]
        print('v(t+1) = ', new_v)
        return new_v

    def print_vel(self):
        t = (self.n_0 + 0.5)
        vel_u = self.t_poly_weights[1, 0] + 2 * self.t_poly_weights[2, 0] * t + 3 * self.t_poly_weights[3, 0] * (t ** 2)
        vel_v = self.t_poly_weights[1, 1] + 2 * self.t_poly_weights[2, 1] * t + 3 * self.t_poly_weights[3, 1] * (t ** 2)

        # print 'velocity: (', vel_u, ', ', vel_v, ')'


