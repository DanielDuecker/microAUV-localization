import numpy as np
from pyquaternion import Quaternion


class ExtendedKalmanFilter(object):
    def __init__(self, x0=[1.0, 1.0, 0.5, 0]):  # states: x y z v(body frame)
        """ initialize EKF """
        self.__x_est_0 = np.array([[x0[0]], [x0[1]], [x0[2]], [x0[3]]]).reshape((4, 1))
        self.__x_est = self.__x_est_0
        self.__x_est_last_step = self.__x_est
        # standard deviations
        self.__sig_x1 = 0.200
        self.__sig_x2 = 0.200
        self.__sig_x3 = 0.100
        self.__sig_x4 = 0.100
        self.__p_mat_0 = np.array(np.diag([self.__sig_x1 ** 2,
                                           self.__sig_x2 ** 2,
                                           self.__sig_x3 ** 2,
                                           self.__sig_x4 ** 2]))
        self.__p_mat = self.__p_mat_0
        # process noise
        self.__sig_w1 = 0.05
        self.__sig_w2 = 0.05
        self.__sig_w3 = 0.05
        self.__sig_w4 = 0.05
        self.__q_mat = np.array(np.diag([self.__sig_w1 ** 2,
                                         self.__sig_w2 ** 2,
                                         self.__sig_w3 ** 2,
                                         self.__sig_w4 ** 2]))

        # measurement noise
        # --> see measurement_covariance_model
        self.__sig_r = 1
        self.__r_mat = self.__sig_r ** 2
        # measurement noise velocity
        self.__sig_v = 0.5
        self.__v_mat = np.array(np.diag([self.__sig_v ** 2]))

        self.__max_dist_to_tag = 3
        self.yaw_current = 0
        self.pitch_current = 0
        self.roll_current = 0
        self.frequency_prediction = 50
        self.frequency_update = 10
        # initial values and system dynamic (=eye)
        self.__f_mat = np.asarray([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])

    def yaw_pitch_roll_to_quat(self, yaw, pitch, roll):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        return (Quaternion(x=cy * cp * sr - sy * sp * cr, y=sy * cp * sr + cy * sp * cr, z=sy * cp * cr - cy * sp * sr,
                           w=cy * cp * cr + sy * sp * sr))

    def set_x_0(self, x0):
        self.__x_est = x0
        return True

    def set_p_mat_0(self, p0):
        self.__p_mat = p0
        return True

    def reset_ekf(self):
        self.__x_est = self.__x_est_0
        self.__p_mat = self.__p_mat_0

    def get_x_est(self):
        return self.__x_est.round(decimals=3)

    def get_p_mat(self):
        return self.__p_mat

    def get_z_meas(self):
        return self.__z_meas

    def get_y_est(self):
        return self.__y_est

    def get_roll_current(self):
        return self.roll_current

    def get_pitch_current(self):
        return self.pitch_current

    def get_yaw_current(self):
        return self.yaw_current

    # measurement function
    def h(self, x, vis_tags):
        num_vis_tags = vis_tags.shape[0]
        z = np.zeros((num_vis_tags, 1))

        for i, tag in enumerate(vis_tags):
            tag_pos = tag[1:4]
            # print("tag pos + " + str(tag_pos))
            # print("x= " + str(x))
            # r = sqrt((x - x_tag) ^ 2 + (y - y_tag) ^ 2 + (z - z_tag) ^ 2)
            r_dist = np.sqrt((x[0] - tag_pos[0]) ** 2 +
                             (x[1] - tag_pos[1]) ** 2 +
                             (x[2] - tag_pos[2]) ** 2)
            # print ("r = " + str(r_dist))
            z[i, 0] = r_dist

        return z

    # Jacobian of the measurement function
    def h_jacobian(self, x, vis_tags):

        num_vis_tags = vis_tags.shape[0]
        h_jac = np.zeros((num_vis_tags, 3))

        for i, tag in enumerate(vis_tags):
            tag_pos = tag[1:4]
            # r = sqrt((x - x_tag) ^ 2 + (y - y_tag) ^ 2 + (z - z_tag) ^ 2)
            r_dist = np.sqrt((x[0] - tag_pos[0]) ** 2 +
                             (x[1] - tag_pos[1]) ** 2 +
                             (x[2] - tag_pos[2]) ** 2)

            # dh /dx1
            h_jac_x1 = 0.5 * 2.0 * (x[0] - tag_pos[0]) / r_dist
            # dh /dx2
            h_jac_x2 = 0.5 * 2.0 * (x[1] - tag_pos[1]) / r_dist
            # dh /dx3
            h_jac_x3 = 0.5 * 2.0 * (x[2] - tag_pos[2]) / r_dist

            h_jac[i, 0:3] = [h_jac_x1, h_jac_x2, h_jac_x3]

        return h_jac  # dim [num_tag X 3]

    def current_rotation(self, yaw_current, pitch_current, roll_current):
        self.yaw_current = yaw_current
        self.pitch_current = pitch_current
        self.roll_current = roll_current

    def prediction(self, x_rot_vel, y_rot_vel, z_rot_vel):
        """ prediction """
        # accel_x=0

        self.yaw_current = self.yaw_current-z_rot_vel / self.frequency_prediction
        rotation = self.yaw_pitch_roll_to_quat(self.yaw_current, self.pitch_current, self.roll_current)
        update_x_y_z = rotation.rotate(
            np.asarray([[self.__x_est[3] / self.frequency_prediction], [0], [0]]))  # yaw_pitch_roll_to_quat
        # update_x_y_z = rotation.rotate(np.asarray([[0], [0], [0]]))
        update_x_y_z_v = np.asarray([[update_x_y_z[0]], [update_x_y_z[1]], [update_x_y_z[2]], [0]])
        # print(update_x_y_z_v)
        self.__x_est = np.matmul(self.__f_mat, self.__x_est) + update_x_y_z_v  # prediction = f * x_est + u
        self.__p_mat = np.matmul(self.__f_mat, np.matmul(self.__p_mat, np.transpose(self.__f_mat))) + self.__q_mat
        # print(self.__p_mat)
        return True

    def update(self, z_meas_tags):
        """ innovation """
        num_meas = z_meas_tags.shape[0]
        # get new measurement
        z_meas = z_meas_tags[:, 0].reshape(num_meas, 1)

        # estimate measurement from x_est
        z_est = self.h(self.__x_est[0:3], z_meas_tags)
        z_tild = z_meas - z_est

        # calc K-gain
        h_jac_mat = self.h_jacobian(self.__x_est[0:3], z_meas_tags)

        k_mat = np.zeros((3, num_meas))
        r_mat_temp = np.eye(num_meas) * self.__r_mat  # same measurement noise for all measurements, for the moment

        s_mat = np.dot(h_jac_mat, np.dot(self.__p_mat[0:3, 0:3], h_jac_mat.transpose())) + r_mat_temp
        s_diag = np.diag(s_mat)
        # compute k_mat in an interative way
        for i_tag in range(num_meas):
            k_mat[:, i_tag] = np.dot(self.__p_mat[0:3, 0:3], h_jac_mat[i_tag, :].transpose()) / s_diag[
                i_tag]  # 1/s scalar since s_mat is dim = 1x1
        # check distance to tag and reject far away tags
        b_tag_in_range = z_meas <= self.__max_dist_to_tag
        # print("bevore estimation")
        # print(self.__x_est)
        # print(z_meas)
        # print(z_est)
        # self.__x_est = self.__x_est + np.dot(k_mat[:, b_tag_in_range], z_tild[b_tag_in_range,0])  # = x_est + k * y_tild
        # print(k_mat[:, b_tag_in_range[:, 0]])

        self.__x_est[0:3] = self.__x_est[0:3] + np.matmul(k_mat[:, b_tag_in_range[:, 0]],
                                                          z_tild[b_tag_in_range]).reshape(
            (3, 1))  # = x_est + k * y_tild
        # self.__p_mat[0:3, 0:3] = np.matmul(
        #    (np.eye(3) - np.matmul(k_mat[:, b_tag_in_range[:, 0]], h_jac_mat[b_tag_in_range[:, 0], :])),
        #    self.__p_mat[0:3, 0:3])  # = (I-KH)*P
        # velocity calculation:
        # innovation y=z-h(x)
        y_vel = np.linalg.norm(self.__x_est[0:3] - self.__x_est_last_step[0:3]) / (
                1.0 / self.frequency_update) - self.__x_est[3]

        s_k_mat = self.__p_mat[3, 3] + self.__v_mat
        kalman_gain_v = self.__p_mat[3, 3] / s_k_mat
        p_update = np.zeros((4, 4))
        self.__x_est[3] = self.__x_est[3] + np.matmul(kalman_gain_v, y_vel)
        # self.__p_mat[3,3] = np.matmul((np.eye(3)-kalman_gain_v),self.__p_mat[3,3])
        if abs(self.__x_est[3]) > 1:
            self.__x_est[3] = self.__x_est[3] / abs(self.__x_est[3])
        p_update[0:3, 0:3] = (np.eye(3) - np.matmul(k_mat[:, b_tag_in_range[:, 0]], h_jac_mat[b_tag_in_range[:, 0], :]))
        p_update[3, 3] = (1 - kalman_gain_v)
        self.__p_mat = np.matmul(p_update, self.__p_mat)

        # print("after estimation")
        # print(self.__x_est)
        # save last state
        self.__x_est_last_step = np.copy(self.__x_est)

        if self.__x_est[0] > 5 or np.isnan(self.__x_est[0]) or self.__x_est[0] < -1:
            self.__x_est[0] = 1.5
            self.__x_est_last_step[0] = self.__x_est[0]
            self.__p_mat[0] = self.__p_mat_0[0]
        if self.__x_est[1] > 3 or np.isnan(self.__x_est[1]) or self.__x_est[1] < -1:
            self.__x_est[1] = 1
            self.__x_est_last_step[1] = self.__x_est[1]
            self.__p_mat[1] = self.__p_mat_0[1]

        if self.__x_est[2] > 2 or np.isnan(self.__x_est[2]) or self.__x_est[2] < -1:
            self.__x_est[2] = 0.5
            self.__x_est_last_step[2] = self.__x_est[2]
            self.__p_mat[2] = self.__p_mat_0[2]

        return True
