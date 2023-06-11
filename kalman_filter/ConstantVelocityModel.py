"""
Constant Velocity Model
"""
import numpy as np
from kalman_filter.KalmanFilter import KalmanFilter



dt = 0.1  # second
transform_matrix = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]], dtype=np.float32)  # x, y , vx, vy
predict_error = np.diag([20., 20., 10., 10.])

observed_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)  # algorithm only observe x, y

observe_error = np.array([[1, 0], [0, 1]], dtype=np.float32)


sv = 10  # 0.5 m/s^2
acceleration_array = np.array([[0.5 * dt** 2],
                               [0.5 * dt** 2],
                               [dt],
                               [dt]], dtype=np.float32)  # acceleration matrix

# acceleration uncertainty
Q = acceleration_array * acceleration_array.T * sv** 2

# test data
data_num = 200
x = 20
y = 10
mx = np.array(x + np.random.randn(data_num))
my = np.array(y + np.random.randn(data_num))
measurements = np.vstack((mx ,my))


kf = KalmanFilter(np.array([[x], [y], [0], [0]], dtype=np.float32), predict_error)
for index in range(data_num):
    x += 10
    y -= 10
    predict_state_cur, predict_error_cur = kf.predict(transform_matrix)
    print(predict_error_cur)
    kf.update_predict_error(predict_error + Q)
    cur_state = kf.update_state(observe_error, observed_matrix, np.array([[x], [y]], dtype=np.float32))
    # print((cur_state * 100).astype(np.int32) / 100.)