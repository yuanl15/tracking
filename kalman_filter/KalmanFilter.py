import numpy as np


class KalmanFilter():
    def __init__(self, init_state, predict_error):
        self.state = init_state
        self.predict_error = predict_error

    def predict(self, transform_matrix):
        self.predict_state = np.matmul(transform_matrix, self.state)
        self.predict_error = np.matmul(np.matmul(transform_matrix, self.predict_error), transform_matrix.T)
        return self.predict_state, self.predict_error

    def update_predict_error(self, predict_error):
        self.predict_error = predict_error

    def update_state(self, observed_error, observe_matrix, observed_state):
        S = np.matmul(np.matmul(observe_matrix, self.predict_error), observe_matrix.T) + observed_error
        kalman_gain = np.matmul(np.matmul(self.predict_error, observe_matrix.T), np.linalg.pinv(S))
        error =  observed_state - np.matmul(observe_matrix, self.state)
        self.state = self.predict_state + np.matmul(kalman_gain,error)
        self.predict_error = np.matmul((np.identity(self.state.shape[0]) - np.matmul(kalman_gain, observe_matrix)), self.predict_error)
        return self.state