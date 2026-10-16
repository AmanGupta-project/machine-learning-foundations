import numpy as np

class LinearRegressionScratch:
    def __init__(self, alpha=0.01, n_iters = 1000):
        self.alpha = alpha
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.cost_history = []

    def compute_cost(self, X, y):
        m = X.shape[0]
        y_hat = np.dot(X,self.w) + self.b
        cost = (1 / (2*m)) * np.sum((y_hat - y) ** 2)
        return cost

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        self.cost_history = []

        for i in range(self.n_iters):
            y_hat = np.dot(X,self.w) + self.b

            #vectorized gradients
            error = y_hat - y
            dj_dw = (1 / m) * np.dot(X.T, error)
            dj_db = (1 / m) * np.sum(error)

            self.w -= self.alpha * dj_dw
            self.b -= self.alpha * dj_db

            if i % 10 == 0 or i == self.n_iters-1:
                cost = self.compute_cost(X, y)
                self.cost_history.append((i, cost))

    def predict(self, X):
        return np.dot(X, self.w) + self.b
                   