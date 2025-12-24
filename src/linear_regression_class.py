from imports import *

class LinearRegression:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = np.asarray(X).reshape(len(X), len(X.axes[1]))
        self.y = np.asarray(y).reshape(len(y), 1)
    def optimal_weights(self) -> pd.Series:
        pass

    def mean_square_error(self, y_pred: np.array) -> float:
        square_error_mean_short = np.mean((self.y - y_pred) ** 2)
        return square_error_mean_short

    def predict(self, w:np.array, b:float) -> np.array:
        w = w.reshape((self.X.shape[1], 1))
        y_pred: np.array = self.X @ w + b
        return y_pred

        
