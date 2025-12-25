from imports import *
from typing import Tuple, Union

class LinearRegression:
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, w: np.ndarray, b:float, X_test: pd.DataFrame):
        self.X_train = np.asarray(X_train).reshape(len(X_train), len(X_train.axes[1]))
        self.y_train = np.asarray(y_train).reshape(len(y_train), 1)
        self.w = w.reshape(w.shape[0], 1)
        self.b = b
        self.X_test = np.asarray(X_test).reshape(len(X_test), len(X_test.axes[1]))

    def mean_square_error(self) -> float:
        square_error_mean_short = np.mean((self.y_train - self.predict()) ** 2)
        return square_error_mean_short
    
    def get_weights(self) -> Tuple[np.ndarray, float]:
        iter = 1
        step_size = 1
        while step_size > 0.0001 and iter <= 5:
            print(self.mean_square_error())
            step_size = self.step(lr=1e-4)
            iter += 1
        print(iter)

        return self.w, self.b

    def step(self, lr:float) -> float:
        y_diff: np.ndarray = self.predict() - self.y_train
        w_steps = -lr * 2 * (self.X_train.T @ y_diff) /self.X_train.shape[1]
        b_step = -lr * 2 * np.mean(y_diff)

        self.w = self.w + w_steps
        self.b = self.b + b_step

        step_size: float = (np.sum(abs(w_steps)) + abs(b_step)) / (w_steps.shape[0] + 1)

        return step_size

    def predict(self, X: Union[None, pd.DataFrame] = None) -> np.ndarray:
        if X is None:
            X = self.X_train

        y_pred: np.ndarray = X @ self.w + self.b
        return y_pred
    
    def submit(self) -> pd.Series:
        output = self.predict(self.X_test)
        return pd.Series(output.ravel())

    

    


        
