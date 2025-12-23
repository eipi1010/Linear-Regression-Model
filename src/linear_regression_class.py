from imports import *

class linear_regression:
    def __init__(self, X: pd.Series, y: pd.Series):
        self.X = X
        self.y = y
    def optimal_weights(self) -> pd.Series:
        pass

    def mean_square_error(self, y_pred: pd.Series) -> float:
        square_error_sum: float = 0
        for i in range(len(y_pred)):
            square_error_sum += (self.y.iloc[i] - y_pred.iloc[i]) ** 2
        square_error_mean: float = square_error_sum/len(y_pred)

        #square_error_mean = ((self.y - y_pred) ** 2).mean()

        return square_error_mean

    def predict(self) -> pd.Series:
        y_pred: list[int] = []
        for x in self.X:
            y_pred_i = 3*x + 5
            y_pred.append(y_pred_i)

        y_pred = pd.DataFrame(y_pred)
        y_pred.columns = ['diagnosed_diabetes']
        
        return y_pred

        
