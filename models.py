import numpy as np
import torch
from torch import nn


class Ridge_Regression:

    def __init__(self, lambd):
        self.lambd = lambd

    def fit(self, X, Y):
        """
        Fit the ridge regression model to the provided data.
        :param X: The training features.
        :param Y: The training labels.
        """
        Y = 2 * (Y - 0.5)  # transform the labels to -1 and 1, instead of 0 and 1.
        # compute the ridge regression weights using the formula from class / exercise.
        # you may not use np.linalg.solve, but you may use np.linalg.inv
        N_train = X.shape[0]
        identity_matrix = np.identity(X.shape[1])
        self.weights = np.linalg.inv(X.T @ X / N_train + self.lambd * identity_matrix) @ X.T @ Y / N_train

    def predict(self, X):
        """
        Predict the output for the provided data.
        :param X: The data to predict. np.ndarray of shape (N, D).
        :return: The predicted output. np.ndarray of shape (N,), of 0s and 1s.
        """
        # compute the predicted output of the model.
        preds = np.sign(X @ self.weights)
        # transform the labels to 0s and 1s, instead of -1s and 1s.
        preds = (preds + 1) / 2
        return preds


class Logistic_Regression(nn.Module):
    def __init__(self, input_dim, output_dim, lambd=0.0):
        super(Logistic_Regression, self).__init__()
        # define a linear operation.
        self.linear = nn.Linear(input_dim, output_dim)
        self.lambd = lambd

    def forward(self, x):
        """
        Computes the output of the linear operator.
        :param x: The input to the linear operator.
        :return: The transformed input.
        """
        # compute the output of the linear operator
        # return the transformed input.
        # first perform the linear operation
        # should be a single line of code.
        output = self.linear(x.float())
        return output

    def l2_regularization_loss(self):
        """
        Computes L2 regularization loss.
        """
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2) ** 2
        return self.lambd * l2_loss

    def predict(self, x):
        """
        THIS FUNCTION IS NOT NEEDED FOR PYTORCH. JUST FOR OUR VISUALIZATION
        """
        x = torch.from_numpy(x).float().to(self.linear.weight.data.device)
        x = self.forward(x)
        x = nn.functional.softmax(x, dim=1)
        x = x.detach().cpu().numpy()
        x = np.argmax(x, axis=1)
        return x
