import torch
import numpy as np

from scipy.optimize import fsolve


class TReLU(torch.nn.Module):
    """
    PyTorch implementation of the scaled TReLU activation function as reported in
    Deep Learning without Shortcuts section 4.1
    """
    def __init__(self, alpha, trainable, device):
        super().__init__()
        self.device = device
        if trainable:
            self.alpha = torch.nn.Parameter(torch.tensor([alpha]), requires_grad=True)
        else:
            self.alpha = torch.tensor([alpha], device=self.device)

    def forward(self, x):
        return torch.sqrt(2. / (1. + self.alpha ** 2.)) * \
            (torch.maximum(x, torch.tensor([0], device=self.device)) +
             self.alpha * torch.minimum(x, torch.tensor([0], device=self.device)))

def optimal_trelu_params():
    """
    This function implements the approach described in Deep Learning Without Shortcuts section 4.1, equation 12.
    Given a hyperparameter eta (0 <= eta <= 1) the claimed optimal static value for alpha is the solution of the equation

    C_f(alpha) - eta = 0

    where C_f is the composition of the C maps of the TReLU activation with a given alpha starting at c=0.
    The initial guesses for the possible solutions were motivated by the graph of C_f.
    """
    def C(c, alpha):
        """Returns the value of the C map for the TReLU function with given alpha"""
        return c + (((1 - alpha) ** 2) / (torch.pi * (1 + alpha ** 2))) * (np.sqrt(1 - c ** 2) - c * np.arccos(c))

    def C_f(alpha, c=0):
        """Computes the composition of the C maps of the TReLU function with given alpha starting from c=0"""
        cf = C(c, alpha)
        for _ in range(50):
            cf = C(cf, alpha)
        return cf

    return fsolve(lambda x: C_f(x) - 0.9, x0=np.array([0.0, 1.0]))