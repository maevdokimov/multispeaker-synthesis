import math

import torch
from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types.elements import LossType, NormalDistributionLogVarianceType, NormalDistributionMeanType
from nemo.core.neural_types.neural_type import NeuralType


class Tacotron2VAELoss(Loss):
    def __init__(
        self,
        anneal_function="logistic",
        anneal_k=0.0025,
        anneal_x0=10000,
        anneal_upper=0.2,
        anneal_lag=50000,
    ):
        super().__init__()

        self.anneal_function = anneal_function
        self.lag = anneal_lag
        self.k = anneal_k
        self.x0 = anneal_x0
        self.upper = anneal_upper

    def kl_anneal_function(self, step):
        if self.anneal_function == "logistic":
            return self.upper / (self.upper + math.exp(-self.k * (step - self.x0)))
        elif self.anneal_function == "linear":
            if step > self.lag:
                return min(self.upper, step / self.x0)
            else:
                return 0
        elif self.anneal_function == "constant":
            return 0.001

    @property
    def input_types(self):
        return {
            "mu": NeuralType(("B", "D"), NormalDistributionMeanType()),
            "logvar": NeuralType(("B", "D"), NormalDistributionLogVarianceType()),
        }

    @property
    def output_types(self):
        return {
            "kl_loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, mu, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return kl_loss
