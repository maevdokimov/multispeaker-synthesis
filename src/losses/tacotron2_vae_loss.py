import torch
from nemo.collections.tts.helpers.helpers import get_mask_from_lengths
from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types.elements import LengthsType, LogitsType, LossType, MelSpectrogramType
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

    def kl_anneal_function(self, anneal_function, lag, step, k, x0, upper):
        if anneal_function == "logistic":
            return upper / (upper + torch.exp(-k * (step - x0)))
        elif anneal_function == "linear":
            if step > lag:
                return min(upper, step / x0)
            else:
                return 0
        elif anneal_function == "constant":
            return 0.001

    def forward(self, mu, logvar, step):
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_weight = self.kl_anneal_function(self.anneal_function, self.lag, step, self.k, self.x0, self.upper)

        return kl_loss, kl_weight
