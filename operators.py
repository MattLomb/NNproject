from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.model.sampling import Sampling
import numpy as np


class NormalRandomSampling(Sampling):
    def __init__(self, mu=0, std=1, var_type=np.float):
        super().__init__()
        self.mu = mu
        self.std = std
        self.var_type = var_type

    def _do(self, problem, n_samples, **kwargs):
        return np.random.normal(self.mu, self.std, size=(n_samples, problem.n_var))


def get_operators(config):
    if config.config.split("_")[0] == "StyleGAN2":
        return dict(
            sampling=NormalRandomSampling(),
            crossover=get_crossover("real_sbx", prob=1.0, eta=3.0),
            mutation=get_mutation("real_pm", prob=0.5, eta=3.0)
        )
    else:
        raise Exception("Unknown config")
