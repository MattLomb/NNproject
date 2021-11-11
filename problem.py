from pymoo.model.problem import Problem
from generator import Generator
import numpy as np
import tensorflow as tf


class GenerationProblem(Problem):
    def __init__(self, config):
        self.generator = Generator(config)
        self.config = config
        super().__init__(**self.config.problem_args)

    def _evaluate(self, x, out, *args, **kwargs):
        # X shape (pop_size, n_var)
        x_tensor = tf.convert_to_tensor(x)
        generated = self.generator.generate(x_tensor)
        sim = self.generator.clip_similarity(generated)
        if self.config.use_discriminator:
            dis = self.generator.discriminate(generated)
            hinge = tf.nn.relu(1 - dis)
            hinge = tf.squeeze(hinge).numpy()
            out["F"] = np.column_stack((sim, hinge))
        else:
            out["F"] = sim
        out["G"] = np.zeros((x.shape[0]))  # Constrains
