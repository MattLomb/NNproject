from pymoo.core.problem import Problem
from generator import Generator
import numpy as np
import tensorflow as tf


class GenerationProblem(Problem):
    def __init__(self, config):
        self.generator = Generator(config)
        self.config = config

        super().__init__(**self.config.problem_args)

    def _evaluate(self, x, out, *args, **kwargs):
        rnd = np.random.RandomState()
        ls = rnd.randn(self.config.batch_size, self.config.dim_z).astype('float32')

        generated = self.generator.generate(ls)

        sim = self.generator.clip_similarity(generated)

        if self.config.problem_args["n_obj"] == 2 and self.config.use_discriminator:
            dis = self.generator.discriminate(generated)
            hinge = tf.nn.relu(1 - dis)
            hinge = tf.squeeze(hinge).numpy()
            out["F"] = np.column_stack((-sim, hinge))
        else:
            out["F"] = -sim

        out["G"] = np.zeros((x.shape[0]))
