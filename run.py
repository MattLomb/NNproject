import argparse
import os
import pickle
import numpy as np
import tensorflow as tf

from pymoo.factory import get_algorithm, get_decision_making, get_decomposition
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from problem import GenerationProblem
from operators import get_operators
from config import get_config

parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str, default="StyleGAN2_ffhq_d")  # MODEL
parser.add_argument("--generations", type=int, default=500)  # Number of images generated
parser.add_argument("--save-each", type=int, default=50)  # Images saved each 50 generations
parser.add_argument("--tmp-folder", type=str, default="./tmp")  # Folder in which save the generated images
parser.add_argument("--target", type=str, default="A male with mustache")  # txt2img

config = parser.parse_args()
vars(config).update(get_config(config.config))

iteration = 0


def save_callback(algorithm):
    global iteration
    global config

    iteration += 1
    if iteration % config.save_each == 0 or iteration == config.generations:
        if config.problem_args["n_obj"] == 1:
            sortedpop = sorted(algorithm.pop, key=lambda p: p.F)
            x = np.stack([p.X for p in sortedpop])
        else:
            x = algorithm.pop.get("X")

        ls = tf.convert_to_tensor(x)

        generated = algorithm.problem.generator.generate(ls)
        ext = "jpg"
        name = "genetic-it-%d.%s" % (
            iteration, ext) if iteration < config.generations else "genetic-it-final.%s" % (ext,)
        algorithm.problem.generator.save(generated, os.path.join(config.tmp_folder, name))


problem = GenerationProblem(config)
operators = get_operators(config)

if not os.path.exists(config.tmp_folder):
    os.mkdir(config.tmp_folder)

algorithm = get_algorithm(
    config.algorithm,
    pop_size=config.pop_size,
    sampling=operators["sampling"],
    crossover=operators["crossover"],
    mutation=operators["mutation"],
    eliminate_duplicates=True,
    callback=save_callback,
    **(config.algorithm_args[
           config.algorithm] if "algorithm_args" in config and config.algorithm in config.algorithm_args else dict())
)

res = minimize(
    problem,
    algorithm,
    ("n_gen", config.generations),
    save_history=False,
    verbose=True,
)

pickle.dump(dict(
    X=res.X,
    F=res.F,
    G=res.G,
    CV=res.CV,
), open(os.path.join(config.tmp_folder, "genetic_result"), "wb"))

if config.problem_args["n_obj"] == 2:
    plot = Scatter(labels=["similarity", "discriminator", ])
    plot.add(res.F, color="red")
    plot.save(os.path.join(config.tmp_folder, "F.jpg"))

if config.problem_args["n_obj"] == 1:
    sortedpop = sorted(res.pop, key=lambda p: p.F)
    X = np.stack([p.X for p in sortedpop])
else:
    X = res.pop.get("X")

rnd = np.random.RandomState()
ls = rnd.randn(config.batch_size, config.dim_z).astype('float32')
ls = ls.astype(float)

