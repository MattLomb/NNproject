import argparse
import os
import numpy as np
import pickle
'''
from pymoo.optimize import minimize
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_algorithm, get_decision_making, get_decomposition
from pymoo.visualization.scatter import Scatter
'''
import models
from models import StyleGAN2
config = "kfkdf"
a=StyleGAN2(config)
