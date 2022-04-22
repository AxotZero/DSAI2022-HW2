from copy import deepcopy
from pdb import set_trace as bp

import numpy as np


def load_data(path):
    return np.genfromtxt(path, delimiter=',')