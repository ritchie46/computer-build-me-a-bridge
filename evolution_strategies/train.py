from evolution_strategies.model import Model
from environment import build_single_bridge, det_grid_positions
import numpy as np
import torch


def perturbate(model, seed, sigma=0.05):
    """
    Create two new models. And modify both in opposite direction.

    :param model: (Model)
    :param seed: (int)
    :param sigma: (flt)
    :return: (Model, Model)
    """
    args = [model.n_in, model.n_out, model.hidden_layers, model.hidden_neurons]

    new_model = Model(*args)
    anti_model = Model(*args)
    new_model.load_state_dict(model.state_dict())
    anti_model.load_state_dict(model.state_dict())
    np.random.seed(seed)

    for (k, v), (anti_k, anti_v) in zip(new_model.es_params(),
                                        anti_model.es_params()):

        # draw from a gaussian distribution eps ~phi(mean, std)
        eps = np.random.normal(0, 1, v.size())
        noise = torch.from_numpy(sigma * eps).float()
        v += noise
        anti_v -= noise

    return new_model, anti_model


def generate_seeds_and_models(model, sigma):
    """
    Return a seed and 2 modified models
    :param model: (Model)
    :return: (int, (Model, Model)
    """
    seed = np.random.randint(2**30)
    return seed, perturbate(model, sigma)


def get_fitness(x, w, l, h, n,):
    """
    x_input and w, l and h should be normalized

    indexes input
    0: stiff -> 0, 1 (no, yes)
    1: height
    2: length


    :param x:
    :param w:
    :param l:
    :param n:
    :return:
    """

    out = np.zeros(3)

    w0 = (100 * l**3) / (48 * EI)
    if w < w0:
        out[0] = 0
    else:
        out[0] = 1
    out[1] = h
    out[2] = l

    return -(np.mean((x - out) ** 2) + n / n_max)


def evaluate(x, dna, seed, queue, is_negative):
    """

    :param x: (array) Normalized input of the model.
    :param dna: (array) Output of the model. Rounded to binary values.
    :param seed: (int) Seed of this percurbation.
    :param queue: (Queue) Queue for parallel process.
    :param is_negative: (bool) Model or anti-model.
    :return: None
    """
    w, l, n, h = build_single_bridge(dna, comb, loc, height_grid, return_height=True)
    fitness = get_fitness(x, w, l, h, n)

    queue.put([seed, fitness, is_negative])


def normalize(x, n_max, max_length=10, max_height=8):
    x[0] /= max_length
    x[1] /= max_height
    x[3] /= n_max


# 10 * 8
loc, comb = det_grid_positions(10, 8)
n_max = len(comb)
height_grid = 8

EI = 1e5

print(len(loc), len(comb))
# dna = model(input)

m = Model(2, 10, 1)

perturbate(m, 1, 0.05)

