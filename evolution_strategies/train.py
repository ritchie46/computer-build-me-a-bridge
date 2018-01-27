from evolution_strategies.model import Model
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


m = Model(2, 10, 1)

perturbate(m, 1, 0.05)

