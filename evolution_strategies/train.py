from evolution_strategies.model import Model
from environment import build_single_bridge, det_grid_positions
import numpy as np
import torch
from torch.autograd import Variable
import torch.multiprocessing as mp



def perturbate(model, seed, sigma=0.05):
    """
    Create two new models. And modify both in opposite direction.

    :param model: (Model)
    :param seed: (int)
    :param sigma: (flt)
    :return: [Model, Model]
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

    return [new_model, anti_model]


def generate_seeds_and_models(model, sigma):
    """
    Return a seed and 2 modified models
    :param model: (Model)
    :return: (int, (Model, Model)
    """
    seed = np.random.randint(2**30)
    return seed, perturbate(model, seed, sigma)


def get_fitness(x, w, l, h, n,):
    """
    x_input, l and h should be normalized

    indexes input
    0: stiff -> 0, 1 (no, yes)
    1: height
    2: length

    :param x: (array)
    :param w: (flt)
    :param l: (flt)
    :param n: (flt)
    :return: (flt)
    """

    out = np.zeros(3)

    w0 = (100 * l**3) / (48 * EI)
    if w < w0:
        out[0] = 0
    else:
        out[0] = 1
    out[1] = h
    out[2] = l

    return 1 / (np.mean((x - out) ** 2) + n / N_MAX)


def evaluate(x, dna, seed, queue, is_negative):
    """

    :param x: (array) Normalized input of the model.
    :param dna: (array) Output of the model. Rounded to binary values.
    :param seed: (int) Seed of this percurbation.
    :param queue: (Queue) Queue for parallel process.
    :param is_negative: (bool) Model or anti-model.
    :return: None
    """

    result = build_single_bridge(dna, COMB, LOC, HEIGHT_GRID, es=True, support_btm=True)

    if result is not None:
        if isinstance(result, tuple):
            fitness = get_fitness(x, result[0], result[1], result[2], result[3])
        else:
            fitness = result
    else:
        fitness = 0

    queue.put([seed, fitness, is_negative])


def normalize(x):
    return np.array([x[0], x[1] / HEIGHT_GRID, x[2] / LENGTH_GRID])


def sample_input(h, l):
    """

    :param h: (int) Max height of the grid.
    :param l: (int) Max length of the grid.
    :return: (array) [[stiff, h, l]]
    """
    h = np.random.randint(h + 1)
    l = np.random.randint(l + 1)
    stiff = np.random.randint(2)
    return np.array([stiff, h, l])


def train_loop(model, dn, iterations=100000, pop_size=40, sigma=0.05):

    for i in range(iterations):
        x = normalize(sample_input(HEIGHT_GRID, LENGTH_GRID))

        processes = []
        queue = mp.Queue()
        all_seeds, all_models = [], []

        # generate a perturbation and its antithesis
        for j in range(pop_size // 2):
            seed, models = generate_seeds_and_models(model, sigma)
            all_seeds += [seed, seed]
            all_models += models

        assert len(all_seeds) == len(all_models)

        # Keep track of which perturbations were positive and negative
        # Start with negative true because the pop() method makes us go backwards
        # and the function perturbate returns [positive perturbated model, negative perturbated model].
        is_negative = True

        # Add all perturbed models to the queue.
        while all_models:
            perturbed_model = all_models.pop()
            seed = all_seeds.pop()
            dna = np.round(perturbed_model(Variable(torch.from_numpy(x), volatile=True).float()).data.numpy())
            p = mp.Process(target=evaluate,
                           args=(x, dna, seed, queue, is_negative))

            p.start()
            processes.append(p)
            is_negative = not is_negative
            p.join()

        assert len(all_seeds) == 0

        # Evaluate the unmodified model as well
        p = mp.Process(target=evaluate,
                       args=(x, dna, "dummy", queue, "dummy"))
        p.start()
        processes.append(p)
        for p in processes:

            p.join()

        results = [queue.get() for _ in processes]
        seeds = list(map(lambda x: x[0], results))
        fitness = list(map(lambda x: x[1], results))
        neg_list = list(map(lambda x: x[2], results))

        # separate the original from the perturbed results.
        idx = seeds.index("dummy")
        seeds.pop(idx)
        original_f = fitness.pop(idx)
        neg_list.pop(idx)
        print(results)


if __name__ == "__main__":

    #   indexes input
    #   0: stiff -> 0, 1 (no, yes)
    #   1: height
    #   2: length

    # 10 * 8
    HEIGHT_GRID = 10
    LENGTH_GRID = 10
    LOC, COMB = det_grid_positions(LENGTH_GRID, HEIGHT_GRID)
    N_MAX = len(COMB)
    EI = 1e5

    m = Model(3, N_MAX, 1, (35,))
    print(N_MAX)

    train_loop(m, "dn", 1, 40, 0.05)



