from evolution_strategies.model import Model
from environment import build_single_bridge, det_grid_positions
import numpy as np
import torch
from torch.autograd import Variable
import torch.multiprocessing as mp
from torch.legacy.optim import adam
import matplotlib.pyplot as plt
import os


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

    h = np.random.randint(h // 3, h + 1)
    l = np.random.randint(l // 3, l + 1)
    stiff = np.random.randint(2)

    while True:
        yield np.array([stiff, h, l])


def rank_fitness(fitness):
    rank_p = 1 / np.arange(1, fitness.shape[0] + 1)
    order = np.argsort(fitness)[::-1]
    return rank_p[order]


def gradient_update(model, fitness, seeds, neg_list, original_fitness, sigma=0.05, plot=False,
                    dn="img"):

    batch_size = len(fitness)
    assert len(seeds) == batch_size

    rank = rank_fitness(fitness)
    original_rank = np.argwhere(np.sort(fitness)[::-1] == original_fitness)[0][0] + 1

    print(f"Original rank: {original_rank} out of {fitness.shape[0]}. Score: {original_fitness}")

    if plot:
        os.makedirs("dn", exist_ok=True)
        AVG_F.append(np.mean(fitness))
        CURR_F.append(original_fitness)
        MAX_F.append(np.max(fitness))

        plt_avg = plt.plot(EPISODES, AVG_F, label="Average", color="g")
        plt_curr = plt.plot(EPISODES, CURR_F, label="Current", color="b")
        plt_max = plt.plot(EPISODES, MAX_F, label="Max", color="r")

        plt.ylabel('fitness')
        plt.xlabel('episode num')
        #plt.legend()

        fig1 = plt.gcf()

        plt.draw()
        fig1.savefig(dn + '/graph.png', dpi=100)

    # Optimize using Adam
    global_gradients = None  # Final gradients for original model
    for i in range(fitness.shape[0]):
        # Make the same noise using the seeds
        np.random.seed(seeds[i])
        factor = -1 if neg_list[i] else 1
        rank_f = rank[i]

        local_gradients = []  # gradients based on this model (mini batch)

        for k, v in model.es_params():  # loop over weights and biases of all layers
            eps = np.random.normal(0, 1, v.size())
            grad = torch.from_numpy(sigma * rank_f * factor * eps * fitness.shape[0]).float()
            local_gradients.append(grad)

        if global_gradients:
            for j in range(len(global_gradients)):
                global_gradients[j] = torch.add(global_gradients[j], local_gradients[j])
        else:
            global_gradients = local_gradients

    c = 0
    for k, v in model.es_params():
        shift, _ = adam(lambda x: (1, -global_gradients[c]), v, {'learningRate': 0.001})
        v.copy_(shift)
        c += 1

    # lr = 0.1
    # for i in range(fitness.shape[0]):
    #     np.random.seed(seeds[i])
    #     factor = -1 if neg_list[i] else 1
    #     rank_f = rank[i]
    #
    #     for k, v in model.es_params():
    #         eps = np.random.normal(0, 1, v.size())
    #         v += torch.from_numpy(lr / (fitness.shape[0] * sigma) *
    #                               (rank_f * factor * eps)).float()

    torch.save(model.state_dict(), "dn/latest.pth")


def train_loop(model, dn="img", iterations=100000, pop_size=40, sigma=0.05, plot=False):

    for i in range(iterations):
        sample = sample_input(HEIGHT_GRID, LENGTH_GRID)
        x = normalize(next(sample))

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

        EPISODES.append(i)
        gradient_update(model, np.array(fitness), seeds, neg_list, original_f, sigma, plot, dn)


if __name__ == "__main__":
    CURR_F = []
    MAX_F = []
    AVG_F = []
    EPISODES = []

    #   indexes input
    #   0: stiff -> 0, 1 (no, yes)
    #   1: height
    #   2: length

    # 10 * 8
    HEIGHT_GRID = 5
    LENGTH_GRID = 10
    LOC, COMB = det_grid_positions(LENGTH_GRID, HEIGHT_GRID)
    N_MAX = len(COMB)
    EI = 1e5

    m = Model(3, N_MAX, 1, (35,))

    train_loop(m, "dn", 100000, 40, 0.05, plot=True)



