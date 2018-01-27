import numpy as np
import os
import pickle
from scipy.spatial.distance import euclidean
from multiprocessing import Pool
from functools import partial
from itertools import combinations, product
import matplotlib.pyplot as plt
from environment import build_single_bridge


class DNA:
    def __init__(self, length, height, pop_size=600, cross_rate=0.8, mutation_rate=0.01, parallel=False,
                 unit="deflection", EI=15e3, roll=True, support_btm=True, fixed_n=None):
        """
        Define a population with DNA that represents an element in a bridge.

        :param length: (int) Maximum of the bridge.
        :param height: (int) Maximum height of the bridge.
        :param pop_size: (int) Size of the population.
        :param cross_rate: (flt): Factor of the population that will exchange DNA.
        :param mutation_rate: (flt): Chance of random DNA mutation.
        :param parallel: (bool) Parallelize the computation.
        :param unit: (str) Make this important in the fitness score evaluation. {deflection, axial compression,
                                                                                 tension, moment)
        :param EI: (flt) Bending stiffness of the structure.
        :param roll: (bool) Add a support that is free in x.
        :param support_btm: (bool) Place the support at the bottom of the grid.
        :param fixed_n: (int) Set a maximum limit to the elements build.
        """
        self.normalized = False
        self.max_fitness_n = 0
        self.max_fitness_u = 0
        self.length = length
        self.height = height
        self.pop_size = pop_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        # Assumed that length > height
        # product: permutations with replacement.
        self.loc = np.array(list(filter(lambda x: x[1] <= height and x[0] <= self.length // 2,
                                        product(range(max(self.height + 1, self.length // 2)), repeat=2))))

        # Index tuples of possible connections
        # filters all the vector combinations with an euclidean distance < 1.5.
        # dna
        self.comb = np.array(list(filter(lambda x: euclidean(self.loc[x[1]], self.loc[x[0]]) < 1.5,
                                         combinations(range(len(self.loc)), 2))))

        # Population
        self.pop = np.random.randint(0, 2, size=(pop_size, len(self.comb)))
        self.unit = unit
        self.parallel = parallel
        self.EI = EI
        self.roll = roll
        self.support_btm = support_btm
        self.fixed_n = fixed_n

    def show_grid(self, ratio_n_u=(0.5, 1)):
        ss = build_single_bridge(np.ones(len(self.comb)), self.comb, self.loc, self.height, get_ss=True,
                                 roll=self.roll, support_btm=self.support_btm)
        unit, length, n_elements = build_single_bridge(np.ones(len(self.comb)), self.comb, self.loc, self.height,
                                                       roll=self.roll, support_btm=self.support_btm)

        print("Fitness:", self.evaluate_fitness(np.array([unit]), np.array([length]),
                                                np.array([n_elements]), ratio_n_u))
        ss.show_structure(verbosity=1)

    def build(self):
        """
        Build a bridge based from the current DNA. The bridge will be mirror symmetrical.
        """
        f = partial(build_single_bridge, comb=self.comb, loc=self.loc, height=self.height,
                    EI=self.EI, roll=self.roll, support_btm=self.support_btm)
        if self.parallel:
            with Pool(PROCESSES) as pool:
                sol = pool.map(f, self.pop[np.arange(0, self.pop.shape[0])])
        else:
            sol = list(map(f, self.pop[np.arange(0, self.pop.shape[0])]))

        unit = np.array(list(map(lambda x: x[0] if x is not None else 1e6, sol)))
        length = np.array(list(map(lambda x: x[1] if x is not None else 0, sol)))
        n_elements = np.array(list(map(lambda x: x[2] if x is not None else 1e-6, sol)))

        return unit, length, n_elements

    def evaluate_fitness(self, unit, length, n_elements, ratio_n_u):
        fitness_n = 1 / np.log(n_elements)
        fitness_l = length**2

        if self.unit == "deflection":
            fitness_u = np.sqrt((1.0 / (unit / ((100 * length ** 3) / (48 * self.EI)))))
        else:
            fitness_u = 1 / unit
            fitness_u[fitness_u < 0] = 100

        if not self.normalized:
            self.normalized = True
            # normalize the fitness scores
            self.max_fitness_n = np.max(fitness_n)
            self.max_fitness_u = np.max(fitness_u)

        fitness = fitness_n * ratio_n_u[0] / self.max_fitness_n + \
                  fitness_u * ratio_n_u[1] / self.max_fitness_u + \
                  fitness_l / self.length ** 2

        if self.unit == "deflection":
            fitness[unit == 0] = 0
        if self.fixed_n is not None:
            fitness[n_elements > self.fixed_n] = 0

        return fitness

    def get_fitness(self, ratio_n_u=(0.5, 1)):
        """
        Get the fitness score of the current generation.

        :param ratio_n_u (tpl) Factor to multiply the unique fitness parts with. The first index is the fitness score
        for the amount of elements. The second is the fitness score for deflection of the bridge.
        :return: (flt)
        """

        unit, length, n_elements = self.build()

        return self.evaluate_fitness(unit, length, n_elements, ratio_n_u)

    def evolve(self, fitness):
        pop = rank_selection(self.pop, fitness)
        self.pop = mutate(crossover(pop, self.cross_rate), self.mutation_rate)


def crossover(pop, cross_rate):
    """
    Vectorized crossover

    :param pop: (array)
    :param cross_rate: (flt)
    :return: (array)
    """
    # [bool] Rows that will crossover.
    selection_rows = np.random.rand(pop.shape[0]) < cross_rate

    selection = pop[selection_rows]
    shuffle_seed = np.arange(selection.shape[0])
    np.random.shuffle(shuffle_seed)

    # 2d array with [rows of the (selected) population, bool]
    cross_idx = np.array(np.round(np.random.rand(selection.shape[0], pop.shape[1])), dtype=np.bool)
    idx = np.where(cross_idx)

    selection[idx] = selection[shuffle_seed][idx]
    pop[selection_rows] = selection

    return pop


def mutate(pop, mutation_rate):
    """
    Vectorized random mutations.
    :param pop: (array)
    :param mutation_rate: (flt)
    :return: (array)
    """
    idx = np.where(np.random.rand(pop.shape[0], pop.shape[1]) < mutation_rate)
    val = np.random.randint(0, 2, idx[0].shape[0])
    pop[idx] = val
    return pop


def rank_selection(pop, fitness):
    """
    Rank selection. And make a selection based on their ranking score. Note that this isn't the fitness.

    :param pop: (array) Population.
    :param fitness: (array) Fitness values.
    :return: (array) Population selection with replacement, selected for mating.
    """
    order = np.argsort(fitness)[::-1]
    # Population ordered by fitness.
    pop = pop[order]

    # Rank probability is proportional to you position, not you fitness. So an ordered fitness array, would have these
    # probabilities [1, 1/2, 1/3 ... 1/n] / sum
    rank_p = 1 / np.arange(1, pop.shape[0] + 1)
    # Make a selection based on their ranking.
    idx = np.random.choice(np.arange(pop.shape[0]), size=pop.shape[0], replace=True, p=rank_p / np.sum(rank_p))
    return pop[idx]


if __name__ == "__main__":
    base_dir = "./"
    PROCESSES = 2  # number of threads
    PARALLEL = False  # Parallel processing.

    # 15e3 is a realistic bending stiffness compared to the prefixed EA (axial stiffness).
    # If you want to simulate low bending stiffnesses, go for values of 1e2 - 1e3.
    EI = 1e1
    roll = True  # One support can freely move in the x direction.
    name = "grid_10_1_4(1.5,1)_roll"
    os.makedirs(os.path.join(base_dir, "img", name), exist_ok=1)

    np.random.seed(1)
    population = DNA(10, 1, 250, cross_rate=0.8, mutation_rate=0.01, parallel=PARALLEL, unit="deflection", roll=roll,
                     support_btm=True, fixed_n=None, EI=EI)

    last_fitness = 0
    for i in range(1, 50):
        fitness = population.get_fitness(ratio_n_u=(2, 1))
        max_idx = np.argmax(fitness)
        best_ss = build_single_bridge(population.pop[max_idx], population.comb, population.loc,
                                      population.height, True,
                                      support_btm=population.support_btm, roll=population.roll)
        population.evolve(fitness)
        print("Generation:", i, "Maximum fitness:", fitness[max_idx])
        if last_fitness != fitness[max_idx]:
            try:
                fig = best_ss.show_structure(show=False, verbosity=1)

                plt.title(f"fitness = {round(fitness[max_idx], 3)}")

                fig.savefig(os.path.join(base_dir, "img", name, f"ga{i}.png"))
                with open(os.path.join(base_dir, "img", name, "save.pkl"), "wb") as f:
                    pickle.dump(population, f)
            except AttributeError:
                pass

            last_fitness = fitness[max_idx]




