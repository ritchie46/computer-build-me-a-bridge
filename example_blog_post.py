import numpy as np
import os
import pickle
from scipy.spatial.distance import euclidean
from multiprocessing import Pool
from functools import partial
from itertools import combinations, product
import matplotlib.pyplot as plt
from anastruct.fem.system import SystemElements


def build_single_bridge(dna, comb, loc, height, get_ss=False,
                        unit="deflection", EI=15e3, roll=True, support_btm=False):
    """
    Build a single bridge structure.

    :param dna: (array) DNA from population.
    :param comb: (array) All possible combinations.
    :param loc: (array) All possible locations.
    :param mirror_line: (int) x value from the middle line.
    :param height: (int) Maximum height of the bridge.
    :param unit: (str) Make this important in the fitness score evaluation. {deflection, axial compression,
                                                                         tension, moment)
    :param EI: (flt) Bending stiffness of the structure.
    :param roll: (bool) Add a support that is free in x.
    :param support_btm: (bool) Place the support at the bottom of the grid.
    :return: (tpl)
    """
    ss = SystemElements(EI=EI, mesh=3)
    on = np.argwhere(dna == 1).flatten()

    # Add the elements based on the dna
    mirror_line = 0
    for j in on:
        n1, n2 = comb[j]
        l1 = loc[n1]
        l2 = loc[n2]
        ss.add_element([l1, l2])
        mirror_line = max(mirror_line, l1[0], l2[0])

    # add mirrored element
    for j in on:
        n1, n2 = comb[j]
        l1 = loc[n1]
        l2 = loc[n2]
        ss.add_element([mirror(l1, mirror_line), mirror(l2, mirror_line)])

    # Placing the supports on the outer nodes, and the point load on the middle node.
    x_range = ss.nodes_range('x')

    # A bridge of one element is no bridge, it's a beam.
    if len(x_range) <= 2:
        return None
    else:
        length = max(x_range)
        start = min(x_range)
        ids = list(ss.node_map.keys())

        # Find the ids of the middle node for the force application,
        # and the most right node for the support of the bridge
        max_node_id = ids[np.argmax(x_range)]

        for j in range(height):
            middle_node_id = ss.nearest_node("both", np.array([(length + start) / 2, height - j]))
            if middle_node_id:
                break

        if middle_node_id is None:
            middle_node_id = ids[np.argmin(np.abs(np.array(x_range) - (length + start) / 2))]

        # Find the support ids in case the supports should be place in the middle.
        if support_btm:
            left_node_id = 1
            right_node_id = max_node_id
        else:
            idx = np.argsort(np.abs(np.arange(height) - height // 2))

            for j in idx:
                left_node_id = ss.nearest_node("both", np.array([start, j]))
                if left_node_id:
                    break
            for j in idx:
                right_node_id = ss.nearest_node("both", np.array([start + length, j]))
                if right_node_id:
                    break

        # Add support conditions
        ss.add_support_hinged(left_node_id)
        if roll:
            ss.add_support_roll(right_node_id)
        else:
            ss.add_support_hinged(right_node_id)
        ss.point_load(middle_node_id, Fz=-100)

        if ss.validate():
            if get_ss:
                return ss

            ss.solve()

            if unit == "deflection":
                val = np.abs(ss.get_node_displacements(middle_node_id)["uy"])
            elif unit == "axial compression":
                val = -np.min(ss.get_element_result_range("axial"))
            elif unit == "tension":
                val = np.max(ss.get_element_result_range("axial"))
            elif unit == "moment":
                val = np.abs((ss.get_element_result_range("moment")))
            else:
                raise LookupError("Unit is not defined")
            return val, length - start, on.size


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

    def get_fitness(self, ratio=(0.5, 1)):
        """
        Get the fitness score of the current generation.

        :param ratio (tpl) Factor to multiply the unique fitness parts with. The first index is the fitness score
        for the amount of elements. The second is the fitness score for deflection of the bridge.
        :return: (flt)
        """

        unit, length, n_elements = self.build()
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

        fitness = fitness_n * ratio[0] / self.max_fitness_n + \
                  fitness_u * ratio[1] / self.max_fitness_u + \
                  fitness_l / self.length**2

        if self.unit == "deflection":
            fitness[unit == 0] = 0
        if self.fixed_n is not None:
            fitness[n_elements > self.fixed_n] = 0

        return fitness

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


def mirror(v, m_x):
    """
    Mirror an array allong the x-axis.

    :param v: (array) vertex
    :param m_x: (int) mirror x value
    :return: (array) vertex
    """

    return np.array([m_x + m_x - v[0], v[1]])


if __name__ == "__main__":
    base_dir = "./genetic_algorithms"
    PROCESSES = 2  # number of threads
    PARALLEL = False  # Parallel processing.

    # 15e3 is a realistic bending stiffness compared to the prefixed EA (axial stiffness).
    # If you want to simulate low bending stiffnesses, go for values of 1e2 - 1e3.
    EI = 15e3
    roll = False  # One support can freely move in the x direction.
    name = "grid_10_1"

    population = DNA(10, 1, 250, cross_rate=0.8, mutation_rate=0.01, parallel=PARALLEL, unit="deflection", roll=roll,
                     support_btm=True, fixed_n=None, EI=EI)

    os.makedirs(os.path.join(base_dir, "img", name), exist_ok=1)

    last_fitness = 0
    for i in range(1, 50):
        fitness = population.get_fitness(ratio=(1, 1))
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