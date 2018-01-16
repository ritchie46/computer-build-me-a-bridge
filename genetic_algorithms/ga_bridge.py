import numpy as np
import os
import pickle
from scipy.spatial.distance import euclidean
from multiprocessing import Pool
import time
from functools import partial
from itertools import combinations, product
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/ritchie46/code/python/anaStruct")
from anastruct.fem.system import SystemElements

PROCESSES = 2


def build_single_bridge(dna, comb, loc, mirror_line, height, get_ss=False,
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
    on = np.argwhere(dna == 1)

    for j in on.flatten():
        n1, n2 = comb[j]
        l1 = loc[n1]
        l2 = loc[n2]

        ss.add_element([l1, l2])
        # add mirrored element
        ss.add_element([mirror(l1, mirror_line), mirror(l2, mirror_line)])

    # Placing the supports on the outer nodes, and the point load on the middle node.
    x_range = ss.nodes_range('x')

    if len(x_range) <= 2:
        return None
    else:
        length = max(x_range)
        start = min(x_range)
        ids = list(ss.node_map.keys())

        # Find the ids of the middle node for the force application, and the most right node for the support of the
        # bridge
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
            return val, length, on.size


class DNA:
    def __init__(self, length, height, pop_size=600, cross_rate=0.8, mutation_rate=0.01, parallel=False,
                 unit="deflection", EI=15e3, roll=True):
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
        """
        self.normalized = False
        self.max_fitness_n = 0
        self.max_fitness_w = 0
        self.length = length
        self.height = height
        self.mirror_line = length // 2
        self.pop_size = pop_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        # Assumed that length > height
        # product: permutations with replacement.
        self.loc = np.array(list(filter(lambda x: x[1] <= height and x[0] <= self.mirror_line,
                                        product(range(max(self.height + 1, self.mirror_line)), repeat=2))))

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

    def build(self):
        """
        Build a bridge based from the current DNA. The bridge will be mirror symmetrical.
        """
        f = partial(build_single_bridge, comb=self.comb, loc=self.loc, mirror_line=self.mirror_line, height=self.height,
                    EI=self.EI, roll=self.roll)
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

        if self.unit == "deflection":
            fitness_u = np.sqrt((1.0 / (unit / ((100 * length**3) / (48 * self.EI)))))
        else:
            fitness_u = 1 / unit
            fitness_u[fitness_u < 0] = 1
        if not self.normalized:
            self.normalized = True
            # normalize the fitness scores
            self.max_fitness_n = np.max(fitness_n)
            self.max_fitness_w = np.max(fitness_u)

        fitness = fitness_n * ratio[0] / self.max_fitness_n + fitness_u * ratio[1] / self.max_fitness_w
        fitness[np.argwhere(length < self.length)] = 0
        if self.unit == "deflection":
            fitness[np.argwhere(unit == 0)] = 0

        return fitness

    def crossover(self, parent, pop):
        """
        Swap DNA between parents from the population.
        :param parent: (array)
        :param pop: (array) Containing parents
        :return: (array)
        """
        if np.random.rand() < self.cross_rate:
            # Draw a lucky guy to mate.
            i = np.random.randint(0, self.pop_size, size=1)
            # An array with random boolean values.
            cross_index = np.random.randint(0, 2, size=self.comb.shape[0]).astype(np.bool)
            # The True indexes will be replaced by a random sample i from the population.
            parent[cross_index] = pop[i, cross_index]

        return parent

    def mutate(self, child):
        """
        Do random mutations.
        :param child: (array) Return by crossover.
        :return: (array)
        """
        i = np.where(np.random.random(self.comb.shape[0]) < self.mutation_rate)[0]
        child[i] = np.random.randint(0, 2, size=i.shape)
        return child

    def evolve_loop(self, fitness):
        """
        Evaluate a generation.
        :param fitness: (array) Fitness score of the current generation.
        :return: (array) New population.
        """
        pop = rank_selection(self.pop, fitness)
        pop_copy = pop.copy()

        for i in range(pop.shape[0]):
            parent = pop[i]
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child

        self.pop = pop

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


def choose_fit_parent(pop):
    """
    https://www.electricmonk.nl/log/2011/09/28/evolutionary-algorithm-evolving-hello-world/

    :param pop: population sorted by fitness
    :return:
    """
    # product uniform distribution
    i = int(np.random.random() * np.random.random() * (pop.shape[1] - 1))
    return pop[i]


def mirror(v, m_x):
    """
    Mirror an array allong the x-axis.

    :param v: (array) vertex
    :param m_x: (int) mirror x value
    :return: (array) vertex
    """

    return np.array([m_x + m_x - v[0], v[1]])


np.random.seed(1)
a = DNA(6, 10, 300, cross_rate=0.8, mutation_rate=0.02, parallel=1, unit="axial compression")


base_dir = "/home/ritchie46/code/machine_learning/bridge/genetic_algorithms/img"
name = "test"
os.makedirs(os.path.join(base_dir, f"{name}"), exist_ok=1)


# with open(os.path.join(base_dir, f"{name}", "save.pkl"), "rb") as f:
#     a = pickle.load(f)
#
#
# print(a)
# fitness = a.get_fitness(ratio=(1, 1))
# max_idx = np.argmax(fitness)
# best_ss = build_single_bridge(a.pop[max_idx], a.comb, a.loc, a.mirror_line, a.height, True)
#
# best_ss.solve()
# best_ss.show_axial_force()



last_fitness = 0
for i in range(1, 150):
    t0 = time.time()
    fitness = a.get_fitness(ratio=(1, 1))
    max_idx = np.argmax(fitness)
    best_ss = build_single_bridge(a.pop[max_idx], a.comb, a.loc, a.mirror_line, a.height, True)
    a.evolve(fitness)
    print(time.time() - t0)
    print("gen", i, "max fitness", fitness[max_idx])

    if last_fitness != fitness[max_idx]:
        try:
            fig = best_ss.show_structure(show=False, verbosity=1)
            best_ss.solve()
            # fig = best_ss.show_axial_force(show=False)
            plt.title(f"fitness = {round(fitness[max_idx], 3)} {np.min(best_ss.get_element_result_range('axial'))}")
            # plt.show()
            fig.savefig(os.path.join(base_dir, f"{name}", f"ga{i}.png"))
            with open(os.path.join(base_dir, f"{name}", "save.pkl"), "wb") as f:
                pickle.dump(a, f)
        except AttributeError:
            pass

        last_fitness = fitness[max_idx]




