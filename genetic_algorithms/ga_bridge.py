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
EI = 15e3


def build_single_bridge(dna, comb, loc, mirror_line, height, get_ss=False):
    """
    Build a single bridge structure.

    :param dna: (array) DNA from population.
    :param comb: (array) All possible combinations.
    :param loc: (array) All possible locations.
    :param mirror_line: (int) x value from the middle line.
    :param height: (int) Maximum height of the bridge.
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

        middle_node_id = None
        for j in range(height):
            middle_node_id = ss.nearest_node("both", np.array([(length + start) / 2, height - j]))
            if middle_node_id:
                break

        if middle_node_id is None:
            middle_node_id = ids[np.argmin(np.abs(np.array(x_range) - (length + start) / 2))]

        ss.add_support_hinged(1)
        ss.add_support_hinged(max_node_id)
        ss.point_load(middle_node_id, Fz=-100)

        if ss.validate():
            if get_ss:
                return ss

            ss.solve()
            w = np.abs(ss.get_node_displacements(middle_node_id)["uy"])
            return w, length, on.size


class DNA:
    def __init__(self, length, height, pop_size=600, cross_rate=0.8, mutation_rate=0.01, parallel=False):
        """
        Define a population with DNA that represents an element in a bridge.

        :param length: (int) Maximum of the bridge.
        :param height: (int) Maximum height of the bridge.
        :param pop_size: (int) Size of the population.
        :param cross_rate: (flt): Factor of the population that will exchange DNA.
        :param mutation_rate: (flt): Chance of random DNA mutation.
        :param parallel: (bool) Parallelize the computation.
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
        self.loc = np.array(list(filter(lambda x: x[1] <= height, product(range(self.mirror_line + 1), repeat=2))))

        # Index tuples of possible connections
        # filters all the vector combinations with an euclidean distance < 1.5.
        # dna
        self.comb = np.array(list(filter(lambda x: euclidean(self.loc[x[1]], self.loc[x[0]]) < 1.5,
                                         combinations(range(len(self.loc)), 2))))

        # Population
        self.pop = np.random.randint(0, 2, size=(pop_size, len(self.comb)))

        self.parallel = parallel

    def build(self):
        """
        Build a bridge based from the current DNA. The bridge will be mirror symmetrical.
        """
        f = partial(build_single_bridge, comb=self.comb, loc=self.loc, mirror_line=self.mirror_line, height=self.height)
        if self.parallel:
            with Pool(PROCESSES) as pool:
                sol = pool.map(f, self.pop[np.arange(0, self.pop.shape[0])])
        else:
            sol = list(map(f, self.pop[np.arange(0, self.pop.shape[0])]))

        w = np.array(list(map(lambda x: x[0] if x is not None else 1e6, sol)))
        length = np.array(list(map(lambda x: x[1] if x is not None else 0, sol)))
        n_elements = np.array(list(map(lambda x: x[2] if x is not None else 1e-6, sol)))

        return w, length, n_elements

    def get_fitness(self, ratio=(0.5, 1)):
        """
        Get the fitness score of the current generation.

        :param ratio (tpl) Factor to multiply the unique fitness parts with. The first index is the fitness score
        for the amount of elements. The second is the fitness score for deflection of the bridge.
        :return: (flt)
        """
        w, length, n_elements = self.build()
        fitness_n = (length**2 / self.length) * (1 / np.log(n_elements))
        fitness_w = np.sqrt((1.0 / (w / ((100 * length**3) / (48 * EI)))))

        if not self.normalized:
            self.normalized = True
            # normalize the fitness scores
            self.max_fitness_n = np.max(fitness_n)
            self.max_fitness_w = np.max(fitness_w)

        fitness = fitness_n * ratio[0] / self.max_fitness_n + fitness_w * ratio[1] / self.max_fitness_w
        fitness[np.argwhere(w == 0)] = 0

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

    def evolve(self, fitness):
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

    def evolve_(self, fitness):
        pop = rank_selection(self.pop, fitness)
        self.pop = crossover(pop, self.cross_rate)


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


def validate_calc(ss):
    """
    Validate if this is a proper structure.

    :param ss: (SystemElement)
    :return: (bool)
    """
    try:
        displacement_matrix = ss.solve()
        return not np.any(np.abs(displacement_matrix) > 1e9)
    except (np.linalg.LinAlgError, AttributeError):
        return False


def normalize(x):
    if np.allclose(x, x[0]):
        return np.ones(x.shape)*0.1
    return (x - np.min(x)) / (np.max(x) - np.min(x))


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
a = DNA(10, 10, 200, cross_rate=0.8, mutation_rate=0.02, parallel=0)


base_dir = "/home/ritchie46/code/machine_learning/bridge/genetic_algorithms/img"
name = "test"
os.makedirs(os.path.join(base_dir, f"{name}"), exist_ok=1)

# with open(os.path.join(base_dir, f"best_{name}", "save.pkl"), "rb") as f:
#     a = pickle.load(f)
#     a.mutation_rate = 0.01
#     a.cross_rate=0.7
last_fitness = 0
for i in range(1, 150):
    t0 = time.time()
    fitness = a.get_fitness(ratio=(1, 1))
    max_idx = np.argmax(fitness)
    best_ss = build_single_bridge(a.pop[max_idx], a.comb, a.loc, a.mirror_line, a.height, True)
    a.evolve_(fitness)
    print(time.time() - t0)
    print("gen", i, "max fitness", fitness[max_idx])

    if last_fitness != fitness[max_idx]:
        try:
            fig = best_ss.show_structure(show=False, verbosity=1)
            plt.title(f"fitness = {round(fitness[max_idx], 3)}")
            fig.savefig(os.path.join(base_dir, f"{name}", f"ga{i}.png"))
            with open(os.path.join(base_dir, f"{name}", "save.pkl"), "wb") as f:
                pickle.dump(a, f)
        except AttributeError:
            pass

        last_fitness = fitness[max_idx]




