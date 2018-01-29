import numpy as np
#import sys
from itertools import combinations, product
from scipy.spatial.distance import euclidean
#sys.path.append("/home/ritchie46/code/anaStruct")
from anastruct.fem.system import SystemElements


def mirror(v, m_x):
    """
    Mirror an array allong the x-axis.

    :param v: (array) vertex
    :param m_x: (int) mirror x value
    :return: (array) vertex
    """

    return np.array([m_x + m_x - v[0], v[1]])


def build_single_bridge(dna, comb, loc, height, get_ss=False,
                        unit="deflection", EI=15e3, roll=True, support_btm=False, es=False):
    """
    Build a single bridge structure.

    :param dna: (array) DNA from population.
    :param comb: (array) All possible combinations.
    :param loc: (array) All possible locations.
    :param height: (int) Maximum height of the bridge.
    :param unit: (str) Make this important in the fitness score evaluation. {deflection, axial compression,
                                                                         tension, moment)
    :param EI: (flt) Bending stiffness of the structure.
    :param roll: (bool) Add a support that is free in x.
    :param support_btm: (bool) Place the support at the bottom of the grid.
    :param es: (bool) Special code for Evolution Strategies.
    :return: (unit, length, number of elements)
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
        if roll and right_node_id is not None:
            ss.add_support_roll(right_node_id)
        elif right_node_id is not None:
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

            if es:
                return val, length - start, on.size, max(ss.nodes_range('y'))

            return val, length - start, on.size

 
def det_grid_positions(length, height):
    loc = np.array(list(filter(lambda x: x[1] <= height and x[0] <= length // 2,
                                    product(range(max(height + 1, length // 2)), repeat=2))))

    # Index tuples of possible connections
    # filters all the vector combinations with an euclidean distance < 1.5.
    # dna
    comb = np.array(list(filter(lambda x: euclidean(loc[x[1]], loc[x[0]]) < 1.5,
                                     combinations(range(len(loc)), 2))))
    return loc, comb
