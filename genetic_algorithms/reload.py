import pickle
from genetic_algorithms.ga_bridge import *

base_dir = "/home/ritchie46/code/machine_learning/bridge/genetic_algorithms/img/"

dn = "test"
with open(base_dir + dn + "/save.pkl", "rb") as f:
    a = pickle.load(f)
print(ss)
ss.show_structure()