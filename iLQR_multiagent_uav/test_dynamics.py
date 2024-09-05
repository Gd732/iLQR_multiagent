from ilqr.vehicle_model import QuadcopterModel
import numpy as np
import argparse
from arguments import add_arguments
import time

argparser = argparse.ArgumentParser([])
add_arguments(argparser)
args = argparser.parse_args([])
qm = QuadcopterModel(args)
x = np.random.rand(12,10); u = np.random.rand(4,10)

t1 = time.time()
for i in range(10000):
    B_py = qm.get_B_matrix_py(x,u,10)
tcost = time.time()-t1
print(tcost)

t1 = time.time()
for i in range(10000):
    B_cpp = qm.get_B_matrix(x,u,10)
tcost = time.time()-t1
print(tcost)

t1 = time.time()
for i in range(10000):
    B_cpp_steps = qm.get_B_matrix_steps(x,u,10)
tcost = time.time()-t1
print(tcost)