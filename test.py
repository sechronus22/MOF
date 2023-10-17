# import sys

# print(sys.path)
# print(sys.executable)

from MOF import MOF
import numpy as np

A = np.array([[1,2,3],[3,2,1],[3,4,5]])
print(MOF(A))