import numpy as np
from structure_utils import *
last_layer_weights = [[0.9], [0.8], [0.7]]
# feature_matrix = np.array([[1, 2, 0, 4, 8],
#                            [3, 4, 2, 1, 5],
#                            [5, 6, 7, 2, 1]])
feature_matrix = np.array([[1, 2, 3, 4, 4, 5, 5, 5, 6],
                           [3, 4, 2, 3, 3, 6, 5, 5, 3],
                           [5, 6, 7, 5, 5, 5, 5, 5, 0]])

dominated_columns, dominance_equivalent_columns = calc_dominance_brute_force(feature_matrix, last_layer_weights)
np.vstack((dominated_columns, dominance_equivalent_columns))

print("dom", dominated_columns)


feature_matrix = np.array([[1, 2, 8, 3, 4, 4, 5, 5, 5, 6],
                           [3, 4, 0, 2, 3, 3, 6, 5, 5, 3],
                           [5, 6, 1, 7, 5, 5, 5, 7, 5, 0]])