import numpy as np
from sortedcontainers import SortedSet


# Test whether last layer weights and biases are the same across all encountered situations.
def check_consistency_of_last_layer(al):
    trueVectorWeights = np.zeros(shape = [len(al)-1], dtype = bool)
    trueVectorBiases = np.zeros(shape = [len(al)-1], dtype = bool)
    trueVectorGreaterZero = np.zeros(shape = [len(al)], dtype = bool)
    for ix, a in enumerate(al):
        trueVectorGreaterZero[ix] = np.all(np.greater(a.feature_matrix, 0))
        if ix > 0:
            a_minus_one = al[ix-1]
            trueVectorWeights[ix-1] = np.all(np.equal(a.last_layer_weights, a_minus_one.last_layer_weights))
            trueVectorBiases[ix-1] = np.all(np.equal(a.last_layer_bias, a_minus_one.last_layer_bias))
    if np.all(trueVectorWeights) & np.all(trueVectorBiases) & np.all(trueVectorGreaterZero):
        return True
    else:
        return False

# check_consistency_of_last_layer(al)


last_layer_weights = [[0.7], [-0.9], [0.8]]
feature_matrix = np.array([[1, 2, 0, 4, 8], [3, 4, 2, 1, 5], [5, 6, 7, 2, 1]])
def calc_dominance_brute_force(feature_matrix, last_layer_weights):
    n_hidden, n_actions = feature_matrix.shape

    # Order rows by descending weight strength for cumulative dominance
    signs_of_weights = np.sign(last_layer_weights)
    signed_weights = np.multiply(signs_of_weights, last_layer_weights)
    weight_ordering = np.argsort(-signed_weights[:, 0])
    feature_matrix = feature_matrix[weight_ordering, :]

    # Order by first row
    first_row_ordering = feature_matrix[0,:].argsort()
    first_row_ordering_inverted = first_row_ordering.argsort()
    feature_matrix_sorted_by_first_row = feature_matrix[:, first_row_ordering]
    # possibly_dominated_columns = range(0, n_actions-1)
    # possibly_cumulatively_dominated_columns = range(0, n_actions - 1)
    possibly_dominated_columns = SortedSet(range(0, n_actions - 1))
    possibly_cumulatively_dominated_columns = SortedSet(range(0, n_actions - 1))
    cum_sum_matrix = np.cumsum(feature_matrix_sorted_by_first_row, axis=0)
    dominated_columns = np.zeros(n_actions, dtype=bool)
    for col_ix in range(n_actions-1):  # col_ix = 0
        # print("col")
        col = feature_matrix_sorted_by_first_row[:, col_ix]
        compare_col_ix = col_ix + 1
        dominated = False
        while not dominated and compare_col_ix < n_actions:
            # print("compare_col")
            compare_col = feature_matrix_sorted_by_first_row[:, compare_col_ix]
            contradicted = False
            row_ix = 0
            while not contradicted and row_ix < n_hidden:
                # print("row", row_ix)
                if col[row_ix] > compare_col[row_ix]:
                    contradicted = True
                else:
                    row_ix += 1
            if not contradicted:  # Meaning it went through all rows and is dominated
                dominated = True
            compare_col_ix += 1
        if dominated:
            dominated_columns[col_ix] = True
    return dominated_columns



###
###  Trying to build more efficient version
###


# last_layer_weights = [[0.7], [-0.9], [0.8]]
# feature_matrix = np.array([[1,2,0,4,8], [3,4,2,1,5], [5,6,7,2,1]])
# def get_dom_and_cumdom(feature_matrix, last_layer_weights):
#     n_hidden, n_actions = feature_matrix.shape
#
#     # Order rows by descending weight strength for cumulative dominance
#     signs_of_weights = np.sign(last_layer_weights)
#     signed_weights = np.multiply(signs_of_weights, last_layer_weights)
#     weight_ordering = np.argsort(-signed_weights[:, 0])
#     feature_matrix = feature_matrix[weight_ordering, :]
#
#     # Order by first row
#     first_row_ordering = feature_matrix[0,:].argsort()
#     first_row_ordering_inverted = first_row_ordering.argsort()
#     feature_matrix_sorted_by_first_row = feature_matrix[:, first_row_ordering]
#     # possibly_dominated_columns = range(0, n_actions-1)
#     # possibly_cumulatively_dominated_columns = range(0, n_actions - 1)
#     possibly_dominated_columns = SortedSet(range(0, n_actions - 1))
#     possibly_cumulatively_dominated_columns = SortedSet(range(0, n_actions - 1))
#     cum_sum_matrix = np.cumsum(feature_matrix_sorted_by_first_row, axis=0)
#     for rowX in range(1, n_hidden):  # rowX = 2
#         # For dominance
#         rowX_ordering = feature_matrix_sorted_by_first_row[rowX, :].argsort()
#         ranks = np.empty_like(rowX_ordering)
#         ranks[rowX_ordering] = np.arange(len(rowX_ordering))
#         temp_max = n_actions-1
#         for el in possibly_dominated_columns:  # el = 0
#             rank_el = ranks[el]
#             if rank_el == temp_max:
#                 possibly_dominated_columns = possibly_dominated_columns - [el]
#                 temp_max = np.maximum(ranks[:el], temp_max-1)
#
#
#
#         # contradicting_with_first_row = rowX_ordering[np.less(rowX_ordering, range(0, n_actions))]
#         np.argmax(rowX_ordering)
#         contradicting_with_first_row = rowX_ordering[np.less(rowX_ordering, range(0, n_actions))]
#         possibly_dominated_columns = possibly_dominated_columns - contradicting_with_first_row
#
#         # For Cumulative dominance (can put both together later for more efficient code
#         rowX_ordering_cum = cum_sum_matrix[rowX, :].argsort()
#         contradicting_with_first_row_cum = rowX_ordering_cum[np.less(rowX_ordering_cum, range(0, n_actions))]
#         possibly_cumulatively_dominated_columns = possibly_cumulatively_dominated_columns - contradicting_with_first_row_cum
#     np.array(possibly_cumulatively_dominated_columns)
#     first_row_ordering_inverted[possibly_cumulatively_dominated_columns]

