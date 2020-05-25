import time

import numpy as np
from scipy.optimize import linprog


# warnings.filterwarnings('ignore', '.*Ill-conditioned*')


def __linprog_solver_col(value_matrix, precision=4):
    value_matrix = np.nan_to_num(np.round(value_matrix, precision))
    m, n = value_matrix.shape

    # solve col
    # objectif vector c is 0*x1+0*x2+...+0*xn+v
    C = []
    for i in range(n):
        C.append(0)
    C.append(-1)
    A = []
    for i_col in range(n):
        col = value_matrix[:, i_col]
        constraint_row = []
        for item in col:
            constraint_row.append(-item)
        constraint_row.append(1)
        A.append(constraint_row)
    B = []
    for i in range(m):
        B.append(0)

    A_eq = []
    A_eq_row = []
    for i in range(n):
        A_eq_row.append(1)
    A_eq_row.append(0)
    A_eq.append(A_eq_row)
    B_eq = [1]

    bounds = []
    for i in range(n):
        bounds.append((0, 1))
    bounds.append((None, None))

    res = linprog(C, A_ub=A, b_ub=B, A_eq=A_eq, b_eq=B_eq, bounds=bounds,
                  options={'cholesky': False,
                           'sym_pos': False,
                           'lstsq': True,
                           'presolve': True})
    return res['x'][:-1], -res['fun']


def __linprog_solver_row(value_matrix, precision=4):
    policy, value = __linprog_solver_col(-value_matrix.T, precision)
    return policy, -value


def value_solve(value_matrix, precision=4):
    _, value = __linprog_solver_col(value_matrix, precision)
    return value


def linprog_solve(value_matrix, precision=4):
    # rps = nash.Game(np.array(value_matrix))
    # eqs = rps.support_enumeration()
    px, value = __linprog_solver_row(value_matrix,precision)
    py, v2 = __linprog_solver_col(value_matrix,precision)
    # policy_x, policy_y = list(eqs)[0]
    # value = rps[policy_x,policy_y][0]
    return value, py, px


def run():
    v = [
        [0.45294148, - 1.07341754, - 0.08597379, 0.4511938],
        [- 0.86239982, - 0.97331583, - 0.60332541, - 0.91927982],
        [0.43661323, 0.8537626, 0.61845766, 0.34688129],
        [1.99999816, 1.99999816, 1.99999807, 1.99999807]
    ]
    v = np.array(v)
    # policy_x, value_x = __linprog_solver_row(v)
    # policy_y, value_y = __linprog_solver_col(v)
    start = time.time()
    linv, linx, liny = linprog_solve(v)
    print(time.time() - start)
    start = time.time()
    print(time.time() - start)
    print(linv)
    print('done')
