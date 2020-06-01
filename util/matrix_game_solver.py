import time

import numpy as np
from cvxopt import matrix, solvers
from scipy.optimize import linprog


# warnings.filterwarnings('ignore', '.*Ill-conditioned*')


def linprog_solver_row(value_matrix, precision=4):
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

    options = {'cholesky': False,
               'sym_pos': False,
               'lstsq': True,
               'presolve': True},

    res = linprog(C, A_ub=A, b_ub=B, A_eq=A_eq, b_eq=B_eq, bounds=bounds,
                  options={'cholesky': False,
                           'sym_pos': False,
                           'lstsq': True,
                           'presolve': True})
    policy = res['x'][:-1]
    for i, p in enumerate(policy):
        if p < 10e-8:
            policy[i] = 0
    return policy, -res['fun']


def linprog_solver_col(value_matrix, precision=4):
    policy, value = linprog_solver_row(-value_matrix.T, precision)
    return policy, -value


def value_solve(value_matrix, precision=4):
    value_matrix = np.round(value_matrix, precision)
    _, value = cvxpot_solve_row(value_matrix)
    return value


def linprog_solve(value_matrix, precision=4):
    # rps = nash.Game(np.array(value_matrix))
    # eqs = rps.support_enumeration()
    value_matrix = np.round(value_matrix, precision)
    #py, value = cvxpot_solve_col(value_matrix)
    #px, v2 =  cvxpot_solve_row(value_matrix)
    py, value = linprog_solver_col(value_matrix, precision)
    px, _ = linprog_solver_row(value_matrix, precision)
    for i, x in enumerate(px):
        if np.fabs(x) < 10e-6:
            px[i] = 0
    for i, y in enumerate(py):
        if np.fabs(y) < 10e-6:
            py[i] = 0
    px = np.divide(px, np.sum(px))
    py = np.divide(py, np.sum(py))
    # if np.sum(px) < 0.99 or np.sum(py) < 0.99:
    #    print('divide error')
    #    m, n = value_matrix.shape
    #    return v2, np.divide(np.ones(m), m), np.divide(np.ones(n), n)

    # px = np.divide(px, np.sum(px))
    # px = np.nan_to_num(px)
    # py = np.divide(py, np.sum(py))
    # py = np.nan_to_num(py)
    # policy_x, policy_y = list(eqs)[0]
    # value = rps[policy_x,policy_y][0]
    return value, px, py


def cvxpot_solve_row(matrix):
    result = maxmin(matrix)['x']
    value = result[0]
    policy = np.array(result[1:]).reshape((-1,))
    return policy, value


def cvxpot_solve_col(matrix):
    return cvxpot_solve_row(-matrix.T)


def maxmin(A, solver='glpk'):
    solvers.options['show_progress'] = False
    solvers.options['refinement'] = 1

    num_vars = len(A)
    # minimize matrix c
    c = [-1] + [0 for i in range(num_vars)]
    c = np.array(c, dtype="float")
    c = matrix(c)
    # constraints G*x <= h
    G = np.matrix(A, dtype="float").T  # reformat each variable is in a row
    G *= -1  # minimization constraint
    G = np.vstack([G, np.eye(num_vars) * -1])  # > 0 constraint for all vars
    new_col = [1 for i in range(num_vars)] + [0 for i in range(num_vars)]
    G = np.insert(G, 0, new_col, axis=1)  # insert utility column
    G = matrix(G)
    h = ([0 for i in range(num_vars)] +
         [0 for i in range(num_vars)])
    h = np.array(h, dtype="float")
    h = matrix(h)
    # contraints Ax = b
    A = [0] + [1 for i in range(num_vars)]
    A = np.matrix(A, dtype="float")
    A = matrix(A)
    b = np.matrix(1, dtype="float")
    b = matrix(b)
    sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver, options={'glpk': {'msg_lev': 'GLP_MSG_OFF'}}
                     )
    return sol


def run():
    v = [
        [1., 2., 3.], [4., 5., 6.], [7., 8., 9.]
    ]
    v = np.array(v)
    # policy_x, value_x = __linprog_solver_row(v)
    # policy_y, value_y = __linprog_solver_col(v)
    start = time.time()
    # linv, linx, liny = linprog_solve(v)
    linx, linv = linprog_solver_row(v)
    liny, _ = linprog_solver_col(v)

    print(time.time() - start)
    start = time.time()

    px, value = cvxpot_solve_row(v)
    py, value = cvxpot_solve_col(v)
    px, value = cvxpot_solve_row(v*2)
    py, value = cvxpot_solve_col(v*2)
    px, value = cvxpot_solve_row(v*4)
    py, value = cvxpot_solve_col(v*4)
    px, value = cvxpot_solve_row(v*9)
    py, value = cvxpot_solve_col(v*9)
    px, value = cvxpot_solve_row(v*2.5)
    py, value = cvxpot_solve_col(v*2.5)
    print(time.time() - start)
    print('done')