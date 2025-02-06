from typing import Optional
import cvxpy as cp
import numpy as np

# pauli convention:
# 0: identity       1: pauli x gate
# 2: pauli y gate   3: pauli z gate
pauli_converter = ["I", "X", "Y", "Z"]


def init_full_paulis(num_qubits: int, k_local: Optional[int] = None):
    """initalize all possible k_local Pauli observables.

    Args:
        num_qubits (int): number of qubits
        k_local (Optional[int], optional): k-locality of Pauli string aka maximum number of non-trivial operators
        per string. If None, then non-local (k=n). Defaults to None.

    Returns:
        numpy array corresponding to Pauli-Strings.
    """
    basis = np.array([0, 1, 2, 3])
    tiles = np.tile(basis, (num_qubits, 1))
    paulis = np.array(np.meshgrid(*tiles)).T.reshape(-1, num_qubits)

    # skip the all-zero op
    paulis = paulis[1:]

    if k_local is not None:
        paulis = paulis[np.count_nonzero(paulis, axis=1) <= k_local]

    return paulis


def convert_paulis_num_to_strings(paulis):
    # return pauli string characters
    y = [[pauli_converter[x] for x in row] for row in paulis]
    return [''.join(x) for x in y]


def solve_Hmax(paulis, expVals):
    """solves the linear program to find the EM Observable H_max acc. to
    B. T. Kiani, G. De Palma, M. Marvian, Z.-W. Liu, and S. Lloyd, Learning Quantum Data with the Quantum Earth Mover’s Distance, Quantum Sci. Technol. 7, 045002 (2022).
    """
    num_w = len(paulis)
    c = cp.Parameter(shape=(num_w,), name="differences expVals")
    weights = cp.Variable(num_w)

    mask_active_qubits = (paulis > 0)

    obj = cp.Maximize(cp.sum(cp.multiply(c, weights)))
    constraints = []
    for i_qubit in range(len(paulis[0])):

        constraints.append(cp.sum(cp.multiply(mask_active_qubits[:, i_qubit], cp.abs(weights))) <= .5)

    prob = cp.Problem(obj, constraints)

    c.value = expVals

    prob.solve()

    # to do: proper Error handling
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print("Could not solve for H_max")
        print(prob.status)
    return weights.value


if __name__ == "__main__":
    rng = np.random.default_rng()
    nqubits = 2
    klocal = 2

    paulis = init_full_paulis(nqubits, k_local=klocal)
    pauli_strings = convert_paulis_num_to_strings(paulis)
    pauli_strings.sort()
    print(pauli_strings)
    print("num of paulis:", len(pauli_strings))

    for n in range(1, 10):
        for k in range(1, n + 1):
            print(n, k, len(init_full_paulis(n, k_local=k)))  # 4^n scaling if k=n
