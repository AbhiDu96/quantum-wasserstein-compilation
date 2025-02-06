import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library.n_local import PauliTwoDesign

import time

class Timer(object):
    """Context Timer object for simple runtime comparisons. """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        print('[%s]' % self.name, 'elapsed: %s' % np.round(time.time() - self.tstart, 2))


def get_test_states(num_qubits: int, x_basis=True, y_basis=True, z_basis=True, both_states=False):
    """Returns list of test basis states.

    Args:
        num_qubits (int): Number of qubits.
        x_basis (bool, optional): Use Pauli-X basis. Defaults to True.
        y_basis (bool, optional): Use Pauli-Y basis. Defaults to True.
        z_basis (bool, optional): Use Pauli-Z basis. Defaults to True.
        both_states (bool, optional): Use both eigenstates of each basis. Defaults to False.

    Returns:
        list: List of quantum circuits corresponding to basis states.
    """
    basis = []
    if x_basis:
        if both_states:
            basis.append(0)
            basis.append(1)
        else:
            basis.append(0)

    if y_basis:
        if both_states:
            basis.append(4)
            basis.append(5)
        else:
            basis.append(4)

    if z_basis:
        if both_states:
            basis.append(2)
            basis.append(3)
        else:
            basis.append(2)

    basis = np.array(basis)
    tiles = np.tile(basis, (num_qubits, 1))
    paulis = np.array(np.meshgrid(*tiles)).T.reshape(-1, num_qubits)

    # 0  = |0> -> id
    # 1 =  |1> -> X
    # 2 = |+> -> H
    # 3 = |-> -> HX
    # 4 = |i> -> SH
    # 5 = |-i> -> SHX
    converter = ["0", "1", "+", "-", "i", "-i"]

    states = []
    qc = QuantumCircuit(num_qubits)
    for state_config in paulis:
        newqc = qc.copy()
        for qubix, config in enumerate(state_config):
            if config == 0:
                newqc.id(qubix)
            elif config == 1:
                newqc.x(qubix)
            elif config == 2:
                newqc.h(qubix)
            elif config == 3:
                newqc.x(qubix)
                newqc.h(qubix)
            elif config == 4:
                newqc.h(qubix)
                newqc.s(qubix)
            elif config == 5:
                newqc.x(qubix)
                newqc.h(qubix)
                newqc.s(qubix)

        label = "|" + "".join([converter[x] for x in state_config]) + ">"
        states.append(qiskit.converters.circuit_to_gate(newqc, label=label))
    return states


def get_comp_basis_states(n):
    """provides a list of comp basis states"""
    list_circuits = []
    qc = QuantumCircuit(n)

    for state in get_test_states(n, True, False, False, True):
        list_circuits.append(qc.compose(state).decompose())
    return list_circuits


def get_random_states(num_qubits, quantity, reps=None, rng=np.random.default_rng(), epsilon=1e-4, seed=None):
    """Generates (approximatly Haar-)random states.
    Y. Nakata, C. Hirche, C. Morgan, and A. Winter, Unitary 2-Designs from Random X - and Z -Diagonal Unitaries, Journal of Mathematical Physics 58, 052203 (2017).


    Args:
        num_qubits (int): number of qubits
        num_states (int): number of states to create
        reps (int, optional): number of repetitions used. more repetition, better approximation. Defaults to None.
        epsilon (float, optional): indistinguishability. discrepancy to Haar random channel in diamond norm. Defaults to 1e-4.
        rng (optional): random number generator. Defaults to np.random.default_rng().
        seed (int, optional): seed for rng. Defaults to None.

    Returns:
        list: list of random seperable states
    """

    if seed is not None:
        rng = np.random.default_rng(seed=seed)

    if reps is None:
        reps = int(np.ceil(np.log((2 + 4 / (2**num_qubits - 1)) / epsilon) / np.log(2**num_qubits)))

    random_states = []
    for _ in range(quantity):
        pauli2design = PauliTwoDesign(num_qubits=num_qubits, reps=reps, seed=rng.integers(10e5))

        random_states.append(
            pauli2design.assign_parameters(rng.uniform(low=-np.pi, high=np.pi, size=(pauli2design.num_parameters))))
    return random_states


def get_random_product_states(num_qubits, num_states, rng=np.random.default_rng(), seed=None, **kwargs):
    """Generates random seperable states.

    Args:
        num_qubits (int): number of qubits
        num_states (int): number of states to create
        rng (optional): random number generator. Defaults to np.random.default_rng().
        seed (int, optional): seed for rng. Defaults to None.

    Returns:
        list: list of random seperable states
    """
    if seed is not None:
        rng = np.random.default_rng(seed=seed)

    alphas = ParameterVector("alphas", length=num_qubits * 3)
    ansatz = QuantumCircuit(num_qubits, name=f"RPS-n{num_qubits}")
    for quix in range(ansatz.num_qubits):
        ansatz.u(*[*alphas[3 * quix:3 * (quix + 1)], quix])

    states = []
    for _ in range(num_states):
        states.append(ansatz.assign_parameters(rng.uniform(low=-np.pi, high=np.pi, size=(ansatz.num_parameters))))
    return states


if __name__ == "__main__":

    qc = QuantumCircuit(3)

    print("\n\n Basis states ")
    for x in get_test_states(3, True, False, True, True):
        print(qc.compose(x, inplace=False))

    print("\n\n Product states ")
    for x in get_random_product_states(3, 3):
        print(x)

    print("\n\n Comp Basis States ")
    for x in get_comp_basis_states(3):
        qc = QuantumCircuit(3)
        print(qc.compose(x, inplace=False).draw())

    print("\n\n PauliTwoDesign ")
    for x in get_random_states(num_qubits=3, quantity=3):
        qc = QuantumCircuit(3)
        print(qc.compose(x, inplace=False).draw())

    import qccompiler
    with qccompiler.Timer("wo QC"):
        get_random_states(3, 1000)