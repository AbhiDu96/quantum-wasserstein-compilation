"""
This file contains a circuit library. At its core is the method "circuit_parser" that allows easy
access to target circuits and ansätze.
"""
import sys
import logging
import pickle
from pathlib import Path

import numpy as np
import qiskit
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RXGate, RYGate, RZGate, PauliEvolutionGate, MCXGate
from qiskit.synthesis import LieTrotter
from utils import teststates

def circuit_parser(name: str,
                   nqubits,
                   depth=None,
                   is_target=True,
                   load_fixed_init=True,
                   rng=np.random.default_rng(),
                   **kwargs) -> QuantumCircuit:
    """Parses name and other parameters to return a circuit. This circuit can be parameterized.

    Args:
        name (str): Name of circuit.
        nqubits (int): Number of qubits
        depth (int, optional): Depth or repetation argument. Mandatory for some circuits. Defaults to None.
        is_target (bool, optional): If the circuit is parameterized, it can be initialized by setting this to true.
            Defaults to True. If true and parameterized, it either uses random or fixed values.
        load_fixed_init (bool, optional): If true, load values from file. Defaults to True.
        rng (numpy rng, optional): Random number generator. Defaults to np.random.default_rng().

    Raises:
        ValueError: If name is unknown.

    Returns:
        QuantumCircuit
    """

    name = name.lower()
    if name == "apa":
        circuit = alternating_pair_ansatz(num_qubits=nqubits, depth=depth, **kwargs)
    elif name in ("vatan-williams", "vw"):
        circuit = vatan_williams_ansatz(num_qubits=nqubits, depth=depth, **kwargs)
    elif name == "zycx":
        circuit = ZYCX_ansatz(num_qubits=nqubits, depth=depth, **kwargs)
    elif name == "hea":
        circuit = HEA(num_qubits=nqubits, repetitions=depth, **kwargs)
    elif name == "hea-sack":
        circuit = HEA_sack(num_qubits=nqubits,repetitions=depth, **kwargs)
    elif name == "khatri1":
        circuit = khatri_ex1(num_qubits=nqubits, repetitions = depth, **kwargs)
    elif name == "khatri2":
        circuit = khatri_ex2(num_qubits=nqubits, repetitions = depth, **kwargs)
    elif name == "qft":
        circuit = qft(nqubits)
    elif name == "toffoli":
        circuit = toffoli(nqubits)
    elif name in ("ghz", "ghz-state"):
        circuit = create_ghz_state(nqubits)
    elif name in ("wstate", "w-state"):
        circuit = create_w_state(nqubits)

    elif name in ("random", "haar", "hrs"):
        circuit = teststates.get_random_states(num_qubits=nqubits, quantity=1, rng=rng, **kwargs)[0]
        if kwargs.get("seed") is not None:
            circuit.name = f"HRS-n{nqubits}-seed{kwargs.get('seed')}"
        else:
            circuit.name = f"HRS-n{nqubits}"
    elif name in ("random-product", "randomproduct"):
        circuit = teststates.get_random_product_states(nqubits, num_states=1, rng=rng)[0]
        circuit.name = f"RPS-n{nqubits}"
    elif name == "rps-seed":
        circuit = teststates.get_random_product_states(nqubits, num_states=1, rng=rng, **kwargs)[0]
        circuit.name = f"RPS-n{nqubits}"
        if kwargs.get("seed") is not None:
            circuit.name = f"RPS-n{nqubits}-seed{kwargs.get('seed')}"
    elif name in ("trotter"):
        hams = {
            1: qi.SparsePauliOp(['X', 'Z'], coeffs = [3, 1]),
            2: qi.SparsePauliOp(['XZ', 'ZZ'], coeffs = [3,1]),
            2: qi.SparsePauliOp(['XZ', 'ZZ'], coeffs = [3,1]),
            3: qi.SparsePauliOp(['XZX', 'ZZX'], coeffs=[3,1]),
            4: qi.SparsePauliOp(['XZZX', 'ZZZZ'], coeffs = [3,1]),
            5: qi.SparsePauliOp(['XZXXZ', 'ZZXZZ'], coeffs = [3,1]),
            6: qi.SparsePauliOp(['XZXZZX', 'ZZZZZZ'], coeffs = [3,1])}
        circuit = QuantumCircuit(nqubits)
        circuit.append(trotterized_ham(hams[nqubits], num_time_slices=depth), [i for i in range(nqubits)])
        circuit = circuit.assign_parameters({circuit.parameters[0]: 1.0})
        circuit.name = f"trotter-n{nqubits}-d{depth}"
    else:
        raise ValueError("unkown circuit name %s" % name)
    
    if is_target and (circuit.num_parameters > 0):
        circuit = init_circuit(circuit,
                               load_fixed_initializtaion_from_file=load_fixed_init,
                               is_ansatz=not is_target,
                               rng=rng)

    return circuit


def trotterized_ham(hamiltonian, num_time_slices):
    """return a trotterized hamiltonian."""
    if num_time_slices is None:
        logging.warning("num_time_slices cannot be None.")
    elif isinstance(num_time_slices, (str, float)):
        num_time_slices = int(num_time_slices)

    evo_time = qiskit.circuit.Parameter('t')
    evol_gate = PauliEvolutionGate(hamiltonian, time = evo_time, synthesis = LieTrotter(reps = num_time_slices))
    return evol_gate


def khatri_ex1(num_qubits, repetitions=1):
    """following Khatri et al 2019, example 1. just rotational gates without entangling.
     khatri 1
     ┌──────────┐
q_0: ┤ Rz(θ[0]) ├
     ├──────────┤
q_1: ┤ Rz(θ[1]) ├
     ├──────────┤
q_2: ┤ Rz(θ[2]) ├
     └──────────┘

    """
    qc = QuantumCircuit(num_qubits, name=f"khatri-n{num_qubits}")
    theta = ParameterVector("θ", num_qubits*repetitions)
    for rep in range(repetitions):
        for idq in range(num_qubits):
            qc.rz(theta[rep*num_qubits + idq], qc.qubits[idq])

    return qc


def khatri_ex2(num_qubits, repetitions=1, entanglement="circular"):
    """following Khatri et al 2019, example 2. Z rotations plus entangling block.
     ┌──────────┐ ░       ░       ░ ┌──────────┐
q_0: ┤ U1(θ[0]) ├─░───■───░───────░─┤ U1(θ[3]) ├
     ├──────────┤ ░ ┌─┴─┐ ░       ░ ├──────────┤
q_1: ┤ U1(θ[1]) ├─░─┤ X ├─░───■───░─┤ U1(θ[4]) ├
     ├──────────┤ ░ └───┘ ░ ┌─┴─┐ ░ ├──────────┤
q_2: ┤ U1(θ[2]) ├─░───────░─┤ X ├─░─┤ U1(θ[5]) ├
     └──────────┘ ░       ░ └───┘ ░ └──────────┘

    """
    qc = QuantumCircuit(num_qubits, name=f"khatri2-n{num_qubits}-{entanglement}")
    theta = ParameterVector("θ", 2 * num_qubits*repetitions)
    for rep in range(repetitions):
        for idq in range(num_qubits):
            qc.rz(theta[rep*num_qubits + idq], qc.qubits[idq])
        
        qc.barrier()
        
        qc.compose(qiskit.circuit.library.TwoLocal(num_qubits, entanglement_blocks='cx', insert_barriers=False, reps=1, entanglement=entanglement), inplace=True)
            
        qc.barrier()
        
        for idq in range(num_qubits):
            qc.rz(theta[rep*num_qubits + num_qubits + idq], qc.qubits[idq])

    return qc


def qft(num_qubits):
    """returns qiskits qft implementation"""
    qc = QuantumCircuit(num_qubits, name=f"qft-n{num_qubits}")
    qc.compose(qiskit.circuit.library.QFT(num_qubits, do_swaps=False), inplace=True)
    return qc


def toffoli(num_qubits):
    """returns qiskits toffoli implementation"""
    toff = QuantumCircuit(num_qubits, name="toffoli")
    toff.append(MCXGate(num_qubits-1), [i for i in range(num_qubits-1)] + [num_qubits-1])
    return toff


###############################################################################
# the following code is from https://github.com/vutuanhai237/UC-VQA/blob/main/codes/qtm/state.py
# and https://github.com/vutuanhai237/UC-VQA/blob/main/codes/qtm/gate.py
# publication: http://arxiv.org/abs/2204.11635
# V. T. Hai and L. B. Ho, Universal Compilation for Quantum State Preparation and Tomography, arXiv:2204.11635.


def cf(qc: qiskit.QuantumCircuit, theta: float, qubit1: int, qubit2: int):
    """Add Controlled-F gate to quantum circuit
    Args:
        - qc (qiskit.QuantumCircuit): ddded circuit
        - theta (float): arccos(1/sqrt(num_qubits), base on number of qubit
        - qubit1 (int): control qubit
        - qubit2 (int): target qubit
    Returns:
        - qiskit.QuantumCircuit: Added circuit
    """
    cf = qiskit.QuantumCircuit(2)
    u = np.array([[1, 0, 0, 0], [0, np.cos(theta), 0, np.sin(theta)], [0, 0, 1, 0],
                  [0, np.sin(theta), 0, -np.cos(theta)]])
    cf.unitary(u, [0, 1])
    cf_gate = cf.to_gate(label='CF')
    qc.append(cf_gate, [qubit1, qubit2])
    return qc


def w3(circuit: qiskit.QuantumCircuit, qubit: int):
    """Create W state for 3 qubits
    Args:
        - circuit (qiskit.QuantumCircuit): added circuit
        - qubit (int): the index that w3 circuit acts on
    Returns:
        - qiskit.QuantumCircuit: added circuit
    """
    qc = qiskit.QuantumCircuit(3)
    theta = np.arccos(1 / np.sqrt(3))
    qc.cf(theta, 0, 1)
    qc.cx(1, 0)
    qc.ch(1, 2)
    qc.cx(2, 1)
    w3 = qc.to_gate(label='w3')
    # Add the gate to your circuit which is passed as the first argument to cf function:
    circuit.append(w3, [qubit, qubit + 1, qubit + 2])
    return circuit


qiskit.QuantumCircuit.w3 = w3
qiskit.QuantumCircuit.cf = cf


def w(qc: qiskit.QuantumCircuit, num_qubits: int, shift: int = 0):
    """The below codes is implemented from [this paper](https://arxiv.org/abs/1606.09290)
    Args:
        - num_qubits (int): number of qubits
        - shift (int, optional): begin wire. Defaults to 0.
    Raises:
        - ValueError: When the number of qubits is not valid
    Returns:
        - qiskit.QuantumCircuit
    """
    if num_qubits < 2:
        raise ValueError('W state must has at least 2-qubit')
    if num_qubits == 2:
        # |W> state ~ |+> state
        qc.h(0)
        return qc
    if num_qubits == 3:
        # Return the base function
        qc.w3(shift)
        return qc
    else:
        # Theta value of F gate base on the circuit that it acts on
        theta = np.arccos(1 / np.sqrt(qc.num_qubits - shift))
        qc.cf(theta, shift, shift + 1)
        # Recursion until the number of qubits equal 3
        w(qc, num_qubits - 1, qc.num_qubits - (num_qubits - 1))
        for i in range(1, num_qubits):
            qc.cx(i + shift, shift)
    return qc


def create_w_state(num_qubits):
    """Create n-qubit W state based on the its number of qubits
    Args:
        - qc (qiskit.QuantumCircuit): init circuit
    Returns:
        - qiskit.QuantumCircuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, name=f"wstate-n{num_qubits}")
    qc.x(0)
    qc = w(qc, qc.num_qubits)
    return qc


def create_ghz_state(num_qubits, theta: float = np.pi / 2):
    """Create GHZ state with a parameter
    Args:
        - num_qubits (int): number of qubits
        - theta (float): parameters
    Returns:
        - QuantumCircuit: the added circuit
    """
    if not isinstance(theta, float):
        theta = (theta['theta'])
    qc = qiskit.QuantumCircuit(num_qubits, name=f"ghz-n{num_qubits}")
    qc.ry(theta, 0)
    for i in range(0, qc.num_qubits - 1):
        qc.cx(0, i + 1)
    return qc


def HEA(num_qubits, repetitions, entanglement="circular", insert_barriers=False):
    qc = QuantumCircuit(num_qubits, name=f"HEA-n{num_qubits}-rep{repetitions}-{entanglement}")
    params = ParameterVector("phi", length=4*num_qubits * repetitions)
    
    for i in range(repetitions):
        for iz in range(num_qubits):
            qc.ry(params[4*num_qubits * i + iz], iz)
        for iz in range(num_qubits):
            qc.rz(params[4*num_qubits * i + num_qubits + iz], iz)
            
        qc.barrier()
        
        qc.compose(qiskit.circuit.library.TwoLocal(num_qubits, entanglement_blocks='cx', insert_barriers=False, reps=1, entanglement=entanglement), inplace=True)
        
        qc.barrier()
        
        for iz in range(num_qubits):
            qc.ry(params[4*num_qubits * i + 2*num_qubits + iz], iz)
        for iz in range(num_qubits):
            qc.rz(params[4*num_qubits * i + 3*num_qubits + iz], iz)
    
    return qc


def HEA_sack(num_qubits,
             repetitions,
             entanglement_blocks="cz",
             entanglement="circular",
             insert_barriers=False,
             include_initial_ry=True):
    """ based on S. H. Sack, R. A. Medina, A. A. Michailidis, R. Kueng, and M. Serbyn, Avoiding Barren Plateaus Using Classical Shadows, PRX Quantum 3, 020365 (2022).
    
     ┌────────────┐┌────────────┐      ┌────────────┐                    ┌────────────┐                       
q_0: ┤ R(π/4,π/2) ├┤ Rz(phi[0]) ├─■──■─┤ Rz(phi[3]) ├───────────────■──■─┤ Rz(phi[6]) ├───────────────■──■────
     ├────────────┤├────────────┤ │  │ └────────────┘┌────────────┐ │  │ └────────────┘┌────────────┐ │  │    
q_1: ┤ R(π/4,π/2) ├┤ Rx(phi[1]) ├─┼──■───────■───────┤ Ry(phi[4]) ├─┼──■───────■───────┤ Rz(phi[7]) ├─┼──■──■─
     ├────────────┤├────────────┤ │          │       ├────────────┤ │          │       ├────────────┤ │     │ 
q_2: ┤ R(π/4,π/2) ├┤ Rz(phi[2]) ├─■──────────■───────┤ Ry(phi[5]) ├─■──────────■───────┤ Rx(phi[8]) ├─■─────■─
     └────────────┘└────────────┘                    └────────────┘                    └────────────┘         

    
    """

    rot_gates = [RZGate, RYGate, RXGate]

    qc = QuantumCircuit(
        num_qubits,
        name=
        f"HEA[sack]-n{num_qubits}-rep{repetitions}-entblock:{entanglement_blocks}-ent:{entanglement}-initry:{include_initial_ry}"
    )

    phi = qiskit.circuit.ParameterVector("phi", length=num_qubits * repetitions)
    phi_idx = 0
    if include_initial_ry:
        for iqu in qc.qubits:
            qc.ry(np.pi / 4, iqu)

    for rep in range(repetitions):
        rot_block = QuantumCircuit(num_qubits)
        for iqu in rot_block.qubits:
            rot_block.append(np.random.choice(rot_gates)(phi[phi_idx]), [iqu])
            phi_idx += 1
        with_ent = qiskit.circuit.library.TwoLocal(num_qubits,
                                                   rotation_blocks=rot_block,
                                                   reps=1,
                                                   skip_final_rotation_layer=True,
                                                   entanglement_blocks=entanglement_blocks,
                                                   entanglement=entanglement,
                                                   insert_barriers=insert_barriers)
        with_ent = with_ent.assign_parameters(phi[(rep * num_qubits):((rep + 1) * num_qubits)])
        qc.compose(with_ent, qc.qubits, inplace=True)
    return qc


def rot_block(num_qubits):
    rota_param = ParameterVector("rota", num_qubits * 3)
    qc = QuantumCircuit(num_qubits, name="rot-block")
    for qix, qubt in enumerate(qc.qubits):
        qc.u(*[*rota_param[qix * 3:(qix + 1) * 3] + [qubt]])
    return qc.to_gate()


def rz_ry_block(num_qubits):
    rota_param = ParameterVector("rota", num_qubits * 2)
    qc = QuantumCircuit(num_qubits, name="rz-ry-block")
    for qix, qubt in enumerate(qc.qubits):
        qc.rz(rota_param[qix], qubt)
    for qix, qubt in enumerate(qc.qubits):
        qc.ry(rota_param[qix + num_qubits], qubt)
    return qc.to_gate()


def cnot_block():
    """ dressed cnot """
    qc = QuantumCircuit(2, name="dressed-cnot")
    betas = ParameterVector("betas", 6)
    qc.cx(0, 1)
    qc.u(*[*betas[0:3], 0])
    qc.u(*[*betas[3:6], 1])
    return qc.to_gate()


def cnot_zy_block():
    """ weaker dressed cnot"""
    qc = QuantumCircuit(2, name="cnot-zy")
    betas = ParameterVector("betas", 4)
    qc.cx(0, 1)
    qc.rz(betas[0], 0)
    qc.rz(betas[1], 1)
    qc.ry(betas[2], 0)
    qc.ry(betas[3], 1)
    return qc.to_gate()


def vatan_williams():
    """ 2 qubit vatan williams general unitary"""
    vw = QuantumCircuit(2, name="VW2")
    alpha = ParameterVector("α", 15)
    vw.rz(alpha[0], 0)
    vw.ry(alpha[1], 0)
    vw.rz(alpha[2], 0)
    vw.rz(alpha[3], 1)
    vw.ry(alpha[4], 1)
    vw.rz(alpha[5], 1)
    vw.cx(1, 0)
    vw.rz(alpha[6], 0)
    vw.ry(alpha[7], 1)
    vw.cx(0, 1)
    vw.ry(alpha[8], 1)
    vw.cx(1, 0)
    vw.rz(alpha[9], 0)
    vw.ry(alpha[10], 0)
    vw.rz(alpha[11], 0)
    vw.rz(alpha[12], 1)
    vw.ry(alpha[13], 1)
    vw.rz(alpha[14], 1)
    return vw


def vw_block():
    """ F. Vatan and C. Williams, Optimal Quantum Circuits for General Two-Qubit Gates, Phys. Rev. A 69, 032315 (2004).
    
     ┌───┐┌──────────────┐                     ┌───┐┌──────────────┐┌──────────────┐┌──────────────┐
q_0: ┤ X ├┤ Rz(alpha[0]) ├──■──────────────────┤ X ├┤ Rz(alpha[3]) ├┤ Ry(alpha[4]) ├┤ Rz(alpha[5]) ├
     └─┬─┘├──────────────┤┌─┴─┐┌──────────────┐└─┬─┘├──────────────┤├──────────────┤├──────────────┤
q_1: ──■──┤ Ry(alpha[1]) ├┤ X ├┤ Ry(alpha[2]) ├──■──┤ Rz(alpha[6]) ├┤ Ry(alpha[7]) ├┤ Rz(alpha[8]) ├
          └──────────────┘└───┘└──────────────┘     └──────────────┘└──────────────┘└──────────────┘
    
    """
    qc = QuantumCircuit(2, name="vw-block")
    alpha = ParameterVector("alpha", 9)
    qc.cx(1, 0)
    qc.rz(alpha[0], 0)
    qc.ry(alpha[1], 1)
    qc.cx(0, 1)
    qc.ry(alpha[2], 1)
    qc.cx(1, 0)
    qc.rz(alpha[3], 0)
    qc.ry(alpha[4], 0)
    qc.rz(alpha[5], 0)
    qc.rz(alpha[6], 1)
    qc.ry(alpha[7], 1)
    qc.rz(alpha[8], 1)
    return qc.to_gate()


def alternating_pair_ansatz(num_qubits, depth, entanglement=None, **kwargs):
    
    """based on qiskits two local ansatz
    

    
    Args:
            num_qubits (int): Number of qubits in the ansatz.
            depth (int): Depth of the ansatz.
            entanglement (str, optional): Possible entanglements are "full", "linear", "reverse_linear", "pairwise", "circular" and "sca". Defaults to None.

    """
    if kwargs is not None:
        logging.info("APA. unused arguments: %s", kwargs)

    if not isinstance(depth, int):
        raise ValueError("Alternating Pair Ansatz requires depth statement.")

    if entanglement is None:
        entanglement = list(zip(np.arange(0, num_qubits - 1, 2),
                                np.arange(0, num_qubits - 1, 2) + 1)) + list(
                                    zip(np.arange(1, num_qubits - 1, 2),
                                        np.arange(1, num_qubits - 1, 2) + 1))
        if (num_qubits > 2) and ((num_qubits % 2) == 0):
            entanglement.append((num_qubits - 1, 0))

    qc = QuantumCircuit(num_qubits, name=f"APA-n{num_qubits}-d{depth}-{entanglement}")

    qc.compose(rot_block(num_qubits), inplace=True)
    qc.compose(qiskit.circuit.library.TwoLocal(num_qubits=num_qubits,
                                               entanglement_blocks=[cnot_block()],
                                               insert_barriers=True,
                                               reps=depth,
                                               entanglement=entanglement).decompose(),
               inplace=True)
    return qc


def ZYCX_ansatz(num_qubits, depth, entanglement="reverse_linear", **kwargs):
    
    """empirical motivated ansatz. potent for 3 qubits

    Args:
            num_qubits (int): Number of qubits in the ansatz.
            depth (int): Depth of the ansatz.
            entanglement (str, optional): Possible entanglements are "full", "linear", "reverse_linear", "pairwise", "circular" and "sca". Defaults to None.

    """

    if not isinstance(depth, int):
        raise ValueError("ZYCX Ansatz requires depth statement.")

    qc = QuantumCircuit(num_qubits, name=f"ZYCX-n{num_qubits}-d{depth}-{entanglement}")

    qc.compose(rz_ry_block(num_qubits), inplace=True)
    qc.compose(qiskit.circuit.library.TwoLocal(num_qubits=num_qubits,
                                               entanglement_blocks=[cnot_zy_block()],
                                               insert_barriers=True,
                                               reps=depth,
                                               entanglement=entanglement).decompose(),
               inplace=True)

    # replace all parameters by θ
    qc.assign_parameters(ParameterVector("θ", qc.num_parameters), inplace=True)

    return qc


def vatan_williams_ansatz(num_qubits, depth, entanglement=None, **kwargs):
    
    """ inspired from F. Vatan and C. Williams, Optimal Quantum Circuits for General Two-Qubit Gates, Phys. Rev. A 69, 032315 (2004).
    this ansatz extends the 2 qubit VW block to more qubits.

    Args:
            num_qubits (int): Number of qubits in the ansatz.
            depth (int): Depth of the ansatz.
            entanglement (str, optional): Possible entanglements are "full", "linear", "reverse_linear", "pairwise", "circular" and "sca". Defaults to None.

    """

    if entanglement is None:
        entanglement = list(zip(np.arange(0, num_qubits - 1, 2),
                                np.arange(0, num_qubits - 1, 2) + 1)) + list(
                                    zip(np.arange(1, num_qubits - 1, 2),
                                        np.arange(1, num_qubits - 1, 2) + 1))
        if (num_qubits > 2) and ((num_qubits % 2) == 0):
            entanglement.append((num_qubits - 1, 0))

    qc = QuantumCircuit(num_qubits, name=f"VW-n{num_qubits}-d{depth}-{entanglement}")

    qc.compose(rot_block(num_qubits), inplace=True)
    qc.compose(qiskit.circuit.library.TwoLocal(num_qubits=num_qubits,
                                               entanglement_blocks=[vw_block()],
                                               insert_barriers=True,
                                               reps=depth,
                                               entanglement=entanglement).decompose(),
               inplace=True)
    return qc


initization_parameters_path = Path(Path(__file__).resolve().parent.parent / "Data" / "init_params")


def save_initializations():
    """save 2*10000 angles locally to load reproducible initializations from. 1 for ansatzes, 1 for targets."""
    if not initization_parameters_path.is_file():
        rng = np.random.default_rng()
        numbers = rng.uniform(low=-np.pi, high=np.pi, size=(10000, 2))
        with initization_parameters_path.open("wb") as the_file:
            pickle.dump(numbers, the_file)
        print(f"Saved new fixed parameters at {initization_parameters_path.resolve()}")


save_initializations()


def load_initializations(for_ansatz=False, verbose=False):
    """load values from file. verbose to check values"""
    if initization_parameters_path.is_file():
        with initization_parameters_path.open("rb") as the_file:
            param_values_from_file = pickle.load(the_file)
            if verbose:
                print("loaded values: ", param_values_from_file[0:10, 0])
        # to ensure that ansatz != target two sets of parameters are initialized
        if for_ansatz:
            return param_values_from_file[:, 1]
        else:
            return param_values_from_file[:, 0]
    else:
        print("No init file.")
        return None


# pre-load when module is loaded so that the values are stored in memory
init_params_static_ansatz = load_initializations(for_ansatz=True)
init_params_static_target = load_initializations(for_ansatz=False)

def init_circuit(qc: QuantumCircuit,
                 load_fixed_initializtaion_from_file=True,
                 is_ansatz=False,
                 rng=np.random.default_rng()):
    """initalizes parameterized circuit. if is_ansatz, a different set of values are used.

    Args:
        qc (QuantumCircuit): parameterized QC to initialize.
        load_fixed_initializtaion_from_file (bool, optional): if true load from file. random otherwise. Defaults to True.
        is_ansatz (bool, optional): fixed values depend on this. Defaults to False.
        rng (_type_, optional): rng for random initialization. Defaults to np.random.default_rng().

    Returns:
        QuantumCircuit: bound circuit (=all parameters are assigned to a value)
    """
    if qc.num_parameters < 1:
        logging.warning("Cannot initialize Quantum Circuit without parameters. name=%s", qc.name)
        return qc

    if load_fixed_initializtaion_from_file:
        if is_ansatz:
            parameter_values = init_params_static_ansatz[0:qc.num_parameters]
        else:
            parameter_values = init_params_static_target[0:qc.num_parameters]

    else:
        parameter_values = rng.uniform(low=-np.pi, high=np.pi, size=(qc.num_parameters))
    return qc.assign_parameters(parameter_values)


if __name__ == "__main__":

    print("\n\n khatri 1")
    print(circuit_parser("khatri1", 3, depth=3, is_target=False, load_fixed_init=False).draw())

    print("\n\n khatri 2")
    print(circuit_parser("khatri2", nqubits=3, is_target=False).decompose().draw())
    
    print("\n\nAPA")
    print(circuit_parser("apa", nqubits=3, depth=3, is_target=False).decompose().draw())
    
    print("\n\nZYCX")
    print(circuit_parser("zycx", nqubits=3, depth=3, is_target=False).decompose().draw())

    print("\n\nVatan-Williams")
    print(circuit_parser("vw", nqubits=3, depth=3, is_target=False).decompose().draw())

    print("\n\nHEA-SACK")
    print(circuit_parser("hea-sack", nqubits=3, depth=3, is_target=False).decompose().draw())
    
    print("\n\n wstate 3 qbt")
    print(w3(QuantumCircuit(3), 0).draw())

    print("\n\n vw block")
    print(QuantumCircuit(2).compose(vw_block()).decompose().draw())

    wstate = create_w_state(4)
    wstate.measure_all()

    result = qiskit.execute(wstate, backend=qiskit.Aer.get_backend("qasm_simulator"), shots=10e5).result()
    print(result.get_counts(wstate))

    ghz = create_ghz_state(3)
    print(ghz.draw())
    ghz.measure_all()

    result = qiskit.execute(ghz, backend=qiskit.Aer.aer.get_backend("qasm_simulator"), shots=10e5).result()
    print(result.get_counts(ghz))

    print("\n\nTrotter-Decomp")
    print(circuit_parser("trotter", nqubits=3, depth=3).decompose().decompose().draw())
