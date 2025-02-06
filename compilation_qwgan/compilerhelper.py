import json
from typing import Optional, Union, List
import multiprocessing as mp

import numpy as np
import qiskit
import qiskit.quantum_info as quinfo
from qiskit.primitives import Estimator
from qiskit_torch_module.primitives_custom import EstimatorCustom
from qiskit_algorithms.gradients import ReverseEstimatorGradient
from qiskit_aer.primitives import Sampler
from qiskit_algorithms.gradients import ParamShiftSamplerGradient

estimator_custom = EstimatorCustom()
estimator = Estimator()
grad_estimator = ReverseEstimatorGradient(estimator)
sampler = Sampler(backend_options={"method": "statevector", "enable_truncation": False, "zero_threshold": 1e-16, "validation_threshold": 1e-16, }, run_options = {"shots": None})
grad_sampler = ParamShiftSamplerGradient(sampler)
NUM_PROCESS = mp.cpu_count()


class CalculateStats:
    
    def __init__(self, num_states: int, num_parameters: int):
        
        self.estimator_custom = EstimatorCustom()
        self.estimator = Estimator()
        self.grad_estimator = ReverseEstimatorGradient(self.estimator)
        self.sampler = Sampler(backend_options={"method": "statevector", "enable_truncation": False, "zero_threshold": 1e-16, "validation_threshold": 1e-16, }, run_options = {"shots": None})
        self.grad_sampler = ParamShiftSamplerGradient(self.sampler)
        self.s = num_states
        self.num_parameters = num_parameters

    def prepare_indices_for_multiprocessing(self, num_probe_states: int):
            
        indices = []
        start = 0
        size = num_probe_states // NUM_PROCESS
        remainder = num_probe_states % NUM_PROCESS

        for i in range(NUM_PROCESS):
            if i < remainder:
                end = start+size+1
            else:
                end = start+size
            indices += [list(range(start, end))]
            start = end

        indices += [list(range(end, num_probe_states))]
        indices = [x for x in indices if x != []]
        
        return indices
    
    def get_samplergrads(self, circuit: List[qiskit.QuantumCircuit], params: np.ndarray):
    
        return self.grad_sampler.run(circuit, [params]*len(circuit)).result().gradients
    
    def get_probs(self, circuits: List[qiskit.QuantumCircuit], params: np.ndarray):
    
        return self.sampler.run(circuits, [params]*len(circuits)).result().quasi_dists
    
    def get_exps(self, circuits: List[qiskit.QuantumCircuit], obs: quinfo.SparsePauliOp):
        
        return self.estimator_custom.run(circuits, observables = obs, parameter_values = [], shots = None).result().values.flatten()
    
    def stats_LET(self, circuits: List[qiskit.QuantumCircuit], params: np.ndarray):
        
        indices = self.prepare_indices_for_multiprocessing(self.s)
        
        mp_result_grads = mp.Array('f', (self.num_parameters * self.s))
        mp_result_probs = mp.Array('f', (1))
        
        circ_splited = [[circuits[i] for i in ind] for ind in indices]
        process = [ mp.Process(target=get_samplergrads_mp, args=(circ_splited[i], inds, [params]*len(inds), self.num_parameters, mp_result_grads, mp_result_probs)) for i, inds in enumerate(indices)]

        for p in process:
            p.start()

        for p in process:
            p.join()

        all_zero_grads_per_state = np.array(mp_result_grads, dtype=np.float32).reshape(self.s, len(params)).T
        C_LET = np.array(mp_result_probs)
        
        return all_zero_grads_per_state, C_LET
    
    def grads_wasserstein(self, circuits: List[qiskit.QuantumCircuit], obs: quinfo.SparsePauliOp, params: np.ndarray):
        
        indices = self.prepare_indices_for_multiprocessing(self.s)
        
        mp_result = mp.Array('f', (self.num_parameters * self.s))
        
        circ_splited = [[circuits[i] for i in ind] for ind in indices]
        obs_splited = [[obs[i] for i in ind] for ind in indices]
        process = [ mp.Process(target=get_estimatorgrads_mp, args=(circ_splited[i], obs_splited[i], inds, [params]*len(inds), len(params), mp_result)) for i, inds in enumerate(indices)]
        for p in process:
            p.start()
        for p in process:
            p.join()
        
        gradients_per_state = np.array(mp_result, dtype=np.float32).reshape(self.s, self.num_parameters)
        
        return gradients_per_state
    
    def expvals_wasserstein(self, circuits: List[qiskit.QuantumCircuit], obs: quinfo.SparsePauliOp, method: str = "custom"):
        
        indices = self.prepare_indices_for_multiprocessing(self.s)
        if method == "custom":
            mp_result = mp.Array('f', (len(obs) * self.s))
            circ_splited = [[circuits[i] for i in ind] for ind in indices]
            process = [ mp.Process(target=get_expval_mp, args=(circ_splited[i], obs, inds, mp_result, method)) for i, inds in enumerate(indices)]
            for p in process:
                p.start()
            for p in process:
                p.join()
        else:
            mp_result = mp.Array('f', (self.s))
            circ_splited = [[circuits[i] for i in ind] for ind in indices]
            obs_splited = [[obs[i] for i in ind] for ind in indices]
            process = [ mp.Process(target=get_expval_mp, args=(circ_splited[i], obs_splited[i], inds, mp_result, method)) for i, inds in enumerate(indices)]
            for p in process:
                p.start()
            for p in process:
                p.join()
    
        circuit_expect_vals = np.array(mp_result, dtype=np.float32)
        
        return circuit_expect_vals
    
    

def get_expval_mp(circuit: List[qiskit.QuantumCircuit], obs: quinfo.SparsePauliOp, ind: List[List[int]], mp_result, method = "custom"):
    
    """Calculates the expectation value of a state for a given operator using multiprocessing.
    Args:
        circuit (List[qiskit.QuantumCircuit]): List of states for measurement.
        obs (quinfo.SparsePauliOp): Operator to be measured.
        ind (List[List[int]]): List of list of indices for multiprocessing.
        mp_result (mp.array): Shared memory array for multiprocessing.
        method (str): Choose from "custom" and "default", to either use the custom estimator or the default one.

    Returns:
        None
    """
    
    if method == "custom":
        result = estimator_custom.run(circuit, observables = obs, parameter_values = [], shots = None).result().values
        result = result.flatten()
        mp_result[ind[0]*len(obs):ind[0]*len(obs)+len(result)] =result
    else:
        result = estimator.run(circuit, observables = obs).result().values
        mp_result[ind[0]:ind[0]+len(result)] =result
        
    return None
    

    
def get_estimatorgrads_mp(circuit: List[qiskit.QuantumCircuit], obs: quinfo.SparsePauliOp, ind: List[List[int]], params: List[int], n_thetas: int, mp_result):
    
    """Calculates the gradients of the expectation value of a state for a given operator using multiprocessing.
    Args:
        circuit (List[qiskit.QuantumCircuit]): List of states for measurement.
        obs (quinfo.SparsePauliOp): Operator to be measured.
        ind (List[List[int]]): List of list of indices for multiprocessing.
        params (List[int]): List of parameters for the gradients spread over the number of indices.
        n_thetas (int): Number of parameters for each state.
        mp_result (mp.array): Shared memory array for multiprocessing.
        method (str): Choose from "custom" and "default", to either use the custom estimator or the default one.

    Returns:
        None
    """
    
    result = grad_estimator.run(circuit, parameter_values = params, observables = obs, shots = None).result().gradients
    grads_per_state = np.zeros((len(ind), n_thetas))
    for i in range(len(ind)):
        for j in range(n_thetas):
            grads_per_state[i,j] = result[i][j]
    grads_per_state = grads_per_state.reshape(-1)
    mp_result[ind[0]*n_thetas:ind[0]*n_thetas+len(grads_per_state)] = grads_per_state
    
    return None
    

def get_samplergrads_mp(circuit: List[qiskit.QuantumCircuit], ind: List[List[int]], params: List[np.ndarray], n_thetas: int, mp_result, mp_result_probs):

    result = grad_sampler.run(circuit, params).result().gradients
    probs = sampler.run(circuit, params).result().quasi_dists
    mp_result_probs[0] = np.mean([1 - d.get(0, 0) for d in probs])
    all_zero_grads_per_state = np.zeros((len(ind), n_thetas))
    for i in range(len(ind)):
        for j in range(n_thetas):
            all_zero_grads_per_state[i, j] = result[i][j].get(0,0)
    mp_result[ind[0]*n_thetas:ind[0]*n_thetas+len(all_zero_grads_per_state.reshape(-1))] = all_zero_grads_per_state.reshape(-1)
    
    return None

# def get_expval(state: qiskit.QuantumCircuit,
#                operator: quinfo.SparsePauliOp,
#                method: str = "eval") -> float:
#     """Calculates the expectation value of a state for a given operator.
#     Args:
#         state (Union[qiskit.QuantumCircuit, op.StateFn]): State for measurement.
#         operator (op.OperatorStateFn): Operator to be measured.
#         sampler (Optional[op.CircuitSampler], optional): Necessary for every method but "eval". Defaults to None.
#         method (str, optional): Choose from "eval", "PauliExp", "AerExp", "MatrixExp". Defaults to "eval".

#     Returns:
#         float: expectation value.
#     """
#     if (method == "eval") or (method is None):
#         expectation_value = (~state @ operator @ state).eval().real
#     else:
#         if method == "PauliExp":
#             expectation = estimator.run(state, operator).result().values.real 
#         elif method == "AerExp":
#             expectation = estimator.run(state, operator).result().values.real 
#         elif method == "MatrixExp":
#             expectation = estimator.run(state, operator).result().values.real

#         expectation_value = expectation
#     return expectation_value


def calc_HS_inner_product(unitary1, unitary2, second_adjoint=False):
    """Calculates the Hilbert-Schmidt inner product between to unitaries. Accepts QuantumCircuit as
    input."""
    U = np.array(quinfo.Operator(unitary1))
    V = np.array(quinfo.Operator(unitary2))
    if second_adjoint:
        V = V.conj().T
    normHS = (U.shape[0]**(-2)) * np.abs(np.trace(V.conj().T @ U))**2
    return normHS


def calc_Fbar(unitary1, unitary2, second_adjoint=False):
    """Calculates the average fidelity between to unitaries. Accepts QuantumCircuit as input."""
    U = np.array(quinfo.Operator(unitary1))
    V = np.array(quinfo.Operator(unitary2))
    d = U.shape[0]
    if second_adjoint:
        V = V.conj().T
    normHS = (d**(-2)) * np.abs(np.trace(V.conj().T @ U))**2
    Fbar = (d * normHS + 1) / (d + 1)
    return Fbar


def calc_fidelities(input_states, target_qc, generated_qc):
    """Calculates fidelites of output states for two QuantumCircuits and a list of input states."""

    fidelities = []
    for state in input_states:
        gen_output = generated_qc.compose(state, front=True).decompose()
        target_output = target_qc.compose(state, front=True).decompose()
        fidelities.append(quinfo.state_fidelity(quinfo.Statevector(gen_output), quinfo.Statevector(target_output)))

    return fidelities


class NumpyEncoder(json.JSONEncoder):
    """serialize numpy ndarrays. for json export
    just call json.dumps(the_nd_array, cls=NumpyEncoder)
    from https://stackoverflow.com/a/47626762
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)