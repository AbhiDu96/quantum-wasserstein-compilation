import datetime
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")
import json
import logging
import pickle
import tarfile
import time
from cgi import test
from itertools import compress
from pathlib import Path
from typing import List, Literal, Union

import numpy as np
import qiskit
import qiskit.quantum_info as quinfo
import shortuuid

from utils.optimizers import optimizer
from utils.paulis import (convert_paulis_num_to_strings, init_full_paulis, solve_Hmax)
from utils.circuits import (circuit_parser, load_initializations)
from utils.teststates import (get_comp_basis_states, get_random_product_states, get_random_states)
from compilation_qwgan.compilerhelper import (NumpyEncoder, calc_Fbar, calc_fidelities, calc_HS_inner_product, CalculateStats)
from qiskit import QuantumCircuit
from qiskit import qpy

logging.basicConfig(level=logging.INFO)
logging.getLogger("qiskit").setLevel(logging.WARNING)

def last_var(arr, N):
    """Computes variance of last N values in arr

    Args:
        arr (array): 1D numpy array
        N (int): number of values to take into account

    Returns:
        float: variance if there are more than N values in arr
    """
    
    return np.where(len(arr) > N, np.var(arr[-N:-1]), np.nan).item()


class Generator(QuantumCircuit):

    def __init__(self, ansatz, rng=np.random.default_rng()):
        """A trainablte quantum circuit.

        Args:
            ansatz (QuantumCircuit): parameterized QuantumCircuit
            rng (numpy.random._generator.Generator, optional): Random Number Generator. Defaults to np.random.default_rng().

        Raises:
            ValueError: If ansatz is not parameterized.
        """
        if ansatz.num_parameters == 0:
            raise ValueError("Ansatz must be parameterized.")

        super().__init__(ansatz.num_qubits, name="V")
        self.ansatz = ansatz
        self.compose(self.ansatz, inplace=True)
        self.rng = rng
        self.theta = np.zeros(shape=(self.num_parameters))
        self.c_em = -1
        self.training = {}

        self.epsilon_two_design = (
            1e-4  # epsilon as in Nakata et al. 2017 to approximate 2-Designs
        )
        self.max_compile_time = datetime.timedelta(hours=23, minutes=30)

        self.optim = None
        self.state_pool = None
        self.FISC = None
        self.comp_basis_states = get_comp_basis_states(self.num_qubits)

        self.resuming = False
        self.compiler_hyper = None

    def compile(
        self,
        target_qc: QuantumCircuit,
        method: Literal["wasserstein", "HST", "LET"] = "wasserstein",
        compiler_hyper=None,
        test_states: Union[List[QuantumCircuit], Literal["haar", "product", "basis", "allzero"]] = "haar",
        initialization_argument: Union[List[float], Literal["random", "fixed"]] = "random",
        t: int = 0,
        T: int = 100,
        optimizer_hyper={
            "method": "GRAD-DESC",
            "eta": 0.3},
        run_id: str = None,
        verbose=1,
    ):
        """Compile the ansatz to the target circuit.

        Args:
            target (QuantumCircuit): Target circuit to compile against.
            method (Literal[wasserstein, HST, LET], optional): Compilation Method. Defaults to "wasserstein".
            compiler_hyper (dict, optional): Dictionary of hyper parameters for the compiler. Method dependent. Defaults to None.
            test_states (Union[List[QuantumCircuit], Literal[haar, product, basis, allzero]]): Set of test states to use.
                Could be also a list of states to compile on [SISC or FISC]. Defaults to "haar".
            initialization_argument (Union[List[float], Literal[random, fixed]], optional): List of initial theta
                parameters or "random" for iid initialization using rng or "fixed" to load values from fixed random list.
                Defaults to "random".
            t (int, optional): Current iteration if resuming. Defaults to 0.
            T (int, optional): Number of iterations. Defaults to 100.
            optimizer_hyper (dict): Dict of hyper parameters for optimizer. Defaults to gradient descent with lr=0.3.
            run_id (str, optional): str identifier for this compilation run. Defaults to None.
            verbose (int, optional): Log to console at each $verbose iteration. Defaults to 1.

        Raises:
            ValueError: Target is parameterized.

        Returns:
            Generator: compiled circuit.
        """

        if target_qc.num_parameters != 0:
            raise ValueError("Target circuit must be bound (no free parameters).")

        # fresh start, not continued
        if t == 0:
            self.target = target_qc
            self.compiler_hyper = compiler_hyper
            self.training = {}
            self.training.update({
                "run_id": shortuuid.uuid() if run_id is None else run_id,
                "target": target_qc,
                "target-name": target_qc.name,
                "ansatz-name": self.ansatz.name,
                "target-depth": target_qc.depth(),
                "ansatz-depth": self.ansatz.depth(),
                "method": method,
                "n": self.num_qubits,})
            self.training["t"] = t
            self.training["T"] = T

            # get optimizer
            optimizer_hyper.update(dict(num_parameters=self.num_parameters))
            self.optim = optimizer(optimizer_hyper)  # get optimizer from arguments
            self.training.update(self.optim.hyper)  # write the (real) optimizer values into training

            # get initialization of parameters
            self.theta, initialization_method = self._initialize_from_argument(initialization_argument)
            self.training["initialization"] = initialization_method

            self._initialize_test_states(test_states)
            self.stats_calculator = CalculateStats(num_states = self.compiler_hyper["s"], num_parameters = self.num_parameters)

            if method == "HST" and test_states is not None:
                logging.warning("HST ignores input test states (only used for fidelity-tracking).")
        else:
            if self.training is None or self.optim is None:
                logging.error("Cannot continue. (t > 0)")
                return self

        self.training["time_start"] = time.time()

        if method == "wasserstein":
            self._wasserstein_compile(
                target=self.target,
                t=t,
                T=T,
                verbose=verbose,
                **self.compiler_hyper,
            )
        elif method == "HST":
            self._HST_compile(
                target=self.target,
                t=t,
                T=T,
                verbose=verbose,
                **self.compiler_hyper,
            )
        elif method == "LET":
            self._LET_compile(
                target=self.target,
                t=t,
                T=T,
                verbose=verbose,
                **self.compiler_hyper,
            )

        self.training["time_end"] = time.time()
        self.bound_circuit = self.assign_parameters(dict(zip(self.parameters, self.theta)))

        self.learned_unitary_matrix = np.array(quinfo.Operator(self.bound_circuit))
        self.target_unitary_matrix = np.array(quinfo.Operator(self.target))
        phase_correction = np.angle(self.target_unitary_matrix[0, 0]) - np.angle(self.learned_unitary_matrix[0, 0])
        self.learned_unitary_matrix_phase_corrected = (np.exp(1j * phase_correction) * self.learned_unitary_matrix)

        return self

    def _LET_compile(
        self,
        target: QuantumCircuit,
        s: int,
        N_s: int,
        t: int = 0,
        T: int = 100,
        verbose: int = 1,
        local: bool = False,
        cnvgc_window: int = None,
        cnvgc_threshold: float = 1e-4,
        cnvgc_break: bool = True,
        **kwargs,
    ):
        """Compile using Loschmidt echo test (LET). Average LET if s>1.

        Args:
            target (QuantumCircuit): Target quantum circuit.
            s (int): Number of test states. If s=1, then SISC.
            N_s (int): Size of state pool.
            t (int, optional): Current iteration if resuming. Defaults to 0.
            T (int, optional): Number of iterations. Defaults to 100.
            verbose (int, optional): Log to console at each $verbose iteration. Defaults to 1.
            local (bool, optional): Use Local Loschmidt echo test. Defaults to False.
            cnvgc_window (int, optional): Number of last values of cost function to compute the
                variance of. Defaults to None.
            cnvgc_threshold (float, optional): Threshold for convergence. Defaults to 1e-4.
            cnvgc_break (bool, optional): Break if convergence is reached. Defaults to True.

        Returns:
            dict: Training dictionary storing all meta informations and logged values of training.
        """

        a_reg = qiskit.QuantumRegister(self.num_qubits, "a")
        c_reg = qiskit.ClassicalRegister(1, "c")
        if local:
            qc_U_V_dagger = QuantumCircuit(a_reg, c_reg)
        else:
            qc_U_V_dagger = QuantumCircuit(a_reg)
        qc_U_V_dagger.compose(target, a_reg, inplace=True)
        qc_U_V_dagger.compose(self.inverse(), a_reg, inplace=True)
        qc_U_V_dagger.barrier()
        
        if self.training["fixed_states"]:
            cnvgc_threshold = 1e-16
        if cnvgc_window is None:
            cnvgc_window = max([int(T / 10), 2])

        if t == 0:
            self.training.update({  #common
                "s": s,
                "N_s": N_s,
                "successful": False,
                "cnvgc_window": cnvgc_window,
                "cnvgc_threshold": cnvgc_threshold,
                "cnvgc_break": cnvgc_break})

            self.training.update({"local": local})  #specific
            self.training.update({
                "thetas": [],
                "HS": [],
                "Fbar": [],
                "fidelities": [],
                "gradients_per_state": [],
                "gradients": [],
                "CBS-fidelities": [],
                "training_duration": [],
                "iteration_duration": [],
                "C_LET": [],
                "C_LLET": [],})
            self.training["theta_init"] = dict(zip(self.parameters, self.theta))

        cnvgc_logged = False
        self.training["status"] = "finished"  #break criteria change this value

        self._print_training_config()
        
        if self.training["fixed_states"]:  # fixed states case. take all the states in pre-initialized state pool
            test_states = self.state_pool
        else:  # pick randomly s states from state pool
            test_states = [self.state_pool[ix] for ix in self.rng.choice(N_s, size=s, replace=False)]

        # W: state prep circ. Then, W U V_dg W_dg:
        qc_plus_test_states_StateFns = [
            qc_U_V_dagger.compose(state, front=True).compose(state.inverse(), front=False) for state in test_states]
        
        if local:
            qc_plus_test_states_State_Fns_local = []
            for circ in qc_plus_test_states_StateFns:
                for qix in range(self.num_qubits):
                    LLET_qc = circ.copy()
                    LLET_qc.measure(a_reg[qix], c_reg[0])
                    qc_plus_test_states_State_Fns_local.append(LLET_qc)
        else:
            # Need to measure all qubits for sampling probabilities
            for circ in qc_plus_test_states_StateFns:
                    circ.measure_all()

        for iteration in range(t, T):
            iteration_start_time = time.time()

            # here starts training
            training_start_time = time.time()
            
            if local:
                
                Probabilities = np.zeros((s, self.num_qubits))
                # Calculate probabilites from sampler
                probs = self.stats_calculator.get_probs(qc_plus_test_states_State_Fns_local, self.theta)
                for i, d in enumerate(probs):
                    Probabilities[i // self.num_qubits, i % self.num_qubits] = d.get(0, 0)
                # LET local cost = 1 - sum_j(Prob(0_j))/n
                self.training["C_LLET"].append(np.mean(1 - Probabilities.sum(axis=1)/self.num_qubits)/s)
                
                all_zero_grads_per_state = np.zeros((self.num_parameters, s))
                temp_array = np.zeros(self.num_parameters)
                
                # Calculate gradients for sampler
                evaluated_gradients = self.stats_calculator.get_samplergrads(qc_plus_test_states_State_Fns_local, self.theta) 
                
                for i in range(len(evaluated_gradients)):
                    if i % self.num_qubits == 0:
                        all_zero_grads_per_state[:, i // self.num_qubits] = temp_array/self.num_qubits
                        temp_array = np.zeros(self.num_parameters)
                    temp_array += np.array([x.get(0, 0) for x in evaluated_gradients[i]])

                gradvalues = -np.sum(all_zero_grads_per_state, axis=1) / self.num_qubits / s

            else:
                
                all_zero_grads_per_state, C_LET = self.stats_calculator.stats_LET(qc_plus_test_states_StateFns, self.theta)

                # (mean axis=1) averages over states, minus for maximizing probability
                gradvalues = -np.sum(all_zero_grads_per_state, axis=1) / s
                # LET global cost = 1 - Prob(00..00)
                # C_LET = np.array(mp_result_probs)[0]
                self.training["C_LET"].append(C_LET.item())

            self.theta = self.optim.update(parameter_values=self.theta, gradient_values=gradvalues)

            # here starts logging/tracking. do not count for training duration.
            self.training["training_duration"].append(np.round(time.time() - training_start_time, 2))

            self.training["gradients"].append(gradvalues)

            self.training["thetas"].append(dict(zip(qc_U_V_dagger.parameters, self.theta)))
            self.training["HS"].append(
                calc_HS_inner_product(
                    target,
                    self.assign_parameters(dict(zip(self.parameters, self.theta))),
                ))

            self.training["Fbar"].append(
                calc_Fbar(
                    target,
                    self.assign_parameters(dict(zip(self.parameters, self.theta))),
                ))

            # fidelities of test state ensemble
            self.training["fidelities"].append(calc_fidelities(test_states, target, self.assign_parameters(self.theta)))

            # fidelities of computational basis states |Y_CBS>, <Y_CBS| V^dagger U |Y_CBS>
            self.training["CBS-fidelities"].append(
                calc_fidelities(
                    self.comp_basis_states,
                    target,
                    self.assign_parameters(dict(zip(self.parameters, self.theta))),
                ))

            # end of iteration.
            self.training["iteration_duration"].append(np.round(time.time() - iteration_start_time, 2))

            self.training["t"] = iteration + 1

            cost = self.training["C_LLET"] if local else self.training["C_LET"]
            if verbose >= 1:
                if iteration % verbose == 0:
                    time_prediction = str(
                        datetime.timedelta(seconds=int((T - iteration - 1) *
                                                       np.mean(self.training["iteration_duration"][t:iteration + 1]))))

                    if self.training["FISC"]:
                        fid_str = f"InStFid: {np.mean(self.training['fidelities'][-1]):.3f}"
                    else:
                        fid_str = f"CBSfid: {np.mean(self.training['CBS-fidelities'][-1]):.3f} " \
                                  f"± {np.std(self.training['CBS-fidelities'][-1]):.3f}, " \
                                  f"{np.min(self.training['CBS-fidelities'][-1]):.3f}, {np.max(self.training['CBS-fidelities'][-1]):.3f}"

                    if iteration + 1 > cnvgc_window:
                        cnvgc_str = "{:.2e}".format(last_var(cost, cnvgc_window))
                    else:
                        cnvgc_str = "<window"

                    logging.info(
                        " %s %s/%s \t %s %.3f (var: %s) Fbar: %.3f %s duration: %s (%s) ETC: %s",
                        self.training["run_id"],
                        str(iteration + 1).rjust(len(str(T))),
                        T,
                        "C_LLET:" if local else "C_LET:",
                        cost[-1],
                        cnvgc_str,
                        self.training["Fbar"][-1],
                        fid_str,
                        str(datetime.timedelta(seconds=self.training["iteration_duration"][-1])).split(".",
                                                                                                       maxsplit=1)[0],
                        str(datetime.timedelta(seconds=self.training["training_duration"][-1])).split(".",
                                                                                                      maxsplit=1)[0],
                        time_prediction,
                    )

            # testing for convergence
            if iteration + 1 > cnvgc_window:
                if (last_var(cost, cnvgc_window) <= cnvgc_threshold and not cnvgc_logged):
                    logging.info("Convergence reached. Last %s values of cost has variance of less than %s. Abort: %s",
                                 cnvgc_window, cnvgc_threshold, cnvgc_break)
                    cnvgc_logged = True  # only show once
                    if cnvgc_break:
                        self.training["status"] = "convergence"
                        break

        return self.training

    def _HST_compile(
        self,
        target: QuantumCircuit,
        s: int = 1,
        N_s: int = 1,
        t: int = 0,
        T: int = 100,
        verbose: int = 1,
        local: bool = False,
        cnvgc_window: int = None,
        cnvgc_threshold: float = 1e-4,
        cnvgc_break: bool = True,
        **kwargs,
    ):
        """Compile using Hilbert-Schmidt Test (HST).

        Args:
            target (QuantumCircuit): Target quantum circuit.
            s (int): Number of test states. Only used for calculations of test state dep. fidelities.
            N_s (int): Size of state pool.
            t (int, optional): Current iteration if resuming. Defaults to 0.
            T (int, optional): Number of iterations. Defaults to 100.
            verbose (int, optional): Log to console at each $verbose iteration. Defaults to 1.
            local (bool, optional): Use Local Loschmidt echo test. Defaults to False.
            cnvgc_window (int, optional): Number of last values of cost function to compute the
                variance of. Defaults to None.
            cnvgc_threshold (float, optional): Threshold for convergence. Defaults to 1e-4.
            cnvgc_break (bool, optional): Break if convergence is reached. Defaults to True.

        Returns:
            dict: Training dictionary storing all meta informations and logged values of training.
        """

        a_reg = qiskit.QuantumRegister(self.num_qubits, "a")
        b_reg = qiskit.QuantumRegister(self.num_qubits, "b")
        c_reg = qiskit.ClassicalRegister(2, "c")
        if local:
            qc = QuantumCircuit(a_reg, b_reg, c_reg)
        else:
            qc = QuantumCircuit(a_reg, b_reg)
        qc.h(a_reg)
        qc.barrier()
        qc.cx(a_reg, b_reg)
        qc.barrier()
        qc.compose(target, a_reg, inplace=True)
        qc.compose(self.inverse(), a_reg, inplace=True)
        qc.barrier()

        # gather local HST circuits and corresponding probability gradients
        LHST_circuits = []
        #local_grads = []
        if local:
            for qix in range(self.num_qubits):
                LHST_qc = qc.copy()

                LHST_qc.cx(a_reg[qix], b_reg[qix])
                LHST_qc.h(a_reg[qix])
                LHST_qc.measure([a_reg[qix], b_reg[qix]], [c_reg[0], c_reg[1]])
                LHST_circuits.append(LHST_qc)

        for qix in range(self.num_qubits):
            qc.cx(a_reg[self.num_qubits - qix - 1], b_reg[self.num_qubits - qix - 1])
        qc.barrier()
        qc.h(a_reg)
        qc.measure_all()

        if cnvgc_window is None:
            cnvgc_window = max([int(T / 10), 2])

        if t == 0:
            self.training.update({  #common
                "s": s,
                "N_s": N_s,
                "successful": False,
                "cnvgc_window": cnvgc_window,
                "cnvgc_threshold": cnvgc_threshold,
                "cnvgc_break": cnvgc_break})

            self.training.update({"local": local})  #specific

            self.training.update({
                "thetas": [],
                "HS": [],
                "Fbar": [],
                "fidelities": [],
                "gradients_per_state": [],
                "gradients": [],
                "CBS-fidelities": [],
                "training_duration": [],
                "iteration_duration": [],
                "C_HST": [],
                "C_LHST": [],})
            self.training["theta_init"] = dict(zip(self.parameters, self.theta))

        cnvgc_logged = False
        self.training["status"] = "finished"  #break criteria change this value

        self._print_training_config()

        for iteration in range(t, T):
            iteration_start_time = time.time()
            training_start_time = time.time()

            if local:
                grad_values = np.zeros((self.num_parameters, self.num_qubits))
                evaluated_gradients = self.stats_calculator.get_samplergrads(LHST_circuits, self.theta) 
                probs = self.stats_calculator.get_probs(LHST_circuits, self.theta) 
                F_LHST = 0
                for d in probs:
                    F_LHST += d.get(0,0)
                for qix in range(self.num_qubits):
                    grad_values[:, qix] = np.array([-x.get(0, 0) for x in evaluated_gradients[qix]])/self.num_qubits
                gradvalues = np.sum(grad_values, axis=1)
                self.training["C_LHST"].append(1 - F_LHST / self.num_qubits)
            else:
                evaluated_gradients = self.stats_calculator.get_samplergrads([qc], self.theta) 
                # costs
                # C_HST is simply 1-Prob(allzero)
                probs = self.stats_calculator.get_probs([qc], self.theta)
                for d in probs:
                    self.training["C_HST"].append(1 - d.get(0, 0))
                gradvalues = np.array([-x.get(0,0) for x in evaluated_gradients[0]])

            self.theta = self.optim.update(parameter_values=self.theta, gradient_values=gradvalues)
            self.training["training_duration"].append(np.round((time.time() - training_start_time), 2))

            self.training["gradients"].append(gradvalues)

            self.training["thetas"].append(dict(zip(qc.parameters, self.theta)))
            self.training["HS"].append(
                calc_HS_inner_product(
                    target,
                    self.assign_parameters(dict(zip(self.parameters, self.theta))),
                    second_adjoint=False,
                ))
            self.training["Fbar"].append(
                calc_Fbar(
                    target,
                    self.assign_parameters(dict(zip(self.parameters, self.theta))),
                    second_adjoint=False,
                ))

            self.training["CBS-fidelities"].append(
                calc_fidelities(
                    self.comp_basis_states,
                    target,
                    self.assign_parameters(dict(zip(self.parameters, self.theta))),
                ))

            # just for fidelity tracking
            if self.training[
                    "fixed_states"]:  # fixed states case. just take all the states in pre-initialized state pool
                test_states_ensemble = self.state_pool
            else:  # pick randomly s states from state pool
                test_states_ensemble = [self.state_pool[ix] for ix in self.rng.choice(N_s, size=s, replace=False)]
            # fidelities of test state ensemble
            self.training["fidelities"].append(
                calc_fidelities(test_states_ensemble, target, self.assign_parameters(self.theta)))

            self.training["iteration_duration"].append(np.round((time.time() - iteration_start_time), 2))

            self.training["t"] = iteration + 1

            cost = self.training["C_LHST"] if local else self.training["C_HST"]
            if verbose >= 1:
                if iteration % verbose == 0:
                    time_prediction = str(
                        datetime.timedelta(seconds=int((T - iteration - 1) *
                                                       np.mean(self.training["iteration_duration"][t:iteration + 1]))))

                    if self.training["FISC"]:
                        fid_str = (f"InStFid: {np.mean(self.training['fidelities'][-1]):.3f}")
                    else:
                        fid_str = f"CBSfid: {np.mean(self.training['CBS-fidelities'][-1]):.3f} ± {np.std(self.training['CBS-fidelities'][-1]):.3f}, {np.min(self.training['CBS-fidelities'][-1]):.3f}, {np.max(self.training['CBS-fidelities'][-1]):.3f}"

                    if iteration + 1 > cnvgc_window:
                        cnvgc_str = "{:.2e}".format(last_var(cost, cnvgc_window))
                    else:
                        cnvgc_str = "<window"

                    logging.info(
                        "   %s %s/%s    %s %.3f (%s) Fbar: %.3f %s duration: %s (%s) ETC: %s",
                        self.training["run_id"],
                        str(iteration + 1).rjust(len(str(T))),
                        T,
                        "C_LHST" if local else "C_HST",
                        cost[-1],
                        cnvgc_str,
                        self.training["Fbar"][-1],
                        fid_str,
                        str(datetime.timedelta(seconds=self.training["iteration_duration"][-1])).split(".",
                                                                                                       maxsplit=1)[0],
                        str(datetime.timedelta(seconds=self.training["training_duration"][-1])).split(".",
                                                                                                      maxsplit=1)[0],
                        time_prediction,
                    )

            # testing for convergence
            if iteration + 1 > cnvgc_window:
                if (last_var(cost, cnvgc_window) <= cnvgc_threshold and not cnvgc_logged):
                    logging.info("Convergence reached. Last %s values of C_HST has variance of less than %s. Abort: %s",
                                 cnvgc_window, cnvgc_threshold, cnvgc_break)
                    cnvgc_logged = True  # only show once
                    if cnvgc_break:
                        self.training["status"] = "convergence"
                        break

        return self.training

    def _wasserstein_compile(
        self,
        target: QuantumCircuit,
        s,
        N_s,
        t: int = 0,
        T: int = 100,
        m: int = 0,
        k: int = 2,
        t_oc: int = 10,
        verbose: int = 1,
        P=0.8,
        cnvgc_window: int = None,
        cnvgc_threshold: float = 1e-4,
        cnvgc_break: bool = True,
        **kwargs,
    ):
        """Compile using Quantum Wasserstein Compilation.

        Args:
            target (QuantumCircuit): Target quantum circuit.
            s (int): Number of test states. If s=1, then SISC.
            N_s (int): Size of state pool.
            t (int, optional): Current iteration if resuming. Defaults to 0.
            T (int, optional): Number of iterations. Defaults to 100.
            m (int, optional): Number of Pauli observables. Defaults to using all k-lcoal n-qubit
                Pauli observables.
            k (int, optional): Locality of Pauli observables. Defaults to 2.
            t_oc (int, optional): Cycle observables every t_oc iterations. Defaults to 10.
            verbose (int, optional): Log to console at each $verbose iteration. Defaults to 1.
            P (float, optional): Control operator cycling. Obserable is free for cycling if
                expectation value difference is smaller than P*c where c is smallest active
                expectation value difference. Defaults to 0.8.
            cnvgc_window (int, optional): Number of last values of cost function to compute the
                variance of. Defaults to None.
            cnvgc_threshold (float, optional): Threshold for convergence. Defaults to 1e-4.
            cnvgc_break (bool, optional): Break if convergence is reached. Defaults to True.


        Returns:
            dict: Training dictionary storing all meta informations and logged values of training.
        """

        self.paulis = init_full_paulis(num_qubits=self.num_qubits, k_local=k)
        M = len(self.paulis)
        self.pauli_strings = convert_paulis_num_to_strings(self.paulis)
        self.pauli_operators = quinfo.PauliList([quinfo.Pauli(operator) for operator in self.pauli_strings])

        if m == 0:
            m = M

        if m < M:
            # if operator cycling, initiliaze mask with #operator_cycling Trues
            operator_mask = np.array([True] * m + [False] * (M - m))
            self.rng.shuffle(operator_mask)
        else:
            # if not operator cycling, use mask with all True
            operator_mask = np.array([True] * M)

        # precomputation if fixed states
        if self.training["fixed_states"]:
            # precompute: get all expectation values for every test state + target unitary and operator
            targetPlusTestStates = [target.compose(state, front=True) for state in self.state_pool]
            pre_target_expect_vals = self.stats_calculator.get_exps(targetPlusTestStates, list(compress(self.pauli_operators, operator_mask))) 
            cnvgc_threshold = 1e-8
        else:
            pre_target_expect_vals = None

        if cnvgc_window is None:
            cnvgc_window = max([int(T / 10), 2])

        if t == 0:
            self.training.update({  #common
                "s": s,
                "N_s": N_s,
                "successful": False,
                "cnvgc_window": cnvgc_window,
                "cnvgc_threshold": cnvgc_threshold,
                "cnvgc_break": cnvgc_break})

            self.training.update({  #specific
                "m": m,
                "M": M,
                "P": P,
                "t_oc": t_oc,
                "operator_cycling": (m < M),
                "k": k,
                "set_M": self.pauli_strings})

            self.training.update({
                "thetas": [],
                "HS": [],
                "Fbar": [],
                "fidelities": [],
                "gradients_per_state": [],
                "gradients": [],
                "CBS-fidelities": [],
                "training_duration": [],
                "iteration_duration": [],
                "D_EM": [],
                "C_EM": [],
                "mask": []})
            self.training["theta_init"] = dict(zip(self.parameters, self.theta))

        cnvgc_logged = False
        self.training["status"] = "finished"  #break criteria change this value
        
        self._print_training_config()

        for iteration in range(t, T):
            iteration_start_time = time.time()

            self.training["mask"].append(operator_mask)

            if self.training["fixed_states"]:
                test_states = self.state_pool
            else:  # pick states out of state pool
                test_states = [self.state_pool[ix] for ix in self.rng.choice(N_s, size=s, replace=False)]

            theta_dict = dict(zip(self.parameters, self.theta))

            training_start_time = time.time()
            D_EM, list_H_max, expect_vals_diff, active_operators = self._measure_wasserstein_distance(
                self.target, self.assign_parameters(theta_dict), test_states, operator_mask, pre_target_expect_vals)
            gradients, gradients_per_state = self._get_gradient_on_H_EM(D_EM, list_H_max, test_states)

            self.theta = self.optim.update(parameter_values=self.theta, gradient_values=gradients)

            # here starts logging/tracking. do not count for training duration.
            self.training["training_duration"].append(np.round(time.time() - training_start_time, 2))

            self.training["gradients"].append(gradients)
            self.training["gradients_per_state"].append(gradients_per_state)

            self.training["D_EM"].append(D_EM)
            self.training["C_EM"].append(np.sum(np.square(D_EM)) / len(D_EM))

            self.training["thetas"].append(dict(zip(self.parameters, self.theta)))
            self.training["HS"].append(calc_HS_inner_product(target, self.assign_parameters(self.theta)))
            self.training["Fbar"].append(calc_Fbar(target, self.assign_parameters(self.theta)))
            self.training["fidelities"].append(calc_fidelities(test_states, target, self.assign_parameters(self.theta)))

            self.training["CBS-fidelities"].append(
                calc_fidelities(
                    input_states=self.comp_basis_states,
                    target_qc=target,
                    generated_qc=self.assign_parameters(dict(zip(self.parameters, self.theta))),
                ))

            if m < M and (iteration % t_oc == 0):
                operator_mask, cycling_string = self._cycle_operators(operator_mask, expect_vals_diff, active_operators,
                                                                      P)
            else:
                cycling_string = ("#cycled: [in {}]".format(t_oc - (iteration % t_oc)) if (m < M) else "#cycled: OFF")

            self.training["iteration_duration"].append(np.round(time.time() - iteration_start_time, 2))
            self.training["t"] = iteration + 1

            # progress report
            if verbose >= 1:
                if iteration % verbose == 0:
                    time_prediction = str(
                        datetime.timedelta(seconds=int((T - iteration - 1) *
                                                       np.mean(self.training["iteration_duration"][t:iteration + 1]))))

                    if self.training["FISC"]:
                        fid_str = (f"InStFid: {np.mean(self.training['fidelities'][-1]):.3f}")
                    else:
                        fid_str = f"CBSfid: {np.mean(self.training['CBS-fidelities'][-1]):.3f} ± {np.std(self.training['CBS-fidelities'][-1]):.3f}, {np.min(self.training['CBS-fidelities'][-1]):.3f}, {np.max(self.training['CBS-fidelities'][-1]):.3f}"

                    if iteration + 1 > cnvgc_window:
                        cnvgc_str = "{:.2e}".format(last_var(self.training["C_EM"], cnvgc_window))
                    else:
                        cnvgc_str = "<window"

                    logging.info(
                        "  %s %s/%s ETC: %s C_EM: %.4f (var: %s) Fbar: %.4f %s %s duration: %s (%s)",
                        self.training["run_id"],
                        str(iteration + 1).rjust(len(str(T))),
                        T,
                        time_prediction,
                        self.training["C_EM"][-1],
                        cnvgc_str,
                        self.training["Fbar"][-1],
                        fid_str,
                        cycling_string,
                        datetime.timedelta(seconds=int(self.training["iteration_duration"][-1])),
                        datetime.timedelta(seconds=int(self.training["training_duration"][-1])),
                    )

            # testing for convergence
            if iteration + 1 > cnvgc_window:
                if (last_var(self.training["C_EM"], cnvgc_window) <= cnvgc_threshold and not cnvgc_logged):
                    logging.info("Convergence reached. Last %s values of C_EM has variance of less than %s. Abort: %s",
                                 cnvgc_window, cnvgc_threshold, cnvgc_break)
                    cnvgc_logged = True  # only show once
                    if cnvgc_break:
                        self.training["status"] = "convergence"
                        break

        return self.training

    def _cycle_operators(self, operator_mask, expect_vals_diff, active_operators, P):
        """Manipulate operator_mask to cycle Pauli observables.

        Args:
            operator_mask (array): Mask of Pauli observables. True if corresponing operator was evaluated.
            expect_vals_diff (array): expectation value differences.
            active_operators (array): list of bools, true if operaot is active.
            P (float): Factor for cycling out inactive observables.

        Returns:
            array,str: new operator mask, string for logging
        """

        # [OC 3.] find smallest active exp diff as threshold value
        exp_diff_smallest_active = np.min(np.abs(expect_vals_diff[np.array(active_operators)]))  # c*

        # [OC 4.a] max exp val diff per operator
        exp_diff_max_per_operator = np.max(np.abs(expect_vals_diff), axis=0)

        # [OC 4.b] find operators that are smaller than P * (smallest active exp diff)
        operators_to_close = (exp_diff_max_per_operator <= P * exp_diff_smallest_active)

        # [OC 6.a] get list of operators that can be picked as new operators (not active and not marked for cycling)
        free_operators = np.where(~operator_mask)[0]

        # [OC 6b] how many operators to pick from free operators
        num_op_to_cycle = np.count_nonzero(operators_to_close)
        operators_to_open = self.rng.choice(
            free_operators,
            size=min(num_op_to_cycle, len(free_operators)),
            replace=False,
        )

        # [OC 7] manipulate mask == cycling operators
        operator_mask[np.where(operator_mask)[0][np.where(operators_to_close)[0]]] = False
        operator_mask[operators_to_open] = True
        cycling_string = f"#cycled: {num_op_to_cycle}".ljust(15)

        return operator_mask, cycling_string

    def _get_gradient_on_H_EM(self, D_EM, H_EM_list, probe_states):
        """_summary_

        Args:
            D_EM (array): Wasserstein distances between target and generated output state.
            H_max_list (list(Operator)): List of EM Observables. (Note: H_EM is dependent on
                input state.)
            probe_states (list(QuantumCircuits)): List of states.

        Returns:
            array, array: mean gradients over states, gradients per state
        """
        probe_state_plus_circ = [self.compose(state, front = True).decompose() for state in probe_states]
        
        gradients_per_state = self.stats_calculator.grads_wasserstein(probe_state_plus_circ, H_EM_list, self.theta)

        # scaling acc. to derivation rule. Ch 3.1.2
        gradients = np.einsum("i,ik->ik", 2 * D_EM, gradients_per_state)  
        gradients_mean_over_states = np.mean(gradients, axis=0)

        return gradients_mean_over_states, gradients_per_state

    def _measure_wasserstein_distance(
        self,
        circuit1,
        circuit2,
        probe_states,
        operator_mask=None,
        circuit1_pre_computed_exp_vals=None,
    ):
        """Measure Wasserstein Distance between circuit 1 and circuit 2 given input states probe_states.

        Args:
            circuit1 (QuantumCircuit): First Circuit.
            circuit2 (QuantumCircuit): Second Circuit.
            probe_states (list(QuantumCircuit)): List of input states.
            operator_mask (array, optional): Boolean mask of available Pauli observables. Defaults to
                None. Then all observables are taken.
            circuit1_pre_computed_exp_vals (array, optional): If fixed_states, target output state
                expectation values can be computed beforehand. Defaults to None.

        Returns:
             D_EM (array): Measured Wasserstein distances.
             H_MAX (list(operators)): constructed EM observables.
             expect_vals_diff (array): Measured expectation value differences for every Pauli Observable.
             active_operators (array): boolean mask of active operators out of all Pauli observables.
        """

        if operator_mask is None:
            operator_mask = np.array([True] * len(self.paulis))

        masked_paulis = self.paulis[operator_mask]
        masked_pauli_operators = list(compress(self.pauli_operators, operator_mask))
        masked_pauli_strings = list(compress(self.pauli_strings, operator_mask))

        circuit1_plus_probes = [state.compose(circuit1) for state in probe_states] 
        circuit2_plus_probes = [state.compose(circuit2) for state in probe_states] 
        
        if self.training.get("fixed_states", False) and circuit1_pre_computed_exp_vals is not None:
            circuit1_expect_vals = circuit1_pre_computed_exp_vals
        else:
            circuit1_expect_vals = self.stats_calculator.expvals_wasserstein(circuit1_plus_probes, masked_pauli_operators, method = "custom")

        circuit2_expect_vals = self.stats_calculator.expvals_wasserstein(circuit2_plus_probes, masked_pauli_operators)

        expect_vals_diff = np.subtract(circuit2_expect_vals, circuit1_expect_vals).reshape(len(probe_states), len(masked_pauli_operators))

        H_MAX = []
        active_operators = []
        
        pool = mp.Pool(mp.cpu_count())
    
        w_i_max = pool.starmap(solve_Hmax, [(masked_paulis, expect_vals_diff_state) for expect_vals_diff_state in expect_vals_diff])
    
        pool.close()
    
        # find and store all active operators (for operator cycling)
        active_operators  = np.abs(w_i_max) > 1e-9
    
        # find all active pauli strings (active = weight is over threshold of 1e-9
        idx_first_active_pauli = np.argmax(np.abs(w_i_max) > 1e-9, axis = 1)

        for state_idx in range(len(expect_vals_diff)):

            # "initialize" the H_max operator by setting it to the first active operator with coefficient = its weights
            H_max = quinfo.SparsePauliOp(quinfo.Pauli(masked_pauli_strings[idx_first_active_pauli[state_idx]]), 
                                         coeffs=w_i_max[state_idx][idx_first_active_pauli[state_idx]])

            # add the rest of the active operators
            for idx in range(idx_first_active_pauli[state_idx] + 1, len(masked_paulis)):
                if np.abs(w_i_max[state_idx][idx]) >= 1e-9:
                    H_max += quinfo.SparsePauliOp(quinfo.Pauli(masked_pauli_strings[idx]), coeffs=w_i_max[state_idx][idx])

            H_MAX.append(H_max)
        
        exp_circuit2 = self.stats_calculator.expvals_wasserstein(circuit2_plus_probes, H_MAX, method = "normal")
        
        exp_circuit1 = self.stats_calculator.expvals_wasserstein(circuit1_plus_probes, H_MAX, method = "normal")
    
        D_EM = np.subtract(exp_circuit2, exp_circuit1)

        return D_EM, H_MAX, expect_vals_diff, active_operators

    def _initialize_from_argument(self, initialization_argument):
        """Initializes parameters of ansatz.

        Args:
            initialization_argument (str or array): If "random", the parameters are sampled randomly.
                If "fixed", the parameters are loaded from a file, to ensure constistency over
                different runs. If array/list of values, these are used as initialization.

        Raises:
            ValueError: If str but unknown.
            ValueError: If list/array of values, but length does not fit the number of parameters of
                ansatz.
            TypeError: If neither str nor list.

        Returns:
             theta_values (array): values of parameters.
             initialization_method (str): used initializtaion as str for logging.
        """

        if isinstance(initialization_argument, str):
            if initialization_argument == "random":
                theta_values = self.rng.uniform(low=-1, high=1, size=(self.num_parameters))
            elif initialization_argument == "fixed":
                theta_values = load_initializations(for_ansatz=True)[0:self.num_parameters]
            else:
                raise ValueError

            initialization_method = initialization_argument

        elif isinstance(initialization_argument, (list, tuple, np.ndarray)):
            if len(initialization_argument) != self.num_parameters:
                raise ValueError
            theta_values = np.array(initialization_argument)
            initialization_method = theta_values
        else:
            raise TypeError

        return theta_values, initialization_method

    def _initialize_test_states(self, test_states_argument):
        """Initializes the pool of probe states.

        Args:
            test_states_argument (str or list(QuantumCircuit)): If str, then initialized accordingly.
                Possible values are "haar" for approximately haar random states using a PauliTwoDesign,
                "product" for simple seperable states, "basis" or "cbs" for computational basis states.
                Alternatively, a list of quantum circuits is accepted.

        Raises:
            KeyError: If s is not given in compiler hyper.
            ValueError: If str but unknown.
            TypeError: If neither str nor list.
        """

        if "s" not in self.compiler_hyper:
            logging.error("s is missing in compiler hyper. This argument is mandatory.")
            raise KeyError

        if isinstance(test_states_argument, str):
            test_states_argument = test_states_argument.lower()
            if test_states_argument == "haar":
                self.training["test_states"] = "haar"
                self.state_pool = get_random_states(
                    self.num_qubits,
                    quantity=self.compiler_hyper["N_s"],
                    epsilon=self.epsilon_two_design,
                    rng=self.rng,
                )
                self.training["epsilon_two_design"] = self.epsilon_two_design
            elif test_states_argument == "product":
                self.training["test_states"] = "product"
                self.state_pool = get_random_product_states(self.num_qubits,
                                                            num_states=self.compiler_hyper["N_s"],
                                                            rng=self.rng)
            elif test_states_argument in ("basis", "basis-states", "cbs"):
                self.training["test_states"] = "cbs"
                self.state_pool = get_comp_basis_states(self.num_qubits)
                self.compiler_hyper["N_s"] = self.compiler_hyper.get("N_s", len(
                    self.state_pool))  # if N_s not given, replace by |comp basis states|

                if self.compiler_hyper["N_s"] != len(
                        self.state_pool):  # if N_s is given and smaller than number of possible states, pick N_s states
                    self.state_pool = [
                        self.state_pool[ix] for ix in self.rng.choice(
                            len(self.state_pool),
                            size=self.compiler_hyper["N_s"],
                            replace=False,
                        )]
            elif test_states_argument in ("allzero", "all-zero"):
                self.training["test_states"] = "allzero"
                self.state_pool = [QuantumCircuit(self.num_qubits, name="|0..0>")]
                self.compiler_hyper["N_s"] = 1
            else:
                raise ValueError
        elif isinstance(test_states_argument, list):
            self.training["test_states"] = "custom"
            self.state_pool = test_states_argument
            self.compiler_hyper["N_s"] = self.compiler_hyper.get("N_s", len(
                self.state_pool))  # if N_s not given, replace by |given states|

            if self.compiler_hyper["N_s"] != len(
                    self.state_pool):  # if N_s is given and smaller than number of possible states, pick N_s states
                self.state_pool = [
                    self.state_pool[ix] for ix in self.rng.choice(
                        len(self.state_pool),
                        size=self.compiler_hyper["N_s"],
                        replace=False,
                    )]
        else:
            raise TypeError

        self.training["FISC"] = self.compiler_hyper["N_s"] == 1
        self.training["fixed_states"] = (self.compiler_hyper["N_s"] == self.compiler_hyper["s"])

    def _print_training_config(self):
        """ Printing the config ahead of training."""

        logging.info("Starting compilation with config:")
        logging.info("n=%s; method=%s; T=%s; t=%s; target=%s; ansatz=%s; test-states=%s", self.training["n"],
                     self.training["method"], self.training["T"], self.training["t"], self.training["target-name"],
                     self.training["ansatz-name"], self.training["test_states"])

        if self.training["method"] == "wasserstein":
            logging.info(
                "s=%s; N_s=%s; FISC=%s; k=%s; m=%s; M=%s; op_cycling=%s;  t_oc=%s; P=%s;  cnvgc_break=%s; cnvgc_window=%i; cnvgc_threshold=%s",
                self.training["s"],
                self.training["N_s"],
                self.training["FISC"],
                self.training["k"],
                self.training["m"],
                self.training["M"],
                self.training["operator_cycling"],
                self.training["t_oc"],
                self.training["P"],
                self.training["cnvgc_break"],
                self.training["cnvgc_window"],
                self.training["cnvgc_threshold"],
            )

        elif self.training["method"] in ("LET", "HST"):
            logging.info(
                "s=%s; N_s=%s; FISC=%s; local=%s",
                self.training["s"],
                self.training["N_s"],
                self.training["FISC"],
                self.training["local"],
            )

        logging.info("Optimizer configuration: %s", self.optim.hyper)

    def dump(self, dir_out: Path, prefix: str = ""):
        """Dumps qpy serialized Generator, Target and ansatz together with training as json into
        a tar.gz file named according to the run_id."""

        dir_out = Path(dir_out)
        if not dir_out.is_dir():
            raise TypeError("please pass a directory")

        meta = self.training
        meta.pop("target", None)

        try:
            if isinstance(
                    list(meta["thetas"][0].keys())[0],
                    qiskit.circuit.parametervector.ParameterVectorElement,
            ):
                meta["thetas"] = [[(parameter_vector_element.name, angle)
                                   for parameter_vector_element, angle in thetas_it.items()]
                                  for thetas_it in meta["thetas"]]
                meta["theta_init"] = [(parameter_vector_element.name, angle)
                                      for parameter_vector_element, angle in meta["theta_init"].items()]
        except:
            pass

        path_training = dir_out / (prefix + self.training["run_id"] + "_training.json")
        with path_training.open("w") as file_handle:
            json.dump(meta, file_handle, cls=NumpyEncoder)

        path_ansatz = dir_out / (prefix + self.training["run_id"] + "_ansatz.qpy")
        with path_ansatz.open("wb") as file_handle:
            qpy.dump(self.ansatz, file_handle)

        path_target = dir_out / (prefix + self.training["run_id"] + "_target.qpy")
        with path_target.open("wb") as file_handle:
            qpy.dump(self.target, file_handle)

        path_generator = dir_out / (prefix + self.training["run_id"] + "_generator.qpy")
        with path_generator.open("wb") as file_handle:
            qpy.dump(self.bound_circuit, file_handle)

        with tarfile.open(dir_out / (prefix + self.training["run_id"] + ".tar.gz"), "w:gz") as tar:
            for p in [path_training, path_ansatz, path_target, path_generator]:
                tar.add(p, arcname=p.name)

        for p in [path_training, path_ansatz, path_target, path_generator]:
            p.unlink()

    def _pickle(self, file_path: Path):
        """Pickle Generator. Use with caution because of compatiblity issues when different qiskit
        versions are used."""

        file_path = Path(file_path)
        if file_path.exists():
            raise TypeError("{} already_exists".format(file_path))

        with file_path.open("wb") as file_handle:
            pickle.dump(self, file_handle)


def gen_from_pickle(file_path: Path, resume=True, delete_pickle=True):
    """Unpickles a Generator to resume compilation."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise TypeError("File to unpickle does not exist")

    with file_path.open("rb") as file_handle:
        gen = pickle.load(file_handle)

    if delete_pickle:
        file_path.unlink()

    return gen


if __name__ == "__main__":

    n = 4
    depth = 1
    rng = np.random.default_rng(seed = 5) 

    target = circuit_parser("HEA", nqubits=n, entanglement = 'full', depth=depth, rng = rng)
    ansatz = circuit_parser("HEA", nqubits=n, entanglement = 'full', depth=depth, rng = rng, is_target=False)
    start = time.time()
    gen = Generator(ansatz=ansatz, rng = rng)
    gen.compile(
        target_qc=target,
        method="wasserstein",
        test_states="product",
        initialization_argument='fixed',
        T=100,
        compiler_hyper=dict(m=0, k =n, s = 8, N_s=16, local=False, cnvgc_break=True),
        optimizer_hyper = dict(method="adam", eta=0.2),   
    )
    print("Time taken: ", time.time() - start)
    print("end")