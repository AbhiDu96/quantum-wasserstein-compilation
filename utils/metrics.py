"""
This module serves the purpose of extracting training data from already compiled runs for further analysis.
"""

import os
import datetime
import json
import logging
import tarfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shortuuid
from qiskit import qpy, transpile
from qiskit.providers.fake_provider import Fake27QPulseV1 as fake_boeblingen
from qiskit.qasm2 import dumps
from tqdm.auto import tqdm


viridis = ["#fde725", "#5ec962", "#21918c", "#3b528b", "#440154"]
warnings.filterwarnings("ignore", message="The ParameterVector: ")


class experiment():

    def __init__(self, name):
        self.name = name

        if Path(os.path.dirname(os.getcwd())+'/Experiments/'+name).exists():
            self.path = Path(os.path.dirname(os.getcwd())+'/Experiments/'+name)
        else:
            raise ValueError("directory not found")

        logging.info("Directory Path: %s", self.path)
        
        if "localityCompil" in name.split("_") or "StateCompile" in name.split("_") or "UC" in name.split("_"):

            self.meta = None
            self.C_EM = None
            self.C_HST = None
            self.C_LET = None
            self.cost = None
            self.grad_l1 = None
            self.grad_l2 = None
            self.HS = None
            self.Fbar = None
            self.thetas = None
            self.state_gradients = None
            self.gradients = None
            self.MISF = None
            self.MISIF = None
            self.cost = None
            self.fisc_fidelity = None
            self.fisc_infidelity = None
            
        elif "OneStepGrad" in name.split("_"):
            
            self.method = None
            self.num_states = None
            self.depth = None

    def load(self, max_count: int = 10e5, with_transpiled_depth_ratio: bool = False):
        metadata = []
        dict_C_EM = {}
        dict_C_LET = {}
        dict_C_HST = {}
        dict_cost = {}
        dict_grad_l1 = {}
        dict_grad_l2 = {}
        dfs_thetas = {}
        dict_HS = {}
        dict_Fbar = {}
        dict_gradients = {}
        dfs_state_gradients = {}
        dict_MISF = {}
        dict_fisc_fidelity = {}
        dict_transpiled_depths = {}

        count = 0

        list_files = set(filename for filename in self.path.glob("*.tar.gz"))

        if len(list_files) == 0:
            list_files = set(filename for filename in (self.path / "archive").glob("run_*") if filename.suffix == "")

        if len(list_files) == 0:
            logging.error("Cannot find any run file.")
            return self

        for file in tqdm(list(list_files), desc=self.name):
            count += 1
            if count > max_count:
                break

            tar = tarfile.open(file, "r:gz")
            for member in tar.getmembers():
                f = tar.extractfile(member)
                if f is not None:
                    if "_training.json" in member.name:
                        tr = json.load(f)
                    elif "_generator.qpy" in member.name:
                        gen = qpy.load(f)[0]
                    elif "_ansatz.qpy" in member.name:
                        ansatz = qpy.load(f)[0]
                    elif "_target.qpy" in member.name:
                        target = qpy.load(f)[0]
                    else:
                        logging.info("File %s in archive %s is not read out.", member.name, file.name)

            run_id = tr.get("run_id", "+" + shortuuid.uuid())

            if with_transpiled_depth_ratio:

                ansatz_mock_qasm_hash = hash(dumps(ansatz.assign_parameters(np.ones(ansatz.num_parameters))))
                try:
                    target_qasm_hash = hash(dumps(target))
                except:
                    target_qasm_hash = hash(target.decompose().draw())

                if ansatz_mock_qasm_hash not in dict_transpiled_depths:
                    dict_transpiled_depths[ansatz_mock_qasm_hash] = transpile(ansatz,
                                                                              backend=fake_boeblingen(),
                                                                              optimization_level=0).depth()

                if target_qasm_hash not in dict_transpiled_depths:
                    dict_transpiled_depths[target_qasm_hash] = transpile(target,
                                                                         backend=fake_boeblingen(),
                                                                         optimization_level=0).depth()

                depth_ratio = dict_transpiled_depths[ansatz_mock_qasm_hash] / dict_transpiled_depths[target_qasm_hash]
            else:
                depth_ratio = 0

            ############################# data manipulation #############################
            d = 2**gen.num_qubits
            iHS = tr["HS"]
            Fbar = tr["Fbar"]
            InFbar = 1 - np.array(Fbar)
            target_params = np.array(target.data[0].operation.params)

            if tr["FISC"]:
                fisc_fidelity = np.array(tr.get("fidelities")).ravel()
                fisc_infidelity = 1 - fisc_fidelity
                fisc_infidelity[fisc_infidelity < 8.881784e-16] = 8.881784e-16
                MISF = None
            else:
                fisc_fidelity = [None]
                fisc_infidelity = [None]
                MISF = np.array(tr.get("fidelities")).mean(axis=1)

            if tr["method"] == "wasserstein":
                gradients = tr.get("gradients")
                grad_l1_norm = np.linalg.norm(gradients, ord=1, axis=1) / gen.num_qubits
                grad_l2_norm = np.linalg.norm(gradients, ord=2, axis=1) / np.sqrt(gen.num_qubits)
            else:
                gradients = tr.get("gradients")
                grad_l1_norm = np.linalg.norm(gradients, ord=1, axis=1) / gen.num_qubits
                grad_l2_norm = np.linalg.norm(gradients, ord=2, axis=1) / np.sqrt(gen.num_qubits)

            if tr["method"] == "wasserstein":
                cost = tr["C_EM"]
            elif tr["method"] == "HST":
                cost = tr["C_HST"]
            elif tr["method"] == "LET":
                cost = tr["C_LET"]

            run_meta = {
                "run_id": run_id,
                "method": tr.get("method"),
                "methodk": tr.get("method") + str(tr.get("k", "")),
                "n": gen.num_qubits,
                "s": tr.get("s"),
                "T": tr.get("T")[-1] if isinstance(tr.get("T"), list) else tr.get("T"),
                "M": tr.get('M'),
                "HS_final": tr['HS'][-1],
                "Fbar_final": Fbar[-1],
                "InFbar_final": InFbar[-1],
                "C_final": cost[-1],
                "FISC_fidelity_final": fisc_fidelity[-1],
                "FISC_infidelity_final": fisc_infidelity[-1],
                "status": tr.get("status"),
                "N_s": tr.get('N_s'),
                "P": tr.get('P'),
                "epsilon_two_design": tr.get('epsilon_two_design'),
                "optimizer": tr.get("optimizer"),
                "eta": tr.get("eta"),
                "eta_init": tr.get("eta_init"),
                "eta_final": tr.get("eta_final"),
                "beta1": tr.get('beta1'),
                "beta2": tr.get('beta2'),
                "optim_T": tr.get('optim_T'),
                "k": tr.get('k'),
                "m": tr.get("m"),
                "t_oc": tr.get('t_oc'),
                "operator_cyling": tr.get('operator_cycling'),
                "local": tr.get('local'),
                "FISC": tr.get('FISC'),
                "duration": datetime.timedelta(seconds=tr['time_end'] - tr['time_start']).total_seconds(),
                "parameters": ansatz.num_parameters,
                "depth": ansatz.depth(),
                "target_name": tr.get("target-name").split("-")[0],
                "ansatz_name": ansatz.name,
                "test_states": tr.get("test_states"),
                "initialization": tr.get("initialization"),
                "fixed_states": tr.get("fixed_states"),
                "depth_ratio": depth_ratio,
                "break1": np.argmax(InFbar < 1e-1) if np.argmax(InFbar < 1e-1) > 0 else np.nan,
                "break2": np.argmax(InFbar < 1e-2) if np.argmax(InFbar < 1e-2) > 0 else np.nan,
                "break3": np.argmax(InFbar < 1e-3) if np.argmax(InFbar < 1e-3) > 0 else np.nan,
                "break4": np.argmax(InFbar < 1e-4) if np.argmax(InFbar < 1e-4) > 0 else np.nan,
                "break5": np.argmax(InFbar < 1e-5) if np.argmax(InFbar < 1e-5) > 0 else np.nan}

            metadata.append(dict(sorted(run_meta.items())))

            dict_C_EM[run_id] = pd.Series(tr.get("C_EM"), dtype=float)
            dict_C_HST[run_id] = pd.Series(tr.get("C_HST"), dtype=float)
            dict_C_LET[run_id] = pd.Series(tr.get("C_LET"), dtype=float)

            dict_grad_l1[run_id] = pd.Series(grad_l1_norm, dtype=float)
            dict_grad_l2[run_id] = pd.Series(grad_l2_norm, dtype=float)
            dict_HS[run_id] = pd.Series(tr["HS"], dtype=float)
            dict_Fbar[run_id] = pd.Series(Fbar, dtype=float)
            dict_gradients[run_id] = pd.Series(gradients, dtype=float)
            dict_fisc_fidelity[run_id] = pd.Series(fisc_fidelity, dtype=float)
            dict_MISF[run_id] = pd.Series(MISF, dtype=float)

            dfs_state_gradients[run_id] = pd.DataFrame(tr.get("gradients_per_state"))

            if "thetas" in tr:
                if "theta_init" in tr:
                    dfs_thetas[run_id] = pd.DataFrame({
                        tr["thetas"][0][prm][0]:
                        [tr["theta_init"][prm][1]] + [tr["thetas"][it][prm][1] for it in range(len(tr["thetas"]))]
                        for prm in range(len(tr["thetas"][0]))})
                else:
                    dfs_thetas[run_id] = pd.DataFrame({
                        tr["thetas"][0][prm][0]: [tr["thetas"][it][prm][1] for it in range(len(tr["thetas"]))
                                                 ] for prm in range(len(tr["thetas"][0]))})

        metadf = pd.DataFrame(metadata)
        self.meta = metadf.set_index("run_id")

        self.C_EM = pd.DataFrame(dict_C_EM)
        self.C_HST = pd.DataFrame(dict_C_HST)
        self.C_LET = pd.DataFrame(dict_C_LET)
        self.cost = pd.DataFrame(dict_cost)
        self.cost[self.cost < 8.881784e-16] = 8.881784e-16  # constrain to strictly positive cost
        self.cost_inv = 1 / self.cost
        self.grad_l1 = pd.DataFrame(dict_grad_l1)
        self.grad_l2 = pd.DataFrame(dict_grad_l2)
        self.HS = pd.DataFrame(dict_HS)
        self.Fbar = pd.DataFrame(dict_Fbar)
        self.InFbar = 1 - self.Fbar
        self.thetas = dfs_thetas
        self.gradients = dict_gradients
        self.state_gradients = dfs_state_gradients
        self.fisc_fidelity = pd.DataFrame(dict_fisc_fidelity)
        self.fisc_infidelity = 1 - pd.DataFrame(dict_fisc_fidelity)
        self.MISF = pd.DataFrame(dict_MISF)
        self.MISIF = 1 - pd.DataFrame(dict_MISF)

        logging.info("Experiment %s from %s loaded. %s runs found.", self.name, self.path, len(self.meta))
        return self
