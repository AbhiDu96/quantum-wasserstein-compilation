from configparser import ConfigParser
from argparse import ArgumentParser
import time

import numpy as np

from utils.circuits import circuit_parser
from compilation_qwgan.compiler import Generator

parser = ArgumentParser('parameters')

parser.add_argument('--num_qubits', type = int, default = 4, help = "Number of qubits for target and ansatz.")
parser.add_argument('--depth', type = int, default = 3, help = "Depth of the circuit.")
parser.add_argument('--seed', type = int, default = 42, help = "Seed for random number generator.")
parser.add_argument('--target_circuit', type = str, default = 'HEA', help = "Circuit type to be generated.")
parser.add_argument('--ansatz_circuit', type = str, default = 'HEA', help = "Circuit type to be generated.")
parser.add_argument('--entanglement', type = str, default = 'circular', help = "Entanglement type for circuit.")
parser.add_argument('--method', type = str, default = 'wasserstein', help = 'Method to run compilation. Possible methods are LET, HST and wasserstein.')
parser.add_argument('--test_states', type = str, default = 'product', help = 'Test states to be used for training. Possible states are product, haar, basis, allzero')
parser.add_argument('--type_of_states', type = str, default = 'random', help = 'Type of states to be used for training. Possible types are random, fixed')
parser.add_argument('--time_steps', type = int, default = 100, help = 'Number of time steps for training.')
parser.add_argument('--num_states', type = int, default = 8, help = 'Number of states to be used for training.')
parser.add_argument('--state_pool', type = int, default = 8, help = 'The size of the state pool from which the test states are sampled whenever num_states < state_pool. num_states should always be less than or equal to state_pool.')
parser.add_argument('--locality', type = bool, default = False, help = "Whether to use the local HST or LET method. Defaults to False.")
parser.add_argument('--k_locality', type = int, default = 2, help = "The locality of Pauli measurements to use for wasserstein compilation. Should be less than or equal to the number of qubits.")
parser.add_argument('--convergence_threshold', type = float, default = 1e-6, help = 'Threshold for convergence.')
parser.add_argument('--convergence_break', type = bool, default = False, help = 'Whether to break the training loop once the last 100 values in the training have variance less than convergence_threshold.')
parser.add_argument('--operator_cycling', type = int, default = 0, help = 'If 0 no operator cycling of Pauli operators is used. Otherwise the number of operator cycling steps.')
parser.add_argument('--optimizer', type = str, default = 'adam', help = 'Optimizer to be used for training.')
parser.add_argument('--learning_rate', type = float, default = 0.01, help = 'Learning rate for the optimizer.')
args = parser.parse_args()
parser = ConfigParser()
parser.read('config.ini')

if __name__ == '__main__':
    
    num_qubits = args.num_qubits
    depth = args.depth
    seed = args.seed
    rng = np.random.default_rng(seed)
    
    target = circuit_parser(args.target_circuit, nqubits = num_qubits, depth = depth, entanglement = args.entanglement, rng = rng, is_target = True)
    anstaz = circuit_parser(args.ansatz_circuit, nqubits = num_qubits, depth = depth, entanglement = args.entanglement, rng = rng, is_target = False)
    start_time = time.time()
    gen = Generator(ansatz=anstaz, rng = rng).compile(
        target_qc = target,
        method = args.method,
        test_states = args.test_states,
        initialization_argument = args.type_of_states,
        T = args.time_steps,
        compiler_hyper = dict(m = args.operator_cycling, k = args.k_locality,
                              s = args.num_states, N_s = args.state_pool, 
                              local = args.locality, cnvgc_threshold = args.convergence_threshold, 
                              cnvgc_break = args.convergence_break),
        optimizer_hyper = dict(optimizer = args.optimizer, lr = args.learning_rate)
    )
    print("Time taken for compilation: ", time.time() - start_time)
    print("DONE!")