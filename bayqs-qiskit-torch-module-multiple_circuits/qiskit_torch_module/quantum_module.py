# This code is part of Qiskit-Torch-Module[sequential].
# Author: Nico Meyer (nico.meyer@iis.fraunhofer.de)
#
# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
from collections.abc import Sequence

import torch
import torch.nn as nn

from qiskit import QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qiskit_torch_module.quantum_neural_network.qnn import QNN
from qiskit_torch_module.quantum_autograd import QuantumAutograd


class QuantumModule(nn.Module):
    """ This class implements a quantum torch-module, based on an underlying quantum neural network

    """

    def __init__(
            self,
            observables: Sequence[BaseOperator] | BaseOperator,
            num_weights: int = None,
            weights_initial: Sequence[float] = None
    ):
        super(QuantumModule, self).__init__()

        if num_weights is None and weights_initial is None:
            raise ValueError('Either the number of trainable weights, or explicit initial values have to be provided.')

        if num_weights is not None and weights_initial is not None:
            if num_weights != weights_initial:
                raise ValueError('Inconsistent number of weights and number of provided initial weights.')

        if num_weights is None:
            num_weights = len(weights_initial)

        # set up pytorch parameters
        if weights_initial is None:
            weights = nn.Parameter(nn.init.uniform_(torch.empty(num_weights), a=0.0, b=2*np.pi))
        else:
            weights = nn.Parameter(torch.tensor(weights_initial))
        self.register_parameter('weights', weights)

        # set up quantum neural network
        self.qnn = QNN(
            observables=observables
        )

    def forward(
            self,
            circuits: Sequence[QuantumCircuit]
    ) -> torch.Tensor:
        """ Calls into QuantumAutograd (instance of torch`s autograd functionality) to compute forward pass and
        constructs tree for backward pass.

            Args:
                circuits: Sequence of Quantum Circuits

            Returns:
                Result of forward pass
        """

        return QuantumAutograd.apply(self.qnn, circuits, self.weights)
