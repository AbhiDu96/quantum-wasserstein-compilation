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

import torch
from torch.autograd import Function

from typing import Any
from collections.abc import Sequence

from qiskit import QuantumCircuit
from qiskit_torch_module.quantum_neural_network.qnn import QNN


class QuantumAutograd(Function):
    """ Implements torch`s autograd functionality to realize automatic differentiation of quantum neural networks.
    """

    # pylint: disable=arguments-differ
    @staticmethod
    def forward( # noqa
            ctx: Any,
            qnn: QNN,
            circuits: Sequence[QuantumCircuit],
            weights: torch.Tensor
    ) -> torch.Tensor:
        """ Realizes forward pass, saves context for backward pass

            Args:
                ctx: Context
                qnn: Quantum neural network instance
                circuits: Sequence of QuantumCircuits
                weights: trainable weights

            Returns:
                Result of forward pass

            Raises:
                ValueError: Wrong dimensionality of input
        """

        # save for backward pass
        ctx.qnn = qnn
        ctx.circuits = circuits
        ctx.save_for_backward(weights)

        # compute expectation values via forward pass
        values = qnn.forward(
            circuits=circuits,
            variational_weights=weights.detach().numpy(),
        )
        return torch.FloatTensor(values)

    # pylint: disable=arguments-differ
    @staticmethod
    def backward( # noqa
            ctx,
            grad_output: torch.Tensor,
    ) -> tuple[None, None, torch.Tensor]:
        """ Realizes forward pass, therefore restores context for forward pass

            Args:
                ctx: Context
                grad_output: Gradient output from previous layer (or from loss function if this is the last one)

            Returns:
                Gradients w.r.t. parameter sets

            Raises:
                ValueError: Wrong dimensionality of input
        """
        qnn = ctx.qnn
        weights = ctx.saved_tensors[0]

        # compute gradients
        gradients = qnn.backward(
            circuits=ctx.circuits,
            variational_weights=weights.detach().numpy()
        )
        # account for gradients from consecutive layer, i.e. compute einsum for batch `b` and output `o` (i.e. j-th
        # observable) to get the overall gradient w.r.t. parameter `p`
        gradients = torch.einsum(
            'bo,bop->p', grad_output.detach().cpu(), torch.FloatTensor(gradients)
        )

        return None, None, gradients
