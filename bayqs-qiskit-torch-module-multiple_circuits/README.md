# Qiskit Torch Module



## Getting started

To install the tool just follow this steps:

```
conda create --name qtm python=3.10.13
conda activate qtm
cd qiskit-torch-module
pip install -e .
```

## Usage

For 5-qubits, 100 observables, and a batch-size of 32 this implementation is about **50 times** faster.

To test this run
```
python examples/test.py
```

**Note:** In principle it should be possible to parallelize ove the batch (i.e. circuit) dimension to get another speed-up
factor of maybe 10-20 (depending on number of threads of used CPU). However, this would require serious additional implementation
overhead, as e.g. the QuantumCircuit objects have to be serialized.
Therefore, I think this is not worth it, and if parallelization should be used then it should be done at the experiment level,
i.e. by starting conducting multiple experiments in parallel by example using a bathc script.
