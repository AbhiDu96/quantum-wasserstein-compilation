# Quantum Wasserstein Compilation of a Unitary

This repository is the code that was implemented for the experiments in the paper: ["Quantum Wasserstein Compilation: Unitary Compilation using the Quantum Earth Mover's Distance"](https://arxiv.org/abs/2409.05849)

To get started create a conda environment and install the basic packages as follows:

```python
conda create --name qml python=3.12 numpy matplotlib
```

Then install the requirements.txt file as follows:

```python
pip install -r requirements.txt
```

Then go into ```bayqs-qiskit-torch-module-multiple_circuits``` and the root directory and run the following command in both the directories one by one:

```python
pip install -e .
```

To finally test the running of the script for known examples you can just run the ```main.py``` file without changing any arguments.

```python
python main.py
```
