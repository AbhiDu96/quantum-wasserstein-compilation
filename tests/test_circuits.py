import unittest
from qiskit import QuantumCircuit
from utils.circuits import circuit_parser, create_w_state, create_ghz_state, vw_block

class TestCircuits(unittest.TestCase):
    
    def test_circuit_parser_zycx(self):
        
        qc = circuit_parser("zycx", nqubits = 3, depth = 3, is_target=False)
        self.assertIsInstance(qc, QuantumCircuit)
        self.assertEqual(qc.num_qubits, 3)
        
    def test_circuit_hea(self):
        
        qc = circuit_parser("hea", nqubits = 3, depth = 3, is_target=False)
        self.assertIsInstance(qc, QuantumCircuit)
        self.assertEqual(qc.num_qubits, 3)
    
    def test_circuit_w_state(self):
        
        qc = create_w_state(3)
        self.assertIsInstance(qc, QuantumCircuit)
        self.assertEqual(qc.num_qubits, 3)
        
    def test_circuit_ghz_state(self):
        
        qc = create_ghz_state(3)
        self.assertIsInstance(qc, QuantumCircuit)
        self.assertEqual(qc.num_qubits, 3)
        
    def test_vw_block(self):
        
        qc = QuantumCircuit(2).compose(vw_block())
        self.assertIsInstance(qc, QuantumCircuit)
        self.assertEqual(qc.num_qubits, 2)
    
        
    def test_hea_cnot_count(self):
        
        num_qubits = 3
        repetitions = 3
        entanglement = ['circular', 'linear', 'full']
        for ent in entanglement:
            qc = circuit_parser("hea", nqubits = num_qubits, depth = repetitions, entanglement = ent, is_target=False).decompose()
            if ent == 'linear':
                cnot_count = (num_qubits - 1) * repetitions
                self.assertEqual(qc.count_ops()['cx'], cnot_count)
            elif ent == 'circular':
                cnot_count = num_qubits * repetitions
                self.assertEqual(qc.count_ops()['cx'], cnot_count)
            elif ent == 'full':
                cnot_count = (num_qubits * (num_qubits - 1) // 2) * repetitions
                self.assertEqual(qc.count_ops()['cx'], cnot_count)
        

if __name__ == '__main__':
    unittest.main()