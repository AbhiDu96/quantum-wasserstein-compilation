import unittest
import numpy as np
from utils.paulis import solve_Hmax, init_full_paulis

class TestPaulis(unittest.TestCase):
    
    def test_solve_Hmax(self):
        
        paulis = init_full_paulis(num_qubits = 3, k_local = 2)
        expVals = np.random.rand(len(paulis))
        
        weights = solve_Hmax(paulis, expVals)
        
        self.assertIsInstance(weights, np.ndarray)
        
        self.assertEqual(len(weights), len(paulis))
        
if __name__ == '__main__':
    unittest.main()