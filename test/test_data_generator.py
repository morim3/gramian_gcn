import unittest
import numpy as np
import torch
from model.data_generator import SystemDataGenerator, solve_continuous_lyapunov, compute_gramian, find_optimal_G, create_graph_data, generate_training_data

class TestDataGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = SystemDataGenerator(n_states=5, n_inputs=2, sparsity=0.5)

    def test_generate_sparse_stable_matrix(self):
        A = self.generator.generate_stable_matrix()
        self.assertEqual(A.shape, (5, 5))
        self.assertTrue(np.all(np.real(np.linalg.eigvals(A)) < 0))

    def test_generate_sparse_B(self):
        B = self.generator.generate_sparse_B()
        self.assertEqual(B.shape, (5, 2))

    def test_is_controllable(self):
        A = np.array([[1, 0], [0, 2]])
        B = np.array([[1], [1]])
        self.assertTrue(self.generator.is_controllable(A, B))

    def test_generate_system(self):
        A, B = self.generator.generate_system()
        self.assertEqual(A.shape, (5, 5))
        self.assertEqual(B.shape, (5, 2))
        self.assertTrue(self.generator.is_controllable(A, B))

    def test_solve_continuous_lyapunov(self):
        A = np.array([[1, 2], [3, 4]])
        Q = np.array([[5, 6], [7, 8]])
        X = solve_continuous_lyapunov(A, Q)
        self.assertTrue(np.allclose(A.dot(X) + X.dot(A.T) + Q, np.zeros_like(Q)))

    def test_compute_gramian(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[1], [1]])
        G = np.array([[1]])
        X = compute_gramian(A, B, G)
        self.assertTrue(np.allclose(A.dot(X) + X.dot(A.T) + B.dot(G).dot(G.T).dot(B.T), np.zeros_like(X)))

    def test_find_optimal_G(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[1, 0], [0, 1]])
        G = find_optimal_G(A, B)
        self.assertEqual(G.shape, (2, 2))

    def test_create_graph_data(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[1], [1]])
        data = create_graph_data(A, B)
        # self.assertEqual(data.num_nodes, 3)
        # self.assertEqual(data.num_edges, 5)

    def test_generate_training_data(self):
        data_list = generate_training_data(num_samples=10, n_states=5, n_inputs=2)
        self.assertEqual(len(data_list), 10)
        self.assertEqual(data_list[0].num_nodes, 7)
        self.assertEqual(data_list[0].y.shape, (2,))

if __name__ == '__main__':
    unittest.main()
