import numpy as np
import torch
from torch_geometric.data import Data
from scipy.linalg import solve_continuous_lyapunov

class SystemDataGenerator:

    def __init__(self, n_states, n_inputs, sparsity=0.1):
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.sparsity = sparsity


    # TODO hitaisyou
    def generate_stable_matrix(self):
        # Generate random eigenvalues with negative real parts
        real_parts = -np.abs(np.random.randn(self.n_states))  # Negative real parts
        imag_parts = np.random.randn(self.n_states)  # Random imaginary parts
        eigvals = real_parts + 1j * imag_parts

        # Generate a random orthogonal matrix for eigenvectors
        Q, _ = np.linalg.qr(np.random.randn(self.n_states, self.n_states))

        # Construct the diagonal matrix of eigenvalues
        D = np.diag(eigvals)

        # Construct the stable matrix A
        A = Q @ D @ Q.T.conj()

        # Ensure the matrix is real (imaginary part should be negligibly small)
        A = np.real(A)

        return A

    def generate_sparse_B(self):
        B = np.random.randn(self.n_states, self.n_inputs)
        B[np.random.rand(*B.shape) < self.sparsity] = 0
        return B

    @staticmethod
    def is_controllable(A, B):
        n_states = A.shape[0]
        C = np.hstack([np.linalg.matrix_power(A, i).dot(B) for i in range(n_states)])
        return np.linalg.matrix_rank(C) == n_states

    def generate_system(self):
        while True:
            A = self.generate_stable_matrix()
            B = self.generate_sparse_B()
            if self.is_controllable(A, B):
                return A, B



def compute_gramian(A, B, G):
    BG = B.dot(G)
    X = solve_continuous_lyapunov(A, -BG.dot(BG.T))
    return X

def find_optimal_G(A, B, k):
    n_inputs = B.shape[1]
    max_trace = -np.inf
    optimal_G = None

    for i in range(2**n_inputs):
        if(bin(i).count("1") != k):
            continue

        G = np.diag([int(b) for b in format(i, f'0{n_inputs}b')])
        X = compute_gramian(A, B, G)
        trace = np.trace(X)
        if trace > max_trace:
            max_trace = trace
            optimal_G = G

    return optimal_G

def create_graph_data(A, B):
    n_states, n_inputs = A.shape[0], B.shape[1]
    # adj_matrix = np.block([[A, B], [B.T, np.ones((n_inputs, n_inputs))]])
    adj_matrix = np.block([[A, B], [B.T, np.identity(n_inputs)]])
    
    edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)
    edge_attr = torch.tensor(adj_matrix[adj_matrix != 0], dtype=torch.float).view(-1, 1)

    # Improved node features
    node_features = []
    for i in range(n_states + n_inputs):
        degree = np.sum(adj_matrix[i] != 0) + np.sum(adj_matrix[:, i] != 0)
        in_degree = np.sum(adj_matrix[i] != 0)
        is_state = 1 if i < n_states else 0
        is_input = 0 if i < n_states else 1
        num_state_neighbors = sum(1 for j in range(len(adj_matrix[i])) if adj_matrix[i][j] and j < n_states)
        num_input_neighbors = sum(1 for j in range(len(adj_matrix[i])) if adj_matrix[i][j] and j > n_states)
        node_features.append([degree, in_degree, is_state, is_input, num_state_neighbors, num_input_neighbors])
    

    x = torch.tensor(node_features, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_states=n_states, num_inputs=n_inputs, A=torch.tensor(A, dtype=torch.float), B=torch.tensor(B, dtype=torch.float))

def generate_training_data(num_samples, n_states, n_inputs, n_select_inputs, normalize_A_B=True):
    generator = SystemDataGenerator(n_states, n_inputs)
    data_list = []
    for _ in range(num_samples):
        A, B = generator.generate_system()
        if normalize_A_B:
            A = A / np.std(A)
            B = B / np.std(B)
        graph_data = create_graph_data(A, B)
        optimal_G = find_optimal_G(A, B, n_select_inputs)
        graph_data.y = torch.tensor(np.diag(optimal_G), dtype=torch.float)
        data_list.append(graph_data)
    return data_list
