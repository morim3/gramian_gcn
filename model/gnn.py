import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from scipy.linalg import solve_continuous_lyapunov

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax as geo_softmax

from model.data_generator import generate_training_data, compute_gramian

is_debug = False


class EGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(EGATConv, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=True)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels + edge_dim))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        # nn.init.normal(self.lin.weight)
        # nn.init.normal(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr):
        num_nodes = x.size(0)

        # # Add self-loops to the adjacency matrix.
        # edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=num_nodes)

        # Linear transformation.
        x = self.lin(x).view(-1, self.heads, self.out_channels)

        # Start propagating messages.
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        # Concatenate or average multi-head results.
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if edge_attr is not None:
            edge_attr = edge_attr.unsqueeze(1).repeat(1, self.heads, 1)
            alpha = (torch.cat([x_i, x_j, edge_attr], dim=-1) * self.att).sum(dim=-1)
        else:
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att[:, :, :-self.edge_dim]).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = geo_softmax(alpha, index, ptr, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

class EGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_dim, num_layers, heads, ):
        super(EGAT, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(EGATConv(in_channels, hidden_channels, edge_dim, heads))
        for _ in range(num_layers - 2):
            self.convs.append(EGATConv(hidden_channels * heads, hidden_channels, edge_dim, heads))
        self.convs.append(EGATConv(hidden_channels * heads, hidden_channels, edge_dim, heads, concat=False))

        self.out_lin = nn.Linear(hidden_channels*heads, 1)
        self.heads = heads
        self.hidden_channels = hidden_channels


    def forward(self, x, edge_index, edge_attr, num_inputs):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index, edge_attr))

        x = self.out_lin(x).squeeze()
        input_nodes = x[-num_inputs:]
        out = torch.sigmoid(input_nodes)
        return out

def compute_loss(pred, true):
    return F.binary_cross_entropy(pred, true)

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.num_inputs)
        loss = compute_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, loader, device, n_select_inputs):
    model.eval()
    total_relative_error = 0
    percentiles = []
    total_samples = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.num_inputs)
            _, top_indices = torch.topk(out, n_select_inputs)
            predicted_G = torch.zeros_like(out)
            predicted_G[top_indices] = 1.0

            A = data.A.cpu().numpy()
            B = data.B.cpu().numpy()

            predicted_gramian = compute_gramian(A, B, np.diag(predicted_G.cpu().numpy()))
            predicted_trace = np.trace(predicted_gramian)

            optimal_G = data.y.cpu().numpy()
            optimal_gramian = compute_gramian(A, B, np.diag(optimal_G))
            optimal_trace = np.trace(optimal_gramian)

            relative_error = abs(optimal_trace - predicted_trace) / optimal_trace
            total_relative_error += relative_error

            percentile = np.searchsorted(data.dist[0], predicted_trace) / len(data.dist[0]) * 100
            percentiles.append(percentile)

            total_samples += 1

    average_relative_error = total_relative_error / total_samples
    return average_relative_error, percentiles

# def train(model, train_loader, optimizer, device):
#     model.train()
#     total_loss = torch.tensor([0.])
#     optimizer.zero_grad()
#     for data in train_loader:
#         data = data.to(device)
#         out = model(data.x, data.edge_index, data.edge_attr)
#         loss = compute_loss(out, data.y)
#         total_loss += loss
#
#     total_loss = total_loss / len(train_loader)
#     total_loss.backward()
#     optimizer.step()
#     return total_loss.item()
#

def get_parameters(debug=False):
    if debug:
        return {
            'min_states': 2,
            'max_states': 5,
            'n_inputs': 2,
            'num_samples': 10,
            'batch_size': 1,
            'hidden_channels': 16,
            'num_epochs': 5,
            'learning_rate': 0.01,
            'num_layers': 2,
            'heads': 2,
            'n_select_inputs': 1,
        }


    else:
        # Parameters
        return {
            'min_states': 5,
            'max_states': 10,
            'n_inputs': 9,
            'num_samples': 500,
            'batch_size': 1,
            'hidden_channels': 20,
            'num_epochs': 100,
            'learning_rate': 0.001,
            'num_layers': 5,
            'heads': 5,
            'n_select_inputs': 3,
            }


