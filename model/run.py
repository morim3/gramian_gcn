
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader
from model.data_generator import compute_gramian, generate_training_data
from torch import optim
from model.gnn import EGAT, train, evaluate

is_debug = False

def get_best_worst_predictions(model, loader, device, n_select_inputs):
    model.eval()
    best_pred = worst_pred = None
    best_error = float('inf')
    worst_error = 0

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

            if relative_error < best_error:
                best_error = relative_error
                best_pred = (data, predicted_G, optimal_G)

            if relative_error > worst_error:
                worst_error = relative_error
                worst_pred = (data, predicted_G, optimal_G)

    return best_pred, worst_pred



def log_gramian_distributions(data_list, prefix):
    for i, data in enumerate(data_list):
        wandb.log({f"{prefix}_gramian_dist_{i}": wandb.Histogram(data.dist)})

def log_graph_statistics(data_list):
    node_counts = [data.x.size(0) for data in data_list]
    edge_counts = [data.edge_index.size(1) for data in data_list]
    avg_degrees = [2 * ec / nc for nc, ec in zip(node_counts, edge_counts)]

    wandb.log({
        "node_count_dist": wandb.Histogram(node_counts),
        "edge_count_dist": wandb.Histogram(edge_counts),
        "avg_degree_dist": wandb.Histogram(avg_degrees),
    })

def plot_prediction(pred_data, title):
    data, predicted_G, optimal_G = pred_data
    
    # Create a NetworkX graph from the data
    G = to_networkx(data, to_undirected=True)
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(title)
    
    # Plot the graph structure
    pos = nx.spring_layout(G)
    nx.draw(G, pos, ax=ax1, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
    
    # Highlight the input nodes
    input_nodes = range(data.num_states, data.num_states + data.num_inputs)
    nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_color='lightgreen', node_size=700, ax=ax1)
    
    # Plot the input selection comparison
    ax2.bar(range(data.num_inputs), optimal_G, alpha=0.5, label='Optimal')
    ax2.bar(range(data.num_inputs), predicted_G, alpha=0.5, label='Predicted')
    ax2.set_xlabel('Input Index')
    ax2.set_ylabel('Selection (0 or 1)')
    ax2.set_title('Input Selection Comparison')
    ax2.legend()
    
    # Adjust layout and return the figure
    plt.tight_layout()
    return fig


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
            'num_epochs': 1,
            'learning_rate': 0.001,
            'num_layers': 5,
            'heads': 5,
            'n_select_inputs': 3,
            }

def test_scalability(model, config, device, factors):
    metrics = []
    for factor in factors:
        large_test_data = generate_training_data(
            num_samples=100,  # サンプル数を減らして計算時間を節約
            n_inputs=config.n_inputs,
            n_select_inputs=config.n_select_inputs,
            min_states=config.max_states * factor,
            max_states=config.max_states * factor
        )
        large_test_loader = DataLoader(large_test_data, batch_size=config.batch_size)
        acc, percentiles = evaluate(model, large_test_loader, device, config.n_select_inputs)
        metrics.append({
            'factor': factor,
            'accuracy': acc,
            'percentiles': percentiles
        })
    return metrics

def log_scalability_results(metrics):
    factors = [m['factor'] for m in metrics]
    accuracies = [m['accuracy'] for m in metrics]
    
    # 精度のグラフ
    plt.figure(figsize=(10, 6))
    plt.plot(factors, accuracies, marker='o')
    plt.title('Model Accuracy vs State Space Size')
    plt.xlabel('Factor of max_states')
    plt.ylabel('Accuracy')
    plt.grid(True)
    wandb.log({"scalability_accuracy": wandb.Image(plt)})
    plt.close()

    # パーセンタイルの箱ひげ図
    plt.figure(figsize=(10, 6))
    plt.boxplot([m['percentiles'] for m in metrics], labels=factors)
    plt.title('Distribution of Percentiles vs State Space Size')
    plt.xlabel('Factor of max_states')
    plt.ylabel('Percentile')
    plt.grid(True)
    wandb.log({"scalability_percentiles": wandb.Image(plt)})
    plt.close()

    # 数値データもログに記録
    for m in metrics:
        wandb.log({
            f"scalability_factor_{m['factor']}_accuracy": m['accuracy'],
            f"scalability_factor_{m['factor']}_percentile_mean": np.mean(m['percentiles']),
            f"scalability_factor_{m['factor']}_percentile_median": np.median(m['percentiles'])
        })

def main():
    wandb.init(project="graph_input_selection", config=get_parameters(debug=is_debug))
    config = wandb.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate data
    train_data = generate_training_data(config.num_samples, config.n_inputs, config.n_select_inputs,
                                        config.min_states, config.max_states)
    test_data = generate_training_data(int(config.num_samples * 0.2), config.n_inputs, config.n_select_inputs,
                                       config.min_states, config.max_states)
    large_test_data = generate_training_data(int(config.num_samples * 0.2), config.n_inputs, config.n_select_inputs,
                                             config.max_states * 2, config.max_states * 2)

    # Log dataset statistics
    wandb.log({
        "train_data_size": len(train_data),
        "test_data_size": len(test_data),
        "large_test_data_size": len(large_test_data),
        "train_n_states_dist": wandb.Histogram([d.num_states for d in train_data]),
        "test_n_states_dist": wandb.Histogram([d.num_states for d in test_data]),
        "large_test_n_states_dist": wandb.Histogram([d.num_states for d in large_test_data])
    })

    # Log gramian distributions
    # log_gramian_distributions(train_data, "train")
    # log_gramian_distributions(test_data, "test")
    # log_gramian_distributions(large_test_data, "large_test")

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size)
    large_test_loader = DataLoader(large_test_data, batch_size=config.batch_size)

    # Initialize model
    in_channels = train_data[0].x.size(1)
    edge_dim = train_data[0].edge_attr.size(1)
    model = EGAT(in_channels=in_channels, hidden_channels=config.hidden_channels,
                 edge_dim=edge_dim, num_layers=config.num_layers, heads=config.heads).to(device)

    # Log model architecture
    wandb.watch(model, log="all")

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Log optimizer details
    wandb.log({"optimizer": optimizer.__class__.__name__,
               "learning_rate": config.learning_rate})

    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({"total_params": total_params, "trainable_params": trainable_params})

    # Training loop
    for epoch in range(config.num_epochs):
        loss = train(model, train_loader, optimizer, device)
        train_acc, train_percentiles = evaluate(model, train_loader, device, config.n_select_inputs)
        test_acc, test_percentiles = evaluate(model, test_loader, device, config.n_select_inputs)
        large_test_acc, large_test_percentiles = evaluate(model, large_test_loader, device, config.n_select_inputs)

        wandb.log({
            "epoch": epoch,
            "loss": loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "large_test_acc": large_test_acc,
            "train_percentiles": wandb.Histogram(train_percentiles),
            "test_percentiles": wandb.Histogram(test_percentiles),
            "large_test_percentiles": wandb.Histogram(large_test_percentiles),
        })

        print(f'Epoch: {epoch+1}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Test Acc: {test_acc:.4f}, Large Test Acc: {large_test_acc:.4f}')

    scalability_factors = [2]
    scalability_metrics = test_scalability(model, config, device, scalability_factors)
    log_scalability_results(scalability_metrics)

    log_graph_statistics(train_data)
    best_pred, worst_pred = get_best_worst_predictions(model, test_loader, device, config.n_select_inputs)
    wandb.log({
        "best_prediction": wandb.Image(plot_prediction(best_pred, "Best Prediction")),
        "worst_prediction": wandb.Image(plot_prediction(worst_pred, "Worst Prediction"))
    })

    wandb.finish()
    return model

if __name__ == "__main__":
    main()
