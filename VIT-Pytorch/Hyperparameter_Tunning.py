import copy
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from solver import Solver
from sklearn.model_selection import train_test_split
from data_loader import get_loader
import csv


def split_dataset(args):
    full_train_loader, _ = get_loader(args)
    train_data = full_train_loader.dataset

    train_indices, val_indices = train_test_split(
        np.arange(len(train_data)), test_size=0.2, random_state=42, shuffle=True)

    train_subset = torch.utils.data.Subset(train_data, train_indices)
    val_subset = torch.utils.data.Subset(train_data, val_indices)

    train_loader = torch.utils.data.DataLoader(dataset=train_subset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_subset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             drop_last=False)

    return train_loader, val_loader


def save_metrics_csv(path, loss_list, acc_list):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'TrainLoss', 'ValAccuracy'])
        for i, (loss, acc) in enumerate(zip(loss_list, acc_list), 1):
            writer.writerow([i, loss, acc])


def tune_hyperparameters(args):
    print("Starting hyperparameter tuning...")
    best_acc = 0
    best_args = copy.deepcopy(args)

    learning_rates = [1e-4, 5e-4, 1e-3]
    embed_dims = [32, 64]
    n_layers_list = [4, 6]
    n_heads_list = [2, 4]
    forward_muls = [2]#I don't use 4 otherwise the net may overfit
    patch_sizes = [4]

    for lr in learning_rates:
        for embed_dim in embed_dims:
            for n_layers in n_layers_list:
                for n_heads in n_heads_list:
                    for forward_mul in forward_muls:
                        for patch_size in patch_sizes:
                            trial_args = copy.deepcopy(args)
                            trial_args.lr = lr
                            trial_args.embed_dim = embed_dim
                            trial_args.n_layers = n_layers
                            trial_args.n_attention_heads = n_heads
                            trial_args.forward_mul = forward_mul
                            trial_args.patch_size = patch_size

                            print(f"\nTraining with lr={lr}, embed_dim={embed_dim}, n_layers={n_layers}, n_heads={n_heads}, forward_mul={forward_mul}, patch_size={patch_size}")
                            trial_args.model_path = os.path.join(args.model_path, f"tune_lr{lr}_ed{embed_dim}_nl{n_layers}_nh{n_heads}_fm{forward_mul}_ps{patch_size}")
                            os.makedirs(trial_args.model_path, exist_ok=True)

                            trial_args.train_loader, trial_args.test_loader = split_dataset(trial_args)

                            solver = Solver(trial_args)
                            train_loss, val_acc_history = solver.train(return_metrics=True)

                            # Save temporary model and training log
                            temp_model_path = os.path.join(trial_args.model_path, 'temp_model.pt')
                            torch.save(solver.model.state_dict(), temp_model_path)
                            save_metrics_csv(os.path.join(trial_args.model_path, 'log.csv'), train_loss, val_acc_history)

                            val_acc = val_acc_history[-1] if val_acc_history else 0

                            if val_acc > best_acc:
                                best_acc = val_acc
                                best_args = copy.deepcopy(trial_args)

    print(f"\nBest hyperparameters: lr={best_args.lr}, embed_dim={best_args.embed_dim}, n_layers={best_args.n_layers}, n_heads={best_args.n_attention_heads}, forward_mul={best_args.forward_mul}, patch_size={best_args.patch_size}, acc={best_acc:.2f}%")
    return best_args


def retrain_on_full_dataset(best_args):
    best_args.epochs = 50
    print("\nRetraining final model on full dataset with best hyperparameters...")
    solver = Solver(best_args)
    train_loss, val_acc_history = solver.train(return_metrics=True)
    solver.test()

    save_p = os.path.join(best_args.model_path, 'Vit_final.pt')
    torch.save(solver.model.state_dict(), save_p)
    print(f"Final model saved to {save_p}")

    # Plot training loss and validation accuracy
    epochs = list(range(1, len(train_loss) + 1))
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(best_args.model_path, 'train_val_curve.png')
    plt.savefig(plot_path)
    print(f"Training curve saved to {plot_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dset', type=str, default='mnist')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--n_channels', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--n_attention_heads', type=int, default=4)
    parser.add_argument('--forward_mul', type=int, default=2)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--load_model', type=bool, default=False)

    args = parser.parse_args()
    args.model_path = os.path.join(args.model_path, args.dset)

    print("Using GPU" if torch.cuda.is_available() else "Using CPU")

    best_args = tune_hyperparameters(args)
    retrain_on_full_dataset(best_args)