import argparse
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import NeuralTPP
from real_data import RealTPPDataset, collate_pad


def train_epoch(model, optimizer, loader, device="cpu"):
    model.train()
    total_loglik = 0.0
    total_events = 0.0
    total_grad_norm = 0.0
    num_batches = 0

    for batch_idx, (deltas, mask) in enumerate(loader):
        deltas = deltas.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        avg_loglik = model(deltas, mask)  # per-event log-likelihood

        loss = -avg_loglik
        loss.backward()

        # gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        optimizer.step()

        n_events = mask.sum().item()
        total_loglik += avg_loglik.item() * n_events
        total_events += n_events

        total_grad_norm += total_norm
        num_batches += 1

        if (batch_idx + 1) % 10 == 0:
            print(
                f"  batch {batch_idx+1}/{len(loader)}  "
                f"avg_loglik={avg_loglik.item():.4f}  grad_norm={total_norm:.4f}"
            )

    avg_loglik_per_event = total_loglik / (total_events + 1e-8)
    avg_grad_norm = total_grad_norm / max(num_batches, 1)

    return avg_loglik_per_event, avg_grad_norm


def eval_epoch(model, loader, device="cpu"):
    model.eval()
    total_loglik = 0.0
    total_events = 0.0

    for deltas, mask in loader:
        deltas = deltas.to(device)
        mask = mask.to(device)

        avg_loglik = model(deltas, mask)
        n_events = mask.sum().item()
        total_loglik += avg_loglik.item() * n_events
        total_events += n_events

    return total_loglik / (total_events + 1e-8)


def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")

    writer = SummaryWriter(log_dir=args.logdir)

    print(f"Loading preprocessed sequences from {args.data_path}")
    obj = torch.load(args.data_path, map_location="cpu")
    train_seqs = obj["train_seqs"]
    test_seqs = obj["test_seqs"]

    print(f"Num train sequences: {len(train_seqs)}")
    print(f"Num test  sequences: {len(test_seqs)}")

    train_dataset = RealTPPDataset(train_seqs)
    test_dataset = RealTPPDataset(test_seqs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pad,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_pad,
    )

    print(f"len(train_loader) = {len(train_loader)}")
    print(f"len(test_loader)  = {len(test_loader)}")

    model = NeuralTPP(
        hidden_dim=args.hidden_dim,
        hazard_hidden_dim=args.hazard_hidden_dim,
        init_hidden_scale=0.0,
        device=device,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Num parameters: {sum(p.numel() for p in model.parameters())}")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "best_sf_model.pt")
    best_val = float("-inf")

    for epoch in range(1, args.epochs + 1):
        print("Entered train_epoch")
        train_ll, grad_norm = train_epoch(model, optimizer, train_loader, device)
        val_ll = eval_epoch(model, test_loader, device)

        print(
            f"[Epoch {epoch:03d}] "
            f"train loglik/event = {train_ll:.4f} | "
            f"test loglik/event = {val_ll:.4f} | "
            f"grad_norm = {grad_norm:.4f}"
        )

        writer.add_scalar("loglik/train_per_event", train_ll, epoch)
        writer.add_scalar("loglik/test_per_event", val_ll, epoch)
        writer.add_scalar("model/grad_norm", grad_norm, epoch)

        if val_ll > best_val:
            best_val = val_ll
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_val_loglik_per_event": best_val,
                },
                ckpt_path,
            )

    print(f"Best test loglik/event = {best_val:.4f}")
    print(f"Saved best model to {ckpt_path}")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Neural TPP on SF 911 calls (real data)."
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="../data/sf_tpp_sequences.pt",
        help="Path to preprocessed .pt file produced by preprocess_sf_calls.py",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--hazard-hidden-dim", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out-dir", type=str, default="../checkpoints_sf")
    parser.add_argument(
        "--logdir",
        type=str,
        default="../runs_sf",
        help="TensorBoard log directory",
    )

    args = parser.parse_args()
    main(args)