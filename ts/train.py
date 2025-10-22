import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import LSTMAnomalyDetector
import torch.optim as optim
import os

def train(args):
    train = np.load(os.path.join(args.data_dir, 'train.npy'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X = torch.tensor(train, dtype=torch.float32)  # shape (n, seq_len, features)
    ds = TensorDataset(X, X)  # autoencoder-style reconstruction
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    input_size = X.shape[-1]
    model = LSTMAnomalyDetector(input_size, hidden_size=args.hidden_size,
                                num_layers=args.num_layers, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(loader.dataset)
        print(f"Epoch {epoch}/{args.epochs} - Loss: {epoch_loss:.6f}")

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_epoch{epoch}.pt"))

    torch.save(model.state_dict(), os.path.join(args.out_dir, "model_final.pt"))
    print("Training complete. Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--out_dir", default="models")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
