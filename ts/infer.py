import numpy as np
import torch
from ts.model import LSTMAnomalyDetector
import joblib
import os

def infer_sequence(model_path, seq_array):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = seq_array.shape[-1]
    model = LSTMAnomalyDetector(input_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = torch.tensor(seq_array[None], dtype=torch.float32).to(device)
        preds = model(x)
        err = ((preds - x)**2).mean().item()
    return err

if __name__ == "__main__":
    # example usage
    print("Use the CLI scripts or import functions for programmatic inference.")
