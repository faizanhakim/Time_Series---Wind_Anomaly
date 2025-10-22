import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import os

def compute_reconstruction_errors(model, data, device):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data, dtype=torch.float32).to(device)
        preds = model(x)
        errors = ((preds - x)**2).mean(dim=(1,2)).cpu().numpy()
    return errors

def evaluate_model(model_path, test_path, threshold=None):
    import torch
    from model import LSTMAnomalyDetector
    test = np.load(test_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = test.shape[-1]
    model = LSTMAnomalyDetector(input_size)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    errors = compute_reconstruction_errors(model, test, device)
    # If threshold not provided, pick mean + 3*std
    if threshold is None:
        threshold = errors.mean() + 3 * errors.std()
    preds = (errors > threshold).astype(int)
    # NOTE: without labels, we can't compute precision/recall; just report anomaly stats
    print(f"Errors - mean: {errors.mean():.6f}, std: {errors.std():.6f}")
    print(f"Threshold: {threshold:.6f} - Anomalies detected: {preds.sum()} / {len(preds)}")
    return errors, threshold

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test", required=True)
    args = parser.parse_args()
    evaluate_model(args.model, args.test)
