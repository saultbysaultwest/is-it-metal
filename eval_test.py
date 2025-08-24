# eval_test.py

import argparse, torch, numpy as np
from torch.utils.data import DataLoader
from cached_dataset import CachedMFCCDataset
from metal_classifier_optimized import MetalClassifier

# optional metrics (falls back to accuracy-only if not installed)
try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
    SK = True
except Exception:
    SK = False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="best_model.pt")
    ap.add_argument("--root", default="mfcc_cache/testset")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CachedMFCCDataset(args.root)
    dl = DataLoader(ds, batch_size=args.batch)

    model = MetalClassifier().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    all_logits, all_y = [], []
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze(1)
            all_logits.append(logits.cpu())
            all_y.append(y.cpu())

    logits = torch.cat(all_logits); y = torch.cat(all_y)
    probs = torch.sigmoid(logits).numpy()
    preds = (probs > args.threshold).astype(np.float32)
    y_np = y.numpy().astype(np.float32)

    # always print accuracy; add more if sklearn is available
    acc = (preds == y_np).mean()
    print(f"Test samples: {len(ds)}")
    print(f"Accuracy: {acc:.4f}")

    if SK:
        prec, rec, f1, _ = precision_recall_fscore_support(y_np, preds, average="binary", zero_division=0)
        try: auc = roc_auc_score(y_np, probs)
        except Exception: auc = float("nan")
        cm = confusion_matrix(y_np, preds).astype(int)
        print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
        print("Confusion matrix [[TN FP],[FN TP]]:\n", cm)

if __name__ == "__main__":
    main()
