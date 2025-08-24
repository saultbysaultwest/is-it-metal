# train_with_metrics.py

import os, json, csv, math, argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from cached_dataset import CachedMFCCDataset
from metal_classifier_optimized import MetalClassifier

# Optional but recommended for metrics/plots
try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        roc_auc_score, roc_curve, confusion_matrix
    )
    SK_OK = True
except Exception:
    SK_OK = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_class_weights(dataset):
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    metal = sum(labels); non = len(labels) - metal
    pos_weight = (non / max(metal, 1))
    print(f"Dataset composition: {metal} metal, {non} non-metal")
    print(f"Calculated pos_weight: {pos_weight:.3f}")
    return pos_weight

@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    all_logits, all_labels = [], []
    val_loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x).squeeze(1)
        val_loss_sum += loss_fn(logits, y).item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())
    logits = torch.cat(all_logits); y = torch.cat(all_labels)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    # Default metrics (no sklearn needed)
    acc = (preds.eq(y)).float().mean().item()
    prec = rec = f1 = auc = None

    if SK_OK:
        y_np = y.numpy()
        p_np = preds.numpy()
        pr_np = probs.numpy()
        acc = float(accuracy_score(y_np, p_np))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_np, p_np, average='binary', zero_division=0
        )
        try:
            auc = float(roc_auc_score(y_np, pr_np))
        except ValueError:
            auc = None

    return {
        "val_loss": val_loss_sum / max(len(loader), 1),
        "acc": acc,
        "precision": None if prec is None else float(prec),
        "recall": None if rec is None else float(rec),
        "f1": None if f1 is None else float(f1),
        "auc": auc,
        "logits": logits.numpy(),
        "labels": y.numpy(),
        "probs": probs.numpy()
    }

def save_learning_curves(run_dir, history):
    if not SK_OK: return
    epochs = np.arange(1, len(history)+1)
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"] for h in history]
    val_acc    = [h["acc"] for h in history]

    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.title("Learning Curves")
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "learning_curves.png")); plt.close()

    plt.figure()
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend()
    plt.title("Validation Accuracy")
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "val_accuracy.png")); plt.close()

def save_roc_confusion(run_dir, labels, probs):
    if not SK_OK: return
    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
        plt.tight_layout(); plt.savefig(os.path.join(run_dir, "roc_curve.png")); plt.close()
    except Exception:
        pass

    cm = confusion_matrix(labels, (probs>0.5).astype(int))
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix"); plt.colorbar()
    plt.xticks([0,1], ["non_metal","metal"]); plt.yticks([0,1], ["non_metal","metal"])
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "confusion_matrix.png")); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="baseline", help="folder under runs/ for artifacts")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--val_batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--step_size", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.6)
    ap.add_argument("--train_root", default="mfcc_cache/dataset")
    ap.add_argument("--val_root", default="mfcc_cache/valset")
    args = ap.parse_args()

    run_dir = os.path.join("runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Datasets/Loaders
    train_ds = CachedMFCCDataset(args.train_root)
    val_ds   = CachedMFCCDataset(args.val_root)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.val_batch)

    # Model / Loss / Optim
    model = MetalClassifier().to(device)
    pos_weight = calculate_class_weights(train_ds)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)

    # Logging headers
    csv_path = os.path.join(run_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch","train_loss","val_loss","acc","precision","recall","f1","auc","lr"])

    best_val_acc = -1.0
    history = []

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze(1)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        train_loss = total_loss / max(len(train_loader), 1)
        eval_out = evaluate(model, val_loader, loss_fn)

        # Step scheduler
        scheduler.step()

        # Store history & CSV row
        hist_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": eval_out["val_loss"],
            "acc": eval_out["acc"]
        } | {k: eval_out[k] for k in ["precision","recall","f1","auc"]}
        history.append(hist_row)

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                epoch, f"{train_loss:.6f}", f"{eval_out['val_loss']:.6f}",
                f"{eval_out['acc']:.6f}",
                "" if eval_out["precision"] is None else f"{eval_out['precision']:.6f}",
                "" if eval_out["recall"]    is None else f"{eval_out['recall']:.6f}",
                "" if eval_out["f1"]        is None else f"{eval_out['f1']:.6f}",
                "" if eval_out["auc"]       is None else f"{eval_out['auc']:.6f}",
                f"{opt.param_groups[0]['lr']:.6e}"
            ])

        # Save best
        if eval_out["acc"] > best_val_acc:
            best_val_acc = eval_out["acc"]
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
            # keep last eval preds/labels for plots
            np.save(os.path.join(run_dir, "val_labels.npy"), eval_out["labels"])
            np.save(os.path.join(run_dir, "val_probs.npy"),  eval_out["probs"])

        print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} | "
              f"val_loss {eval_out['val_loss']:.4f} | acc {eval_out['acc']:.3f} | "
              f"lr {opt.param_groups[0]['lr']:.2e}")

    # Save curves + plots
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    save_learning_curves(run_dir, history)
    if os.path.exists(os.path.join(run_dir, "val_labels.npy")):
        labels = np.load(os.path.join(run_dir, "val_labels.npy"))
        probs  = np.load(os.path.join(run_dir, "val_probs.npy"))
        save_roc_confusion(run_dir, labels, probs)

    # Also save final weights for reference
    torch.save(model.state_dict(), os.path.join(run_dir, "final_model.pt"))
    print(f"\nArtifacts saved to: {run_dir}")

if __name__ == "__main__":
    main()
