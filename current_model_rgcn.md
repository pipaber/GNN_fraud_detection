# Current Fraud Model with RGCN (Notebook-Style Cells, Leak-Safe Temporal Context)

## 1) Imports
```python
import os
import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
)

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import RGCNConv
import torch_geometric.transforms as T
```

## 2) Config
```python
TX_PATH = r"C:\Users\LENOVO\Documents\machine_learning_utec\Dataset_P2\train_transaction.csv"
ID_PATH = r"C:\Users\LENOVO\Documents\machine_learning_utec\Dataset_P2\train_identity.csv"
FEAT_PATH = r"C:\Users\LENOVO\Documents\machine_learning_utec\Dataset_P2\data_with_hdbscan_feat.parquet"

OUTPUT_DIR = r"C:\Users\LENOVO\Documents\machine_learning_utec\Dataset_P2"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots_rgcn")
os.makedirs(PLOT_DIR, exist_ok=True)

id_col, y_col, t_col = "TransactionID", "isFraud", "TransactionDT"

rel_tx_cols = ["card1", "card4", "card6", "addr1", "P_email_bin", "R_email_bin"]
rel_id_cols = ["DeviceType", "DeviceInfo", "id_30", "id_31", "id_33"]

min_freq_by_col = {
    "DeviceInfo": 300,
    "P_email_bin": 1,
    "R_email_bin": 1,
}
default_min_freq = 100

SEED = 42
HIDDEN_CHANNELS = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
EARLY_STOP_PATIENCE = 8
EARLY_STOP_MIN_DELTA = 0.002
THRESHOLD = 0.5
TOPK_LIST = [100, 500, 1000, 5000]

TRAIN_BS = 1024
EVAL_BS = 2048
NUM_NEIGHBORS = [15, 10]
USE_UNDIRECTED_CONTEXT = True
NUM_BASES = 16
```

## 3) Reproducibility
```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
```

## 4) Data Helpers (Same Leak-Safe Temporal Build)
```python
def load_feat_with_transaction_id(feat_path, tx_path, id_col="TransactionID"):
    feat = pd.read_parquet(feat_path)
    if id_col in feat.columns:
        return feat
    if isinstance(feat.index, pd.RangeIndex):
        raise ValueError(
            "Parquet has no TransactionID and RangeIndex only. "
            "Cannot recover row linkage; re-export features with TransactionID."
        )
    tx_ids = pd.read_csv(tx_path, usecols=[id_col])[id_col].reset_index(drop=True)
    feat = feat.reset_index().rename(columns={"index": "orig_idx"})
    if not np.issubdtype(feat["orig_idx"].dtype, np.integer):
        raise ValueError("Recovered index is not integer; cannot map to TransactionID.")
    if feat["orig_idx"].max() >= len(tx_ids) or feat["orig_idx"].min() < 0:
        raise ValueError("Recovered index is out of bounds for train_transaction rows.")
    feat[id_col] = tx_ids.iloc[feat["orig_idx"].to_numpy()].to_numpy()
    return feat.drop(columns=["orig_idx"])


def map_email_to_bin(email_series):
    mapping = {
        "gmail.com": "google", "gmail": "google",
        "hotmail.com": "microsoft", "outlook.com": "microsoft",
        "msn.com": "microsoft", "live.com.mx": "microsoft",
        "yahoo.com": "yahoo", "yahoo.com.mx": "yahoo",
        "yahoo.fr": "yahoo", "ymail.com": "yahoo",
        "anonymous.com": "anonymous",
        "protonmail.com": "encrypted",
    }
    main = {"google", "microsoft", "yahoo", "anonymous", "encrypted"}
    s = email_series.astype("string").fillna("__MISSING__").replace(mapping)
    return s.where(s.isin(main.union({"__MISSING__"})), "other")


def build_relation_mappings(df, relation_cols, min_freq_by_col, default_min_freq=100):
    mappings = {}
    for col in relation_cols:
        if col not in df.columns:
            continue
        min_freq = min_freq_by_col.get(col, default_min_freq)
        s = df[col].astype("string").fillna("__MISSING__")
        vc = s.value_counts(dropna=False)
        frequent = set(vc[vc >= min_freq].index.tolist()) | {"__MISSING__"}
        kept = sorted(v for v in frequent if v != "__MISSING__")
        values = ["__MISSING__"] + kept + ["__RARE__"]
        values = list(dict.fromkeys(values))
        mappings[col] = {"mapping": {v: i for i, v in enumerate(values)}, "frequent_values": frequent, "num_nodes": len(values)}
    return mappings


def encode_relation_values(series, mapping_info):
    s = series.astype("string").fillna("__MISSING__")
    keep = set(mapping_info["frequent_values"]) | {"__MISSING__"}
    s = s.where(s.isin(keep), "__RARE__")
    return s.map(mapping_info["mapping"]).fillna(mapping_info["mapping"]["__RARE__"]).astype(np.int64).to_numpy()


def add_relation_edges_with_mapping(data_obj, df, col, mapping_info):
    node_type = f"{col}_value"
    rel_type = f"has_{col}"
    data_obj[node_type].num_nodes = int(mapping_info["num_nodes"])
    src = torch.arange(len(df), dtype=torch.long)
    dst = torch.from_numpy(encode_relation_values(df[col], mapping_info))
    data_obj["transaction", rel_type, node_type].edge_index = torch.stack([src, dst], dim=0)


def prepare_base_dataframe(
    tx_path, id_path, feat_path, id_col, y_col, t_col,
    rel_tx_cols, rel_id_cols, min_freq_by_col, default_min_freq=100
):
    tx_header = pd.read_csv(tx_path, nrows=0).columns
    id_header = pd.read_csv(id_path, nrows=0).columns

    needed_raw_email_cols = [c for c in ["P_emaildomain", "R_emaildomain"] if c in tx_header]
    tx_usecols = [id_col, y_col, t_col] + [c for c in rel_tx_cols if c in tx_header] + needed_raw_email_cols
    tx_usecols = list(dict.fromkeys(tx_usecols))
    id_usecols = [id_col] + [c for c in rel_id_cols if c in id_header]

    tx = pd.read_csv(tx_path, usecols=tx_usecols)
    ident = pd.read_csv(id_path, usecols=id_usecols)
    raw = tx.merge(ident, on=id_col, how="left")

    if "P_emaildomain" in raw.columns:
        raw["P_email_bin"] = map_email_to_bin(raw["P_emaildomain"])
    if "R_emaildomain" in raw.columns:
        raw["R_email_bin"] = map_email_to_bin(raw["R_emaildomain"])

    feat = load_feat_with_transaction_id(feat_path, tx_path, id_col=id_col)
    feat = feat.drop(columns=[c for c in [y_col, t_col] if c in feat.columns], errors="ignore")

    orig_feat_cols = [c for c in feat.columns if c != id_col]
    rename_map = {c: f"feat__{c}" for c in orig_feat_cols}
    feat = feat.rename(columns=rename_map)
    feat_cols = [rename_map[c] for c in orig_feat_cols]

    df = (
        raw.merge(feat[[id_col] + feat_cols], on=id_col, how="inner")
           .sort_values([t_col, id_col])
           .reset_index(drop=True)
    )

    relation_cols = [c for c in (rel_tx_cols + rel_id_cols) if c in df.columns]
    relation_mappings = build_relation_mappings(df, relation_cols, min_freq_by_col, default_min_freq)

    t = df[t_col].to_numpy()
    q70, q85 = np.quantile(t, [0.70, 0.85])
    return df, feat_cols, relation_cols, relation_mappings, (q70, q85, t.max())


def build_context_graph(
    df_all, feat_cols, relation_cols, relation_mappings,
    t_col, id_col, y_col, context_end_time,
    seed_low_time, seed_high_time, seed_mask_name,
    use_undirected=True,
):
    ctx = (
        df_all[df_all[t_col] <= context_end_time]
        .copy()
        .sort_values([t_col, id_col])
        .reset_index(drop=True)
    )

    X_df = (
        ctx[feat_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    data = HeteroData()
    data["transaction"].x = torch.from_numpy(X_df.to_numpy(np.float32))
    data["transaction"].y = torch.from_numpy(ctx[y_col].astype(np.int64).to_numpy())
    data["transaction"].transaction_id = torch.from_numpy(ctx[id_col].astype(np.int64).to_numpy())
    data["transaction"].timestamp = torch.from_numpy(ctx[t_col].to_numpy())

    seed_mask = (ctx[t_col] > seed_low_time) & (ctx[t_col] <= seed_high_time)
    data["transaction"][seed_mask_name] = torch.from_numpy(seed_mask.to_numpy())
    data["transaction"].seed_mask = torch.from_numpy(seed_mask.to_numpy())

    for col in relation_cols:
        add_relation_edges_with_mapping(data, ctx, col, relation_mappings[col])

    if use_undirected:
        data = T.ToUndirected()(data)
    return data, ctx
```

## 5) Build Temporal Context Graphs (Leak-Safe)
```python
df_all, feat_cols, relation_cols, relation_mappings, (q70, q85, max_t) = prepare_base_dataframe(
    tx_path=TX_PATH,
    id_path=ID_PATH,
    feat_path=FEAT_PATH,
    id_col=id_col,
    y_col=y_col,
    t_col=t_col,
    rel_tx_cols=rel_tx_cols,
    rel_id_cols=rel_id_cols,
    min_freq_by_col=min_freq_by_col,
    default_min_freq=default_min_freq,
)

train_data, train_df = build_context_graph(
    df_all, feat_cols, relation_cols, relation_mappings, t_col, id_col, y_col,
    context_end_time=q70, seed_low_time=-np.inf, seed_high_time=q70,
    seed_mask_name="train_mask", use_undirected=USE_UNDIRECTED_CONTEXT
)
val_data, val_df = build_context_graph(
    df_all, feat_cols, relation_cols, relation_mappings, t_col, id_col, y_col,
    context_end_time=q85, seed_low_time=q70, seed_high_time=q85,
    seed_mask_name="val_mask", use_undirected=USE_UNDIRECTED_CONTEXT
)
test_data, test_df = build_context_graph(
    df_all, feat_cols, relation_cols, relation_mappings, t_col, id_col, y_col,
    context_end_time=max_t, seed_low_time=q85, seed_high_time=np.inf,
    seed_mask_name="test_mask", use_undirected=USE_UNDIRECTED_CONTEXT
)

print("Train graph:", train_data)
print("Val graph:", val_data)
print("Test graph:", test_data)
```

## 6) Convert Hetero Graphs to Homogeneous for RGCN
```python
def hetero_to_rgcn_homo(data_obj, split_mask_name, tx_feat_dim):
    data_tmp = copy.deepcopy(data_obj)

    # Ensure all node types have x/y/seed_mask so to_homogeneous can concatenate safely.
    for nt in data_tmp.node_types:
        if nt == "transaction":
            continue
        n = data_tmp[nt].num_nodes
        data_tmp[nt].x = torch.zeros((n, tx_feat_dim), dtype=torch.float32)
        data_tmp[nt].y = -torch.ones(n, dtype=torch.long)
        data_tmp[nt].seed_mask = torch.zeros(n, dtype=torch.bool)

    data_tmp["transaction"].seed_mask = data_tmp["transaction"][split_mask_name]

    homo = data_tmp.to_homogeneous(node_attrs=["x", "y", "seed_mask"])

    tx_type_id = data_tmp.node_types.index("transaction")
    tx_mask = homo.node_type == tx_type_id
    seed_idx = (homo.seed_mask & tx_mask).nonzero(as_tuple=True)[0]
    return homo, seed_idx


tx_feat_dim = train_data["transaction"].x.size(-1)
train_homo, train_idx = hetero_to_rgcn_homo(train_data, "train_mask", tx_feat_dim)
val_homo, val_idx = hetero_to_rgcn_homo(val_data, "val_mask", tx_feat_dim)
test_homo, test_idx = hetero_to_rgcn_homo(test_data, "test_mask", tx_feat_dim)

print(train_homo)
print("train seeds:", train_idx.numel(), "val seeds:", val_idx.numel(), "test seeds:", test_idx.numel())
```

## 7) Build Neighbor Loaders
```python
train_loader = NeighborLoader(
    train_homo,
    input_nodes=train_idx,
    num_neighbors=NUM_NEIGHBORS,
    batch_size=TRAIN_BS,
    shuffle=True,
)
val_loader = NeighborLoader(
    val_homo,
    input_nodes=val_idx,
    num_neighbors=NUM_NEIGHBORS,
    batch_size=EVAL_BS,
    shuffle=False,
)
test_loader = NeighborLoader(
    test_homo,
    input_nodes=test_idx,
    num_neighbors=NUM_NEIGHBORS,
    batch_size=EVAL_BS,
    shuffle=False,
)
```

## 8) Define RGCN Model
```python
class FraudRGCN(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, num_layers,
        num_relations, num_node_types, dropout=0.2, num_bases=None
    ):
        super().__init__()
        self.dropout = dropout

        self.x_encoder = nn.Linear(in_channels, hidden_channels)
        self.node_type_emb = nn.Embedding(num_node_types, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                RGCNConv(
                    hidden_channels,
                    hidden_channels,
                    num_relations=num_relations,
                    num_bases=num_bases,
                    aggr="mean",
                )
            )

        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, batch):
        h = self.x_encoder(batch.x.float()) + self.node_type_emb(batch.node_type)
        for conv in self.convs:
            h = conv(h, batch.edge_index, batch.edge_type)
            h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
        logits = self.classifier(h).squeeze(-1)
        return logits
```

## 9) Metrics + Plot Helpers
```python
def metrics_from_probs(y_true, probs, threshold=0.5):
    pred = (probs >= threshold).astype(np.int64)
    out = {
        "acc": (pred == y_true).mean(),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }
    if len(np.unique(y_true)) < 2:
        out["auc"], out["ap"] = float("nan"), float("nan")
    else:
        out["auc"] = roc_auc_score(y_true, probs)
        out["ap"] = average_precision_score(y_true, probs)
    return out


def ranking_metrics_at_k(y_true, probs, k_list=(100, 500, 1000, 5000)):
    order = np.argsort(-probs)
    y_sorted = y_true[order]
    total_pos = int(y_true.sum())
    rows = []
    for k in k_list:
        k = min(int(k), len(y_true))
        hits = int(y_sorted[:k].sum())
        rows.append({
            "k": k,
            "hits": hits,
            "recall_at_k": hits / total_pos if total_pos > 0 else np.nan,
            "precision_at_k": hits / k if k > 0 else np.nan,
        })
    return pd.DataFrame(rows)


@torch.no_grad()
def collect_probs_and_labels(model, loader, device):
    model.eval()
    probs_all, y_all = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        bs = batch.batch_size
        probs_all.append(torch.sigmoid(logits[:bs]).cpu().numpy())
        y_all.append(batch.y[:bs].cpu().numpy())
    return np.concatenate(y_all), np.concatenate(probs_all)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, return_raw=False):
    y_true, probs = collect_probs_and_labels(model, loader, device)
    out = metrics_from_probs(y_true, probs, threshold)
    if return_raw:
        out["y_true"], out["probs"] = y_true, probs
    return out


def plot_roc_curve(y_true, probs, title, save_path=None):
    if len(np.unique(y_true)) < 2:
        print(f"Skipping ROC for {title} (single class).")
        return
    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc(fpr, tpr):.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.title(title); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.grid(alpha=0.2)
    if save_path is not None: plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(y_true, probs, threshold=0.5, title="Confusion Matrix", save_path=None):
    pred = (probs >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{title} @ threshold={threshold:.2f}")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks([0, 1], ["0", "1"]); plt.yticks([0, 1], ["0", "1"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.colorbar(); plt.tight_layout()
    if save_path is not None: plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.show()


def plot_training_curves(history, save_dir=None):
    plt.figure(figsize=(7, 5))
    plt.plot(history["epoch"], history["train_loss"], marker="o", label="Train Loss")
    plt.title("RGCN Training Loss Curve"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.grid(alpha=0.2); plt.legend()
    if save_dir is not None: plt.savefig(os.path.join(save_dir, "loss_curve_rgcn.png"), dpi=140, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["val_auc"], label="Val AUC")
    plt.plot(history["epoch"], history["val_ap"], label="Val AP")
    plt.plot(history["epoch"], history["val_recall"], label="Val Recall")
    plt.plot(history["epoch"], history["val_precision"], label="Val Precision")
    plt.plot(history["epoch"], history["val_f1"], label="Val F1")
    plt.title("RGCN Validation Metrics by Epoch"); plt.xlabel("Epoch"); plt.ylabel("Score")
    plt.grid(alpha=0.2); plt.legend()
    if save_dir is not None: plt.savefig(os.path.join(save_dir, "val_metrics_curve_rgcn.png"), dpi=140, bbox_inches="tight")
    plt.show()
```

## 10) Initialize Training Objects
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FraudRGCN(
    in_channels=train_homo.x.size(-1),
    hidden_channels=HIDDEN_CHANNELS,
    num_layers=NUM_LAYERS,
    num_relations=int(train_homo.edge_type.max().item()) + 1,
    num_node_types=int(train_homo.node_type.max().item()) + 1,
    dropout=DROPOUT,
    num_bases=NUM_BASES,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

y_train = train_homo.y[train_idx]
pos = (y_train == 1).sum().item()
neg = (y_train == 0).sum().item()
pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

## 11) Train Loop (with ROC/CM every 10 epochs + early stopping)
```python
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_examples = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        bs = batch.batch_size
        y = batch.y[:bs].float()
        loss = criterion(logits[:bs], y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * bs
        total_examples += bs
    return total_loss / max(total_examples, 1)


history = {"epoch": [], "train_loss": [], "val_auc": [], "val_ap": [], "val_recall": [], "val_precision": [], "val_f1": []}
best_state, best_val_ap, epochs_without_improve = None, -1.0, 0

for epoch in range(1, EPOCHS + 1):
    loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    need_raw = (epoch % 10 == 0)
    val = evaluate(model, val_loader, device, threshold=THRESHOLD, return_raw=need_raw)

    history["epoch"].append(epoch)
    history["train_loss"].append(loss)
    history["val_auc"].append(val["auc"])
    history["val_ap"].append(val["ap"])
    history["val_recall"].append(val["recall"])
    history["val_precision"].append(val["precision"])
    history["val_f1"].append(val["f1"])

    print(
        f"Epoch {epoch:03d} | Loss {loss:.4f} | "
        f"Val AUC {val['auc']:.4f} | Val AP {val['ap']:.4f} | "
        f"Val Recall {val['recall']:.4f} | Val Precision {val['precision']:.4f} | Val F1 {val['f1']:.4f}"
    )

    improved = np.isfinite(val["ap"]) and (val["ap"] > best_val_ap + EARLY_STOP_MIN_DELTA)
    if improved:
        best_val_ap = val["ap"]
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improve = 0
    else:
        epochs_without_improve += 1

    if need_raw:
        plot_roc_curve(
            val["y_true"], val["probs"],
            title=f"RGCN Validation ROC Curve - Epoch {epoch}",
            save_path=os.path.join(PLOT_DIR, f"val_roc_epoch_{epoch:03d}.png"),
        )
        plot_confusion_matrix(
            val["y_true"], val["probs"], threshold=THRESHOLD,
            title=f"RGCN Validation Confusion Matrix - Epoch {epoch}",
            save_path=os.path.join(PLOT_DIR, f"val_confusion_epoch_{epoch:03d}.png"),
        )
        print(f"RGCN validation ranking metrics @ epoch {epoch}:")
        display(ranking_metrics_at_k(val["y_true"], val["probs"], k_list=TOPK_LIST))

    if epochs_without_improve >= EARLY_STOP_PATIENCE:
        print(
            f"Early stopping at epoch {epoch:03d}: "
            f"no Val AP improvement > {EARLY_STOP_MIN_DELTA} for {EARLY_STOP_PATIENCE} epochs."
        )
        break

history_df = pd.DataFrame(history)
history_csv = os.path.join(OUTPUT_DIR, "training_history_rgcn.csv")
history_df.to_csv(history_csv, index=False)
print(f"Saved training history: {history_csv}")

if best_state is not None:
    model.load_state_dict(best_state)
    print(f"Loaded best model by Val AP = {best_val_ap:.4f}")
```

## 12) Final Test + Curves
```python
test = evaluate(model, test_loader, device, threshold=THRESHOLD, return_raw=True)
print(
    f"\nTest AUC {test['auc']:.4f} | AP {test['ap']:.4f} | "
    f"Recall {test['recall']:.4f} | Precision {test['precision']:.4f} | "
    f"F1 {test['f1']:.4f} | ACC {test['acc']:.4f}"
)

plot_roc_curve(
    test["y_true"], test["probs"],
    title="RGCN Test ROC Curve (Best Val AP Model)",
    save_path=os.path.join(PLOT_DIR, "test_roc_best_model.png"),
)
plot_confusion_matrix(
    test["y_true"], test["probs"], threshold=THRESHOLD,
    title="RGCN Test Confusion Matrix (Best Val AP Model)",
    save_path=os.path.join(PLOT_DIR, "test_confusion_best_model.png"),
)

print("RGCN test ranking metrics:")
display(ranking_metrics_at_k(test["y_true"], test["probs"], k_list=TOPK_LIST))
plot_training_curves(history, save_dir=PLOT_DIR)
```

## 13) (Optional) Quick Threshold Sweep for Recall Priority
```python
threshold_grid = np.linspace(0.05, 0.95, 37)
val_raw = evaluate(model, val_loader, device, threshold=THRESHOLD, return_raw=True)
y_val, p_val = val_raw["y_true"], val_raw["probs"]

rows = []
for th in threshold_grid:
    m = metrics_from_probs(y_val, p_val, threshold=th)
    rows.append({
        "threshold": th,
        "recall": m["recall"],
        "precision": m["precision"],
        "f1": m["f1"],
        "acc": m["acc"],
    })

thr_df = pd.DataFrame(rows).sort_values(["recall", "precision"], ascending=False)
display(thr_df.head(10))
```
