# Current Hetero Fraud Model with GATv2 (Notebook-Style Cells, Leak-Safe Temporal Context)

## 1) Imports
```python
import os
import copy
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx

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
from torch_geometric.nn import GATv2Conv
import torch_geometric.transforms as T
```

## 2) Config
```python
TX_PATH = r"C:\Users\LENOVO\Documents\machine_learning_utec\Dataset_P2\train_transaction.csv"
ID_PATH = r"C:\Users\LENOVO\Documents\machine_learning_utec\Dataset_P2\train_identity.csv"
FEAT_PATH = r"C:\Users\LENOVO\Documents\machine_learning_utec\Dataset_P2\data_with_hdbscan_feat.parquet"

OUTPUT_DIR = r"C:\Users\LENOVO\Documents\machine_learning_utec\Dataset_P2"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots_gatv2")
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
HIDDEN_CHANNELS = 32
NUM_LAYERS = 2
HEADS = 4
DROPOUT = 0.2
ATTN_DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
EARLY_STOP_PATIENCE = 8
EARLY_STOP_MIN_DELTA = 0.002
THRESHOLD = 0.5
TOPK_LIST = [100, 500, 1000, 5000]
RELATION_AGGR = "sum"

TRAIN_BS = 1024
EVAL_BS = 2048
NUM_NEIGHBORS = [15, 10]
USE_UNDIRECTED_CONTEXT = True
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

## 4) Data Helpers
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
print("Time cutoffs:", {"q70": q70, "q85": q85, "max_t": max_t})
```

## 6) (Optional) Visualize Seed Subgraphs
```python
def pick_edge_type(data_obj, preferred_rel="has_P_email_bin"):
    for et in data_obj.edge_types:
        if et[0] == "transaction" and et[1] == preferred_rel:
            return et
    for et in data_obj.edge_types:
        if et[0] == "transaction":
            return et
    raise ValueError("No transaction->entity edge type found.")


def build_seed_bipartite_graph(data_obj, edge_type, mask_name="seed_mask", max_tx=250, seed=42):
    tx_idx = data_obj["transaction"][mask_name].nonzero(as_tuple=True)[0]
    if tx_idx.numel() > max_tx:
        g = torch.Generator().manual_seed(seed)
        tx_idx = tx_idx[torch.randperm(tx_idx.numel(), generator=g)[:max_tx]]

    edge_index = data_obj[edge_type].edge_index.cpu()
    src, dst = edge_index[0], edge_index[1]
    keep = torch.isin(src, tx_idx.cpu())
    src, dst = src[keep], dst[keep]

    G = nx.Graph()
    G.add_nodes_from([f"t_{i}" for i in tx_idx.tolist()], ntype="transaction")
    G.add_nodes_from([f"e_{j}" for j in torch.unique(dst).tolist()], ntype=edge_type[2])
    G.add_edges_from((f"t_{s.item()}", f"e_{d.item()}") for s, d in zip(src, dst))
    return G, tx_idx


def plot_seed_graph(data_obj, edge_type, title, mask_name="seed_mask", max_tx=250, seed=42):
    G, tx_idx = build_seed_bipartite_graph(data_obj, edge_type, mask_name, max_tx, seed)
    y = data_obj["transaction"].y.cpu()
    node_colors = ["red" if (n.startswith("t_") and int(y[int(n[2:])]) == 1)
                   else "dodgerblue" if n.startswith("t_")
                   else "lightgray" for n in G.nodes()]

    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=seed, k=0.25)
    nx.draw(G, pos, node_color=node_colors, edge_color="silver", node_size=60, with_labels=False)
    plt.title(f"{title} | edge_type={edge_type} | seed_tx={len(tx_idx)}")
    plt.axis("off")
    plt.show()


for name, graph in [("Train context", train_data), ("Val context", val_data), ("Test context", test_data)]:
    et = pick_edge_type(graph, preferred_rel="has_P_email_bin")
    plot_seed_graph(graph, et, title=name, mask_name="seed_mask", max_tx=250, seed=42)
```

## 7) Build Neighbor Loaders (Leak-Safe)
```python
train_idx = train_data["transaction"].train_mask.nonzero(as_tuple=True)[0]
val_idx = val_data["transaction"].val_mask.nonzero(as_tuple=True)[0]
test_idx = test_data["transaction"].test_mask.nonzero(as_tuple=True)[0]

train_loader = NeighborLoader(
    train_data,
    input_nodes=("transaction", train_idx),
    num_neighbors={et: NUM_NEIGHBORS for et in train_data.edge_types},
    batch_size=TRAIN_BS,
    shuffle=True,
)
val_loader = NeighborLoader(
    val_data,
    input_nodes=("transaction", val_idx),
    num_neighbors={et: NUM_NEIGHBORS for et in val_data.edge_types},
    batch_size=EVAL_BS,
    shuffle=False,
)
test_loader = NeighborLoader(
    test_data,
    input_nodes=("transaction", test_idx),
    num_neighbors={et: NUM_NEIGHBORS for et in test_data.edge_types},
    batch_size=EVAL_BS,
    shuffle=False,
)
```

## 8) Define GATv2 Model
```python
def edge_type_to_key(edge_type):
    return "__".join(edge_type)


class FraudHeteroGATv2(nn.Module):
    def __init__(
        self, data, hidden_channels=32, num_layers=2, heads=4,
        dropout=0.2, attn_dropout=0.2, relation_aggr="sum"
    ):
        super().__init__()
        self.dropout = dropout
        self.relation_aggr = relation_aggr
        self.node_types, self.edge_types = data.metadata()

        self.tx_encoder = nn.Linear(data["transaction"].x.size(-1), hidden_channels)
        self.emb_dict = nn.ModuleDict({
            nt: nn.Embedding(data[nt].num_nodes, hidden_channels)
            for nt in self.node_types if nt != "transaction"
        })

        self.edge_key_map = {et: edge_type_to_key(et) for et in self.edge_types}

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            layer_convs = nn.ModuleDict()
            for et in self.edge_types:
                key = self.edge_key_map[et]
                layer_convs[key] = GATv2Conv(
                    (-1, -1), hidden_channels, heads=heads, concat=False,
                    dropout=attn_dropout, add_self_loops=False
                )
            self.convs.append(layer_convs)

        self.classifier = nn.Linear(hidden_channels, 1)
        self.last_attention_summary = {}

    def _aggregate_relation_messages(self, messages):
        stacked = torch.stack(messages, dim=0)
        if self.relation_aggr == "sum":
            return stacked.sum(dim=0)
        if self.relation_aggr == "mean":
            return stacked.mean(dim=0)
        if self.relation_aggr == "max":
            return stacked.max(dim=0).values
        raise ValueError(f"Unknown relation_aggr: {self.relation_aggr}")

    def forward(self, batch, collect_attention=False):
        x_dict = {"transaction": self.tx_encoder(batch["transaction"].x.float())}
        for nt, emb in self.emb_dict.items():
            if nt in batch.node_types and "n_id" in batch[nt]:
                x_dict[nt] = emb(batch[nt].n_id)

        self.last_attention_summary = {}

        for layer_idx, layer_convs in enumerate(self.convs, start=1):
            out_dict = {nt: [] for nt in x_dict.keys()}
            layer_summary = {}

            for et in self.edge_types:
                src_type, _, dst_type = et
                key = self.edge_key_map[et]
                if src_type not in x_dict or dst_type not in x_dict or et not in batch.edge_types:
                    continue

                conv = layer_convs[key]
                edge_index = batch[et].edge_index

                if collect_attention:
                    out, (_, alpha) = conv((x_dict[src_type], x_dict[dst_type]), edge_index, return_attention_weights=True)
                    alpha_mean = alpha.mean(dim=-1) if alpha.dim() == 2 else alpha
                    layer_summary[et] = float(alpha_mean.mean().detach().cpu())
                else:
                    out = conv((x_dict[src_type], x_dict[dst_type]), edge_index)

                out_dict[dst_type].append(out)

            new_x_dict = {}
            for nt in x_dict.keys():
                if out_dict[nt]:
                    h = self._aggregate_relation_messages(out_dict[nt])
                    h = F.dropout(F.elu(h), p=self.dropout, training=self.training)
                    new_x_dict[nt] = h
                else:
                    new_x_dict[nt] = x_dict[nt]

            x_dict = new_x_dict
            if collect_attention:
                self.last_attention_summary[layer_idx] = layer_summary

        return self.classifier(x_dict["transaction"]).squeeze(-1)
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
        bs = batch["transaction"].batch_size
        probs_all.append(torch.sigmoid(logits[:bs]).cpu().numpy())
        y_all.append(batch["transaction"].y[:bs].cpu().numpy())
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
    plt.title("GATv2 Training Loss Curve"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(alpha=0.2); plt.legend()
    if save_dir is not None: plt.savefig(os.path.join(save_dir, "loss_curve_gatv2.png"), dpi=140, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["val_auc"], label="Val AUC")
    plt.plot(history["epoch"], history["val_ap"], label="Val AP")
    plt.plot(history["epoch"], history["val_recall"], label="Val Recall")
    plt.plot(history["epoch"], history["val_precision"], label="Val Precision")
    plt.plot(history["epoch"], history["val_f1"], label="Val F1")
    plt.title("GATv2 Validation Metrics by Epoch"); plt.xlabel("Epoch"); plt.ylabel("Score")
    plt.grid(alpha=0.2); plt.legend()
    if save_dir is not None: plt.savefig(os.path.join(save_dir, "val_metrics_curve_gatv2.png"), dpi=140, bbox_inches="tight")
    plt.show()


@torch.no_grad()
def summarize_relation_attention(model, loader, device, max_batches=None):
    model.eval()
    bucket = defaultdict(list)
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        _ = model(batch, collect_attention=True)
        for layer_idx, rel_map in model.last_attention_summary.items():
            for message_type, score in rel_map.items():
                bucket[(layer_idx, message_type)].append(score)
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break

    summary = defaultdict(dict)
    for (layer_idx, message_type), scores in bucket.items():
        summary[layer_idx][message_type] = float(np.mean(scores))
    return dict(summary)


def print_relation_attention(summary):
    if not summary:
        print("No attention summary available.")
        return
    for layer_idx in sorted(summary.keys()):
        print(f"Layer {layer_idx} relation attention summary:")
        rows = sorted(summary[layer_idx].items(), key=lambda x: x[1], reverse=True)
        for message_type, score in rows:
            print(f"  mean edge attention {score:.6f} on message type {message_type}")
```

## 10) Initialize Training Objects
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FraudHeteroGATv2(
    data=train_data,
    hidden_channels=HIDDEN_CHANNELS,
    num_layers=NUM_LAYERS,
    heads=HEADS,
    dropout=DROPOUT,
    attn_dropout=ATTN_DROPOUT,
    relation_aggr=RELATION_AGGR,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

y_train = train_data["transaction"].y[train_idx]
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
        bs = batch["transaction"].batch_size
        y = batch["transaction"].y[:bs].float()
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
            title=f"GATv2 Validation ROC Curve - Epoch {epoch}",
            save_path=os.path.join(PLOT_DIR, f"val_roc_epoch_{epoch:03d}.png"),
        )
        plot_confusion_matrix(
            val["y_true"], val["probs"], threshold=THRESHOLD,
            title=f"GATv2 Validation Confusion Matrix - Epoch {epoch}",
            save_path=os.path.join(PLOT_DIR, f"val_confusion_epoch_{epoch:03d}.png"),
        )
        print(f"GATv2 validation ranking metrics @ epoch {epoch}:")
        display(ranking_metrics_at_k(val["y_true"], val["probs"], k_list=TOPK_LIST))

    if epochs_without_improve >= EARLY_STOP_PATIENCE:
        print(
            f"Early stopping at epoch {epoch:03d}: "
            f"no Val AP improvement > {EARLY_STOP_MIN_DELTA} for {EARLY_STOP_PATIENCE} epochs."
        )
        break

history_df = pd.DataFrame(history)
history_csv = os.path.join(OUTPUT_DIR, "training_history_gatv2.csv")
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
    title="GATv2 Test ROC Curve (Best Val AP Model)",
    save_path=os.path.join(PLOT_DIR, "test_roc_best_model.png"),
)
plot_confusion_matrix(
    test["y_true"], test["probs"], threshold=THRESHOLD,
    title="GATv2 Test Confusion Matrix (Best Val AP Model)",
    save_path=os.path.join(PLOT_DIR, "test_confusion_best_model.png"),
)

print("GATv2 test ranking metrics:")
display(ranking_metrics_at_k(test["y_true"], test["probs"], k_list=TOPK_LIST))
plot_training_curves(history, save_dir=PLOT_DIR)
```

## 13) Inspect Relation Attention by Message Type
```python
val_attention_summary = summarize_relation_attention(model, val_loader, device, max_batches=10)
print_relation_attention(val_attention_summary)
```

## 14) (Optional) Quick Threshold Sweep for Recall Priority
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
