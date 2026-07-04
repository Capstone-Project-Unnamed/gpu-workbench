import torch
import torch.nn as nn
import numpy as np

from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from dataset import BrepDataset
from model import BrepGNN

############################################################
# Device
############################################################

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)

############################################################
# Dataset
############################################################

dataset = BrepDataset("processed_dataset")

print("Total Graphs :", len(dataset))

sample = dataset[0]

face_dim = sample["face"].x.shape[1]
edge_dim = sample["edge"].x.shape[1]
vertex_dim = sample["vertex"].x.shape[1]

train_idx, valid_idx = train_test_split(

    list(range(len(dataset))),
    test_size=0.2,
    random_state=42,
    shuffle=True

)

train_dataset = dataset.index_select(train_idx)
valid_dataset = dataset.index_select(valid_idx)

train_loader = DataLoader(

    train_dataset,
    batch_size=8,
    shuffle=True

)

valid_loader = DataLoader(

    valid_dataset,
    batch_size=8,
    shuffle=False

)

############################################################
# Model
############################################################

model = BrepGNN(
    face_dim=face_dim,
    edge_dim=edge_dim,
    vertex_dim=vertex_dim
).to(device)

############################################################
# Loss
############################################################
def compute_pos_weight(dataset):
    face_pos = 0
    face_neg = 0
    edge_pos = 0
    edge_neg = 0

    for data in dataset:
        face_y = data["face"].y
        edge_y = data["edge"].y

        face_pos += (face_y == 1).sum().item()
        face_neg += (face_y == 0).sum().item()

        edge_pos += (edge_y == 1).sum().item()
        edge_neg += (edge_y == 0).sum().item()

    face_pw = min(face_neg / (face_pos + 1e-6), 10.0)
    edge_pw = min(edge_neg / (edge_pos + 1e-6), 10.0)

    return face_pw, edge_pw

face_pw, edge_pw = compute_pos_weight(train_dataset)

face_pos_weight = torch.tensor([face_pw], dtype=torch.float).to(device)
edge_pos_weight = torch.tensor([edge_pw], dtype=torch.float).to(device)

face_loss_fn = nn.BCEWithLogitsLoss(
    pos_weight=face_pos_weight
)

edge_loss_fn = nn.BCEWithLogitsLoss(
    pos_weight=edge_pos_weight
)

############################################################
# Optimizer
############################################################

optimizer = torch.optim.Adam(

    model.parameters(),
    lr=1e-3

)

############################################################
# Train
############################################################

def train_one_epoch():

    model.train()

    total_loss = 0
    

    for graph in train_loader:

        graph = graph.to(device)

        optimizer.zero_grad()

        output = model(graph)

        ####################################################
        # Face Loss
        ####################################################

        face_target = graph["face"].y.float()

        face_loss = face_loss_fn(

            output["face_logits"],
            face_target

        )

        ####################################################
        # Edge Loss
        ####################################################

        edge_target = graph["edge"].y.float()

        edge_loss = edge_loss_fn(

            output["edge_logits"],
            edge_target

        )

        ####################################################
        # Total
        ####################################################

        loss = 0.7 * face_loss + 0.3 * edge_loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

############################################################
# Validation
############################################################

@torch.no_grad()

def validate():

    model.eval()

    total_loss = 0

    face_preds = []
    face_labels = []

    edge_preds = []
    edge_labels = []

    for graph in valid_loader:

        graph = graph.to(device)

        output = model(graph)
        
        face_logits = output["face_logits"]
        edge_logits = output["edge_logits"]

        ####################################################
        # Loss
        ####################################################

        face_loss = face_loss_fn(
            face_logits,
            graph["face"].y.float()

        )

        edge_loss = edge_loss_fn(
            edge_logits,
            graph["edge"].y.float()

        )

        total_loss += (face_loss + edge_loss).item()

        ####################################################
        # Face, Edge Accuracy
        ####################################################

        threshold = 0.7
        
        face_pred = (torch.sigmoid(face_logits) > threshold).cpu().numpy()
        edge_pred = (torch.sigmoid(edge_logits) > threshold).cpu().numpy()

        face_preds.extend(face_pred)
        edge_preds.extend(edge_pred)

        face_labels.extend(graph["face"].y.cpu().numpy())
        edge_labels.extend(graph["edge"].y.cpu().numpy())

    ####################################################
    # Metrics
    ####################################################

    face_precision = precision_score(face_labels, face_preds, zero_division=0)
    face_recall = recall_score(face_labels, face_preds, zero_division=0)
    face_f1 = f1_score(face_labels, face_preds, zero_division=0)

    edge_precision = precision_score(edge_labels, edge_preds, zero_division=0)
    edge_recall = recall_score(edge_labels, edge_preds, zero_division=0)
    edge_f1 = f1_score(edge_labels, edge_preds, zero_division=0)

    return (
        total_loss / len(valid_loader),
        face_precision, face_recall, face_f1,
        edge_precision, edge_recall, edge_f1
    )

############################################################
# Main Loop
############################################################

best_loss = 999999

EPOCHS = 50

for epoch in range(EPOCHS):

    train_loss = train_one_epoch()

    valid_loss, fp, fr, ff1, ep, er, ef1 = validate()

    print(
        f"Epoch {epoch+1:03d}"
        f" | Train {train_loss:.4f}"
        f" | Valid {valid_loss:.4f}"
        f" | Face P/R/F1 {fp:.3f}/{fr:.3f}/{ff1:.3f}"
        f" | Edge P/R/F1 {ep:.3f}/{er:.3f}/{ef1:.3f}"
    )

    ########################################################

    if valid_loss < best_loss:

        best_loss = valid_loss

        torch.save(
            model.state_dict(),
            "best_detection_model.pt"
        )

        print("Model Saved")