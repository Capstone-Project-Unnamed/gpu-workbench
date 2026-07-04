import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    HeteroConv,
    SAGEConv,
)

class FeatureProjection(nn.Module):
    def __init__(self, face_dim, edge_dim, vertex_dim, hidden_dim=64):
        super().__init__()

        self.face_proj = nn.Linear(face_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.vertex_proj = nn.Linear(vertex_dim, hidden_dim)

    def forward(self, x_dict):
        x_dict["face"] = F.relu(self.face_proj(x_dict["face"]))
        x_dict["edge"] = F.relu(self.edge_proj(x_dict["edge"]))
        x_dict["vertex"] = F.relu(self.vertex_proj(x_dict["vertex"]))
        return x_dict
    
class HeteroEncoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim)
        
        self.conv1 = HeteroConv({
            ("face", "contains", "edge"):
                SAGEConv((-1, -1), hidden_dim),
            ("edge", "rev_contains", "face"):
                SAGEConv((-1, -1), hidden_dim),
            ("edge", "contains", "vertex"):
                SAGEConv((-1, -1), hidden_dim),
            ("vertex", "rev_contains", "edge"):
                SAGEConv((-1, -1), hidden_dim),
            ("face", "adj", "face"):
                SAGEConv((-1, -1), hidden_dim),
            ("edge", "adj", "edge"):
                SAGEConv((-1, -1), hidden_dim),
        })

        self.conv2 = HeteroConv({
            ("face", "contains", "edge"):
                SAGEConv((-1, -1), hidden_dim),
            ("edge", "rev_contains", "face"):
                SAGEConv((-1, -1), hidden_dim),
            ("edge", "contains", "vertex"):
                SAGEConv((-1, -1), hidden_dim),
            ("vertex", "rev_contains", "edge"):
                SAGEConv((-1, -1), hidden_dim),
            ("face", "adj", "face"):
                SAGEConv((-1, -1), hidden_dim),
            ("edge", "adj", "edge"):
                SAGEConv((-1, -1), hidden_dim),
        })

    def forward(self, x_dict, edge_index_dict):
            
        x_dict = self.conv1(
            x_dict,
            edge_index_dict
        )

        x_dict = {
            key: self.norm(F.relu(value))
            for key, value in x_dict.items()
        }

        x_dict = self.conv2(
            x_dict,
            edge_index_dict
        )

        x_dict = {
            key: self.norm(F.relu(value))
            for key, value in x_dict.items()
        }

        return x_dict
    
class DetectionHead(nn.Module):

    def __init__(self, hidden_dim=64):

        super().__init__()

        self.face_classifier = nn.Sequential(

            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1)

        )

        self.edge_classifier = nn.Sequential(

            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1)

        )

    def forward(self, x_dict):

        face_logits = self.face_classifier(
            x_dict["face"]
        ).squeeze(-1)

        edge_logits = self.edge_classifier(
            x_dict["edge"]
        ).squeeze(-1)

        return face_logits, edge_logits
    
class BrepGNN(nn.Module):

    def __init__(self, face_dim, edge_dim, vertex_dim, hidden_dim=64):

        super().__init__()

        self.projector = FeatureProjection(
            face_dim,
            edge_dim,
            vertex_dim,
            hidden_dim
        )

        self.encoder = HeteroEncoder(hidden_dim)

        self.detector = DetectionHead(hidden_dim)

    def forward(self, data):

        ############################################
        # Node Feature
        ############################################

        x_dict = {

            "face": data["face"].x,

            "edge": data["edge"].x,

            "vertex": data["vertex"].x,

        }

        ############################################
        # Feature Projection
        ############################################

        x_dict = self.projector(x_dict)

        ############################################
        # Heterogeneous GNN
        ############################################

        x_dict = self.encoder(

            x_dict,

            data.edge_index_dict

        )

        ############################################
        # Detection
        ############################################

        face_logits, edge_logits = self.detector(
            x_dict
        )

        ############################################
        # Return
        ############################################

        return {

            "face_logits": face_logits,

            "edge_logits": edge_logits,

            "face_embedding": x_dict["face"],

            "edge_embedding": x_dict["edge"],

            "vertex_embedding": x_dict["vertex"]

        }
        
if __name__ == "__main__":

    print("BrepGNN Loaded.")