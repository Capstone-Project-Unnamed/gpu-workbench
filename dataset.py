import os
import pickle
import torch
import numpy as np

from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData


class BrepDataset(Dataset):

    def __init__(self, root_dir):

        super().__init__()

        self.root_dir = root_dir

        self.files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".pkl")
        ])

    def len(self):
        return len(self.files)

    def get(self, idx):

        #################################################
        # Load PKL
        #################################################

        with open(self.files[idx], "rb") as f:
            sample = pickle.load(f)

        data = HeteroData()

        #################################################
        # Node Feature
        #################################################

        # Face Node
        data["face"].x = torch.tensor(
            sample["surf_wcs"], dtype=torch.float
        )

        # Edge Node
        data["edge"].x = torch.tensor(
            sample["edge_wcs"], dtype=torch.float
        )

        # Vertex Node
        data["vertex"].x = torch.tensor(
            sample["vertex_wcs"], dtype=torch.float
        )

        #################################################
        # Face -> Edge Graph
        #################################################

        FE = sample["input_FaceEdgeAdj"]

        face_idx, edge_idx = FE.nonzero()

        data["face", "contains", "edge"].edge_index = torch.tensor(
            [face_idx, edge_idx], dtype=torch.long
        )

        #################################################
        # Edge -> Face (Reverse)
        #################################################

        data["edge", "rev_contains", "face"].edge_index = torch.tensor(
            [edge_idx, face_idx], dtype=torch.long
        )

        #################################################
        # Edge -> Vertex Graph
        #################################################

        EV = sample["input_EdgeVertexAdj"]

        edge_idx, vertex_idx = EV.nonzero()

        data["edge", "contains", "vertex"].edge_index = torch.tensor(
            [edge_idx, vertex_idx], dtype=torch.long
        )
      
        #################################################
        # Vertex -> Edge (Reverse)
        #################################################

        data["vertex", "rev_contains", "edge"].edge_index = torch.tensor(
            [vertex_idx, edge_idx], dtype=torch.long
        )
        
        #################################################
        # Face-Face adjacency
        #################################################
        
        FF = FE @ FE.T
        FF = (FF > 0).astype(np.int32)
        np.fill_diagonal(FF, 0)

        f1, f2 = np.where(FF == 1)

        data['face', 'adj', 'face'].edge_index = torch.tensor(
            [f1, f2], dtype=torch.long
        )
        
        #################################################
        # Edge-Edge adjacency
        #################################################
        
        EE = EV @ EV.T
        EE = (EE > 0).astype(np.int32)
        np.fill_diagonal(EE, 0)

        e1, e2 = np.where(EE == 1)

        data['edge', 'adj', 'edge'].edge_index = torch.tensor(
            [e1, e2], dtype=torch.long
        )

        #################################################
        # Detection Label
        #################################################

        data["face"].y = torch.tensor(
            sample["changed_face_label"],
            dtype=torch.float
        )

        data["edge"].y = torch.tensor(
            sample["changed_edge_label"],
            dtype=torch.float
        )

        # #################################################
        # # Input / Target
        # #################################################

        # input_FE = torch.tensor(
        #     sample["input_FaceEdgeAdj"],
        #     dtype=torch.float
        # )

        # target_FE = torch.tensor(
        #     sample["target_FaceEdgeAdj"],
        #     dtype=torch.float
        # )

        # input_EV = torch.tensor(
        #     sample["input_EdgeVertexAdj"],
        #     dtype=torch.float
        # )

        # target_EV = torch.tensor(
        #     sample["target_EdgeVertexAdj"],
        #     dtype=torch.float
        # )

        # data.input_FE = input_FE
        # data.target_FE = target_FE

        # data.input_EV = input_EV
        # data.target_EV = target_EV

        # #################################################
        # # Delta (수정해야 하는 연결만)
        # #################################################

        # data.delta_FE = (input_FE != target_FE).float()
        # data.delta_EV = (input_EV != target_EV).float()

        # #################################################
        # # Repair Target
        # #################################################

        # problem_face = torch.where(
        #     data["face"].y == 1
        # )[0]

        # if len(problem_face) > 0:

        #     problem_face = problem_face[0]

        #     data.problem_face = problem_face

        #     data.repair_target = data.delta_FE[
        #         problem_face
        #     ]

        # else:

        #     data.problem_face = torch.tensor(
        #         -1
        #     )

        #     data.repair_target = torch.zeros(
        #         input_FE.shape[1]
        #     )

        #################################################
        # Error Type (디버깅용)
        #################################################

        data.error_type = sample["error_type"]

        return data
    
if __name__ == "__main__":

    dataset = BrepDataset(
        "processed_dataset"
    )

    print(dataset)

    graph = dataset[0]

    print(graph)

    print("Face Feature :", graph["face"].x.shape)
    print("Edge Feature :", graph["edge"].x.shape)
    print("Vertex Feature :", graph["vertex"].x.shape)

    print("Problem Face :", graph.problem_face)

    print("Repair Target :", graph.repair_target.shape)