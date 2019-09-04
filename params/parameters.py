from torch import sigmoid, tanh
from torch.optim import Adam, SGD
from torch.nn import functional
import os
from betweenness_centrality import BetweennessCentralityCalculator
from bfs_moments import BfsMomentsCalculator
from feature_calculators import FeatureMeta

CODE_DIR = "code"
DATA_INPUT_DIR = "dataset_input"
PKL_DIR = "pkl"
FEATURES_PKL_DIR = os.path.join("features")
DATA_PKL_DIR = os.path.join("data")
GNX_PKL_DIR = os.path.join("gnx")
NORM_REDUCED = "_REDUCED_"
NORM_REDUCED_SYMMETRIC = "_REDUCED_SYMMETRIC_"

DEG = "_DEGREE_"
IN_DEG = "_IN_DEGREE_"
OUT_DEG = "_OUT_DEGREE_"
CENTRALITY = ["betweenness_centrality", FeatureMeta(BetweennessCentralityCalculator, {"betweenness"})]
BFS = ["bfs_moments", FeatureMeta(BfsMomentsCalculator, {"bfs"})]


class DatasetParams:
    def __init__(self):
        self.DATASET_NAME = "Cora"
        self.DATASET_EDGES_FILENAME = "cora_graph_edges.txt"
        self.DATASET_TAGS_FILENAME = "cora_tags.txt"
        self.SRC_COL = "n1"
        self.DST_COL = "n2"
        self.TRAIN_PERCENTAGE = 0.01
        self.DIRECTED = False
        self.NORM = NORM_REDUCED
        self.FEATURES = [DEG, IN_DEG, OUT_DEG, CENTRALITY, BFS]

    @property
    def id(self):
        attributes = ["DATASET_NAME", "NORM", "TRAIN_PERCENTAGE", "DIRECTED", "FEATURES"]

        attr_str = []
        for attr in attributes:
            if attr == "FEATURES":
                attr_str.append(attr + "_" + str([k[0] if type(k) is list else k for k in self.FEATURES]))
            else:
                attr_str.append(attr + "_" + str(getattr(self, attr)))
        return "_".join(attr_str)


class GCNParams:
    def __init__(self, in_dim, out_dim, activation=tanh, dropout=0.3):
        self.ROW_DIM = in_dim
        self.COL_DIM = out_dim
        self.ACTIVATION_FUNC = activation
        self.DROPOUT = dropout


class MLPParams:
    def __init__(self, activation=tanh, layers_dim=(100, 50, 10)):
        self.LAYERS = layers_dim                    # in to out
        self.ACTIVATION_FUNC = activation


class MultiLevelGCNParams:
    def __init__(self, ftr_len=6, num_classes=4):
        self.DROPOUT = 0
        self.LR = 1e-3
        self.OPTIMIZER = Adam
        self.WEIGHT_DECAY = 0
        self.GCN_PARAMS_LIST = [
            GCNParams(in_dim=ftr_len, out_dim=50, activation=tanh, dropout=self.DROPOUT),
            GCNParams(in_dim=50, out_dim=25, activation=tanh, dropout=self.DROPOUT)
        ]
        self.MLP_PARAMS = MLPParams(activation=tanh, layers_dim=(25, 15, num_classes))


class GCNActivatorParams:
    def __init__(self):
        self.TRAIN_SIZE = 0.8
        self.LOSS = functional.cross_entropy
        self.BATCH_SIZE = 64
        self.EPOCHS = 2
        self.DATASET = ""
