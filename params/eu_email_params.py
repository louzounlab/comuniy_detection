from torch import sigmoid, tanh
from torch.nn.functional import relu
from torch.optim import Adam
from torch.nn import functional

from params.parameters import NORM_REDUCED, DEG, CENTRALITY, BFS, GCNParams, DatasetParams, MultiLevelGCNParams, \
    GCNActivatorParams, MLPParams


class EuEmailDatasetParams(DatasetParams):
    def __init__(self):
        super(EuEmailDatasetParams, self).__init__()
        self.DATASET_NAME = "Eu-Email"
        self.DATASET_EDGES_FILENAME = "eu_email_graph_edges.txt"
        self.DATASET_TAGS_FILENAME = "eu_email_tags.txt"
        self.SRC_COL = "n1"
        self.DST_COL = "n2"
        self.TRAIN_PERCENTAGE = 0.5
        self.DIRECTED = False
        self.NORM = NORM_REDUCED
        self.FEATURES = [DEG, CENTRALITY, BFS]


class EuEmailMultiLevelGCNParams(MultiLevelGCNParams):
    def __init__(self, ftr_len=6, num_classes=4):
        super(EuEmailMultiLevelGCNParams, self).__init__(ftr_len, num_classes)
        self.DROPOUT = 0
        self.LR = 1e-3
        self.OPTIMIZER = Adam
        self.WEIGHT_DECAY = 1e-3
        self.GCN_PARAMS_LIST = [
            GCNParams(in_dim=ftr_len, out_dim=100, activation=tanh, dropout=self.DROPOUT),
            GCNParams(in_dim=100, out_dim=50, activation=tanh, dropout=self.DROPOUT)
        ]
        self.MLP_PARAMS = MLPParams(activation=tanh, layers_dim=(50, 15, num_classes))


class EuEmailGCNActivatorParams(GCNActivatorParams):
    def __init__(self):
        super(EuEmailGCNActivatorParams, self).__init__()
        self.TRAIN_SIZE = 0.9
        self.LOSS = functional.cross_entropy
        self.BATCH_SIZE = 16
        self.EPOCHS = 50
        self.DATASET = ""


if __name__ == '__main__':
    from gcn_model import MultiLevelGCN
    from dataset.dataset_model import GnxDataset
    from multi_class_gcn_activator import MultiClassGCNActivator

    ds_ = GnxDataset(EuEmailDatasetParams())
    model_ = MultiLevelGCN(EuEmailMultiLevelGCNParams(ftr_len=ds_.len_features, num_classes=ds_.num_classes))
    activator = MultiClassGCNActivator(model_, EuEmailGCNActivatorParams(), ds_, nni=False)
    activator.train(validation_rate=2)
