from torch.nn import Module, Linear, Dropout, Embedding, ModuleList
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from params.parameters import GCNParams, MultiLevelGCNParams, MLPParams

"""
given A, x0 : A=Adjacency_matrix, x0=nodes_vec
First_model => x1(n x k) = sigma( A(n x n) * x0(n x d) * W1(d x k) )
"""


class MLP(Module):
    def __init__(self, params: MLPParams):
        super(MLP, self).__init__()
        # build linear layers
        self._linear_layers = ModuleList([Linear(row_dim, col_dim) for row_dim, col_dim in
                                          zip(params.LAYERS, params.LAYERS[1:])])
        self._last_layer = len(self._linear_layers) - 1
        self._activation = params.ACTIVATION_FUNC

    def forward(self, x):
        # Dropout layer
        out = encode = x
        for layer_idx, linear_layer in enumerate(self._linear_layers):
            encode = out
            out = linear_layer(out)
            out = softmax(out, dim=1) if layer_idx == self._last_layer else self._activation(out)
        return encode, out


class GCN(Module):
    def __init__(self, params: GCNParams):
        super(GCN, self).__init__()
        # useful info in forward function
        self._linear = Linear(params.ROW_DIM, params.COL_DIM)
        self._activation = params.ACTIVATION_FUNC
        self._dropout = Dropout(p=params.DROPOUT) if params.DROPOUT else None
        self._gpu = False

    def gpu_device(self, is_gpu: bool):
        self._gpu = is_gpu

    def _sync(self):
        if self._gpu:
            torch.cuda.synchronize()

    def forward(self, A, x0):
        # Dropout layer
        x0 = self._dropout(x0) if self._dropout else x0
        # tanh( A(n x n) * x0(n x d) * W1(d x k) )
        Ax = torch.matmul(A, x0)
        self._sync()

        x = self._linear(Ax)
        x1 = self._activation(x)
        return x1


class MultiLevelGCN(Module):
    """
    gcn layer is executed numerous times
    """
    def __init__(self, params: MultiLevelGCNParams):
        super(MultiLevelGCN, self).__init__()
        # create gcn layers
        self._gcn_layers = ModuleList([GCN(gcn_params) for gcn_params in params.GCN_PARAMS_LIST])
        self._mlp = MLP(params.MLP_PARAMS)
        self.optimizer = self.set_optimizer(params.LR, params.OPTIMIZER, params.WEIGHT_DECAY)
        self._gpu = False

    def gpu_device(self, is_gpu: bool):
        self._gpu = is_gpu
        for layer in self._linear_layers:
            layer.gpu_device(is_gpu)
        self._bilinear_layer.gpu_device(is_gpu)

    def set_optimizer(self, lr, opt, weight_decay):
        return opt(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, A, x0, nodes_idx):
        for gcn_layer in self._gcn_layers:
            x0 = gcn_layer(A, x0)
        encode, x1 = self._mlp(x0[nodes_idx])
        return encode, x1


if __name__ == "__main__":
    from dataset.dataset_model import GnxDataset
    from params.parameters import DatasetParams
    from dataset.datset_sampler import ImbalancedDatasetSampler
    ds = GnxDataset(DatasetParams())
    dl = DataLoader(
        dataset=ds,
        batch_size=64,
        sampler=ImbalancedDatasetSampler(ds)
    )
    model_ = MultiLevelGCN(MultiLevelGCNParams(ftr_len=ds.len_features))
    A_ = ds.adjacency_matrix
    x0_ = ds.ftr_vec

    for i, (node_idx_, label_) in enumerate(dl):
        print(model_(A_, x0_, node_idx_))
        e = 0
