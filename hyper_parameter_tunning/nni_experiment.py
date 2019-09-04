import logging
import sys
import os
import nni

f = open("curr_pwd", "wt")
cwd = os.getcwd()

sys.path.insert(1, os.path.join(cwd, ".."))
sys.path.insert(1, os.path.join(cwd, "..", "..", "dev_graph-measures"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "dev_graph-measures", "features_algorithms"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "dev_graph-measures", "graph_infra"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "dev_graph-measures", "features_infra"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "dev_graph-measures", "features_meta"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "dev_graph-measures", "features_algorithms", "vertices"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "dev_graphs-package", "features_processor"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "dev_graphs-package", "multi_graph"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "dev_graphs-package", "temporal_graphs"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "dev_graphs-package", "features_processor", "motif_variations"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "dev_graphs-package"))

logger = logging.getLogger("NNI_logger")

from dataset.dataset_model import GnxDataset
from gcn_model import MultiLevelGCN
from multi_class_gcn_activator import MultiClassGCNActivator
from params.cora_params import CoraDatasetParams, CoraMultiLevelGCNParams, CoraGCNActivatorParams
from torch.optim import Adam, SGD
from params.parameters import BFS, CENTRALITY, DEG, GCNParams, MLPParams

ADAM = Adam
SGD = SGD
NONE = None


def run_trial(params, dataset_param_class, module_param_class, activator_param_class):
    ds_params = dataset_param_class()
    ds_params.FEATURES = [globals()[ftr] for ftr in params['input_vec']]
    dataset = GnxDataset(ds_params)

    # model
    layers = []
    for in_dim, out_dim in zip([dataset.len_features] + params['layers_config'], params['layers_config']):
        layers.append(GCNParams(in_dim=in_dim, out_dim=out_dim))

    model_params = module_param_class(ftr_len=dataset.len_features, num_classes=dataset.num_classes)
    model_params.GCN_PARAMS_LIST = layers
    model_params.DROPOUT = params['dropout']
    model_params.WEIGHT_DECAY = params['regularization']
    model_params.LR = params['learning_rate']
    model_params.OPTIMIZER = globals()[params['optimizer']]
    model_params.MLP_PARAMS = MLPParams(layers_dim=(out_dim, out_dim // 2, dataset.num_classes))

    # activator
    activator_params = activator_param_class()
    activator_params.BATCH_SIZE = params['batch_size']
    activator_params.EPOCHS = params['epochs']

    model = MultiLevelGCN(model_params)
    activator = MultiClassGCNActivator(model, activator_params, dataset, nni=True)
    activator.train(show_plot=False, early_stop=True)


def main(data):
    try:
        # get parameters form tuner
        params = nni.get_next_parameter()
        logger.debug(params)
        run_trial(params, *get_params_by_dataset(data))
    except Exception as exception:
        logger.error(exception)
        raise


def get_params_by_dataset(data):
    dict_classes = {
        "Cora": [CoraDatasetParams, CoraMultiLevelGCNParams, CoraGCNActivatorParams]
    }
    return dict_classes[data]


if __name__ == "__main__":
    data = sys.argv[1]
    # data = "Cora"
    main(data)
