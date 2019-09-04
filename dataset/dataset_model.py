from random import shuffle

from scipy.stats import zscore
from torch import Tensor
from torch.nn import ConstantPad2d
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from features_processor import FeaturesProcessor, log_norm
from graph_features import GraphFeatures
from loggers import PrintLogger
from multi_graph import MultiGraph
from params.parameters import DatasetParams, DATA_INPUT_DIR, PKL_DIR, FEATURES_PKL_DIR, DEG, IN_DEG, OUT_DEG, \
    NORM_REDUCED, NORM_REDUCED_SYMMETRIC, DATA_PKL_DIR, GNX_PKL_DIR
import os
import pandas as pd
import pickle
import numpy as np
import networkx as nx


class GnxDataset(Dataset):
    def __init__(self, params: DatasetParams, external_data=None):
        self._params = params
        self._logger = PrintLogger("logger")
        # path to base directory
        self._base_dir = __file__.replace("/", os.sep)
        self._base_dir = os.path.join(self._base_dir.rsplit(os.sep, 1)[0], "..")
        self._gnx_data_pkl_dir, self._gnx_features_pkl_dir, self._gnx_pkl_dir = self._create_dir_hierarchy()

        self._external_data = external_data
        # init ftr_meta dictionary and other ftr attributes
        self._init_ftrs()
        self._gnx, self._labels, self._label_to_idx, self._idx_to_label = self._build_gnx()
        self._node_order = list(sorted(self._gnx.nodes))
        self._node_to_idx = {node: i for i, node in enumerate(self._node_order)}

        self._data, self._adjacency_matrix, self._ftr_vec = self._build_data()

    @property
    def gnx(self):
        return self._gnx

    @property
    def node_order(self):
        return self._node_order

    @property
    def adjacency_matrix(self):
        return self._adjacency_matrix

    @property
    def ftr_vec(self):
        return self._ftr_vec

    @property
    def all_labels(self):
        return self._idx_to_label

    @property
    def label_count(self):
        return Counter([label for node, label in self._data])

    def label_by_node_name(self, node):
        return self._labels[node] if node in self._labels else 0

    def label(self, idx):
        return self._data[idx][1]

    @property
    def len_features(self):
        return self._ftr_vec.shape[1]

    @property
    def num_classes(self):
        return len(self._idx_to_label)

    def _create_dir_hierarchy(self):
        pkl_dir = os.path.join(self._base_dir, PKL_DIR)
        data_pkl_dir = os.path.join(self._base_dir, PKL_DIR, DATA_PKL_DIR)
        features_pkl_dir = os.path.join(self._base_dir, PKL_DIR, FEATURES_PKL_DIR)
        gnx_pkl_dir = os.path.join(self._base_dir, PKL_DIR, GNX_PKL_DIR)
        graph_data_pkl_dir = os.path.join(self._base_dir, PKL_DIR, DATA_PKL_DIR, self._params.DATASET_NAME)
        graph_features_pkl_dir = os.path.join(self._base_dir, PKL_DIR, FEATURES_PKL_DIR, self._params.DATASET_NAME)

        if not os.path.exists(pkl_dir):
            os.mkdir(pkl_dir)
        if not os.path.exists(data_pkl_dir):
            os.mkdir(data_pkl_dir)
        if not os.path.exists(features_pkl_dir):
            os.mkdir(features_pkl_dir)
        if not os.path.exists(gnx_pkl_dir):
            os.mkdir(gnx_pkl_dir)
        if not os.path.exists(graph_features_pkl_dir):
            os.mkdir(graph_features_pkl_dir)
        if not os.path.exists(graph_data_pkl_dir):
            os.mkdir(graph_data_pkl_dir)

        return graph_data_pkl_dir, graph_features_pkl_dir, gnx_pkl_dir

    def _init_ftrs(self):
        self._deg, self._in_deg, self._out_deg, self._is_ftr, self._ftr_meta = False, False, False, False, {}
        self._is_external_data = False if self._external_data is None else True
        # params.FEATURES contains string and list of two elements (matching to key: value)
        # should Deg/In-Deg/Out-Deg be calculated
        for ftr in self._params.FEATURES:
            if ftr == DEG:
                self._deg = True
            elif ftr == IN_DEG:
                self._in_deg = True
            elif ftr == OUT_DEG:
                self._out_deg = True
            else:
                # add feature to dict {key: val}
                self._ftr_meta[ftr[0]] = ftr[1]

        # indicate to use graph topological features
        if len(self._ftr_meta) > 0:
            self._is_ftr = True

    """
    build multi graph according to csv 
    each community is a single graph, no consideration to time
    """
    def _build_gnx(self):
        path_pkl = os.path.join(self._gnx_pkl_dir, self._params.DATASET_NAME + "_mg.pkl")

        # load pickle if exists
        if os.path.exists(path_pkl):
            return pickle.load(open(path_pkl, "rb"))

        labels = {}
        label_to_idx = {}
        # open basic data csv (with all edges of all times)
        data_df = pd.read_csv(os.path.join(self._base_dir, DATA_INPUT_DIR, self._params.DATASET_EDGES_FILENAME))
        # get graph edges
        edge_list = [(str(edge[self._params.SRC_COL]), str(edge[self._params.DST_COL]))
                     for index, edge in data_df.iterrows()]
        # build gnx
        gnx = nx.DiGraph() if self._params.DIRECTED else nx.Graph()
        gnx.add_edges_from(edge_list)

        # build train -> {node: label} for all data
        with open(os.path.join(self._base_dir, DATA_INPUT_DIR, self._params.DATASET_TAGS_FILENAME)) as f:
            next(f)         # skip first row
            for row in f:
                node, label = row.strip().split()
                label_to_idx[label] = len(label_to_idx) if label not in label_to_idx else label_to_idx[label]
                labels[node] = label_to_idx[label]
        idx_to_label = [l for l in sorted(label_to_idx, key=lambda x: label_to_idx[x])]

        # save pickle
        pickle.dump((gnx, labels, label_to_idx, idx_to_label), open(path_pkl, "wb"))

        return gnx, labels, label_to_idx, idx_to_label

    """
    returns a vector x for gnx 
    basic version returns degree for each node
    """
    def _gnx_vec(self, gnx: nx.Graph, node_order):
        final_vec = []
        if self._deg:
            degrees = gnx.degree(gnx.nodes)
            final_vec.append(np.matrix([np.log(degrees[d] + 1e-3) for d in node_order]).T)
        if self._in_deg and gnx.is_directed():
            degrees = gnx.in_degree(gnx.nodes)
            final_vec.append(np.matrix([np.log(degrees[d] + 1e-3) for d in node_order]).T)
        if self._out_deg and gnx.is_directed():
            degrees = gnx.out_degree(gnx.nodes)
            final_vec.append(np.matrix([np.log(degrees[d] + 1e-3) for d in node_order]).T)

        # TODO ========================== TODO handle external features
        # if self._is_external_data and self._external_data.is_value:
        #     final_vec.append(np.matrix([self._external_data.value_feature(gnx_id, d) for d in node_order]))
        # TODO ========================== TODO ========================

        if self._is_ftr:
            gnx_dir_path = os.path.join(self._gnx_features_pkl_dir)
            if not os.path.exists(gnx_dir_path):
                os.mkdir(gnx_dir_path)
            raw_ftr = GraphFeatures(gnx, self._ftr_meta, dir_path=gnx_dir_path, is_max_connected=False,
                                    logger=PrintLogger("logger"))
            raw_ftr.build(should_dump=True)  # build features
            final_vec.append(FeaturesProcessor(raw_ftr).as_matrix(norm_func=log_norm))

        return np.hstack(final_vec)

    def _degree_matrix(self, gnx, nodelist):
        degrees = gnx.degree(gnx.nodes)
        return np.diag([degrees[d] for d in nodelist])

    def _z_score_all_data(self, data):
        all_data_values_vec = []                # stack all vectors for all graphs
        key_to_idx_map = []                     # keep ordered list (g_id, num_nodes) according to stack order

        # stack
        for g_id, (A, gnx_vec, embed_vec, label) in data.items():
            all_data_values_vec.append(gnx_vec)
            key_to_idx_map.append((g_id, gnx_vec.shape[0]))  # g_id, number of nodes ... ordered
        all_data_values_vec = np.vstack(all_data_values_vec)

        # z-score data
        z_scored_data = zscore(all_data_values_vec, axis=0)

        # rebuild data to original form -> split stacked matrix according to <list: (g_id, num_nodes)>
        new_data_dict = {}
        start_idx = 0
        for g_id, num_nodes in key_to_idx_map:
            new_data_dict[g_id] = (data[g_id][0], z_scored_data[start_idx: start_idx+num_nodes],
                                   data[g_id][2], data[g_id][3])
            start_idx += num_nodes

        return new_data_dict

    """
    builds a data dictionary
    { ... graph_name: ( A = Adjacency_matrix, x = graph_vec, label ) ... }  
    """
    def _build_data(self):
        # id extension for external data
        ext_data_id = "None" if not self._is_external_data else "_embed_ftr_" + str(self._external_data.embed_headers)\
                                                                + "_value_ftr_" + str(self._external_data.value_headers)
        # file name for data pkl
        pkl_path = os.path.join(self._gnx_data_pkl_dir, self._params.id + ext_data_id + "_data.pkl")
        # load pickle if exists
        if os.path.exists(pkl_path):
            return pickle.load(open(pkl_path, "rb"))

        # get Adjacency matrix + normalize matrix
        A = nx.adjacency_matrix(self._gnx, nodelist=self._node_order).todense()
        if self._params.NORM == NORM_REDUCED:
            # D^-0.5 A D^-0.5
            D = self._degree_matrix(self._gnx, nodelist=self._node_order)
            D_sqrt = np.matrix(np.sqrt(D))
            adjacency = D_sqrt * np.matrix(A) * D_sqrt

        elif self._params.NORM == NORM_REDUCED_SYMMETRIC:
            # D^-0.5 [A + A.T + I] D^-0.5
            D = self._degree_matrix(self._gnx, nodelist=self._node_order)
            D_sqrt = np.matrix(np.sqrt(D))
            adjacency = D_sqrt * np.matrix(A + A.T + np.identity(A.shape[0])) * D_sqrt
        else:
            adjacency = A

        data = []

        ftr_vec = self._gnx_vec(self._gnx, self._node_order)
        # TODO ========================== TODO handle external features
        # embed_vec = [self._external_data.embed_feature(gnx_id, d) for d in node_order] \
        #     if self._is_external_data and self._external_data.is_embed else None
        # data[gnx_id] = (adjacency, gnx_vec, embed_vec, self._labels[gnx_id])
        # TODO ========================== TODO =========================

        ftr_vec = zscore(ftr_vec, axis=0)
        # sample randomly nodes from the gnx
        number_of_samples = min(len(self._labels), int(len(self._gnx) * self._params.TRAIN_PERCENTAGE))
        available_tagged_nodes = [key for key in self._labels]
        shuffle(available_tagged_nodes)
        random_pick = [available_tagged_nodes[i] for i in range(number_of_samples)]

        # built data (node_idx to label)
        for node in random_pick:
            data.append((self._node_to_idx[node], self._labels[node]))  # (row_in_adjacency_for_node, label)
        pickle.dump((data, Tensor(adjacency), Tensor(ftr_vec)), open(pkl_path, "wb"))
        return data, Tensor(adjacency), Tensor(ftr_vec)

    def __getitem__(self, index):
        node_idx, label = self._data[index]
        return node_idx, label

    def __len__(self):
        return len(self._data)


if __name__ == "__main__":
    from dataset.datset_sampler import ImbalancedDatasetSampler

    # ext_train = ExternalData(AidsAllExternalDataParams())
    ds = GnxDataset(DatasetParams())
    # ds = BilinearDataset(AidsDatasetTestParams())
    dl = DataLoader(
        dataset=ds,
        batch_size=64,
        sampler=ImbalancedDatasetSampler(ds)
    )
    print(ds.adjacency_matrix)
    print(ds.ftr_vec)
    for i, (node_idx_, label_) in enumerate(dl):
        print(node_idx_, label_)
    e = 0
