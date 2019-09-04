from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
import community
import networkx as nx


def community_detection(gnx: nx.Graph, node_order, encoding, true_labels, alpha: float = 1):
    node_to_idx = {str(node): i for i, node in enumerate(node_order)}
    similarity_matrix = cosine_similarity(encoding.tolist())
    for u in gnx.nodes():
        for v in gnx.nodes():
            if (u, v) in gnx.edges():
                # 1 * (1 - alpha) + alpha * similarity
                gnx.edges()[(u, v)]['weight'] = (1 - alpha) + \
                                                (alpha * abs(similarity_matrix[node_to_idx[u], node_to_idx[v]]))
            else:
                # 0 * (1 - alpha) + alpha * similarity
                gnx.add_edge(u, v)
                gnx.edges()[(u, v)]['weight'] = (alpha * abs(similarity_matrix[node_to_idx[u], node_to_idx[v]]))

    partition = community.best_partition(gnx)
    score = normalized_mutual_info_score([partition[n] for n in node_order], true_labels)
    return partition, score


def encode_by_labels(dataset, gcn_params, activator_params):
    model = MultiLevelGCN(gcn_params(ftr_len=dataset.len_features, num_classes=dataset.num_classes))
    activator = MultiClassGCNActivator(model, activator_params(), dataset, nni=False)
    activator.train(show_plot=False, validation_rate=10, early_stop=True)
    encodings = activator.encode_graph()
    return encodings


if __name__ == "__main__":
    from gcn_model import MultiLevelGCN
    from dataset.dataset_model import GnxDataset
    from multi_class_gcn_activator import MultiClassGCNActivator
    from params.parameters import DatasetParams, MultiLevelGCNParams, GCNActivatorParams
    from params.cora_params import CoraDatasetParams, CoraMultiLevelGCNParams, CoraGCNActivatorParams
    # from params.eu_email_params import EuEmailDatasetParams, EuEmailMultiLevelGCNParams, EuEmailGCNActivatorParams

    ds_ = GnxDataset(CoraDatasetParams())
    true_labels_ = [ds_.label_by_node_name(node) for node in ds_.node_order]

    encodings_ = encode_by_labels(ds_, CoraMultiLevelGCNParams, CoraGCNActivatorParams)

    scores_ = {}
    for alpha_ in [i / 10 for i in range(1, 11)]:
        partition_, score_ = community_detection(ds_.gnx, ds_.node_order, encodings_, true_labels_, alpha=alpha_)
        scores_[alpha_] = score_
        print(alpha_, score_)

    for alpha_, score_ in scores_.items():
        print(alpha_, score_)

