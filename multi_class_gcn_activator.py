from sys import stdout
import nni
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import Module

from dataset.dataset_model import GnxDataset
from dataset.datset_sampler import ImbalancedDatasetSampler
from bokeh.plotting import figure, show
from torch.utils.data import DataLoader, random_split
from collections import Counter
import numpy as np
from params.parameters import GCNActivatorParams

TRAIN_JOB = "TRAIN"
DEV_JOB = "DEV"
TEST_JOB = "TEST"
VALIDATE_JOB = "VALIDATE"
LOSS_PLOT = "loss"
AUC_PLOT = "AUC"
ACCURACY_PLOT = "accuracy"


class MultiClassGCNActivator:
    def __init__(self, gcn_model: Module, params: GCNActivatorParams, data: GnxDataset, nni=False):
        self._nni = nni
        self._dataset = params.DATASET
        self._gpu = torch.cuda.is_available()
        self._device = torch.device("cuda: 1" if self._gpu else "cpu")
        self._dataset = params.DATASET
        self._gcn_model = gcn_model
        self._epochs = params.EPOCHS
        self._batch_size = params.BATCH_SIZE
        self._loss_func = params.LOSS
        self._load_data(data, params.TRAIN_SIZE, params.BATCH_SIZE)
        self._classes = data.all_labels
        self._init_loss_and_acc_vec()
        self._init_print_att()

    # init loss and accuracy vectors (as function of epochs)
    def _init_loss_and_acc_vec(self):
        self._loss_vec_train = []
        self._loss_vec_dev = []
        self._loss_vec_test = []

        self._accuracy_vec_train = []
        self._accuracy_vec_dev = []
        self._accuracy_vec_test = []

        self._auc_vec_train = []
        self._auc_vec_dev = []
        self._auc_vec_test = []

    # init variables that holds the last update for loss and accuracy
    def _init_print_att(self):
        self._print_train_accuracy = 0
        self._print_train_loss = 0
        self._print_train_auc = 0

        self._print_dev_accuracy = 0
        self._print_dev_loss = 0
        self._print_dev_auc = 0

        self._print_test_accuracy = 0
        self._print_test_loss = 0
        self._print_test_auc = 0

    # update loss after validating
    def _update_loss(self, loss, job=TRAIN_JOB):
        if job == TRAIN_JOB:
            self._loss_vec_train.append(loss)
            self._print_train_loss = loss
        elif job == DEV_JOB:
            self._loss_vec_dev.append(loss)
            self._print_dev_loss = loss
        elif job == TEST_JOB:
            self._loss_vec_test.append(loss)
            self._print_test_loss = loss

    # update accuracy after validating
    def _update_auc(self, pred, true, job=TRAIN_JOB):
        auc = 0
        for curr_class in range(len(self._classes)):
            single_class_true = [1 if t == curr_class else 0 for t in true]
            single_class_pred = [p[curr_class] for p in pred]

            num_classes = len(Counter(single_class_true))
            if num_classes < 2:
                auc += 0.5
            # calculate acc
            else:
                auc += roc_auc_score(single_class_true, single_class_pred)
        auc /= len(self._classes)

        if job == TRAIN_JOB:
            self._print_train_auc = auc
            self._auc_vec_train.append(auc)
            return auc
        elif job == DEV_JOB:
            self._print_dev_auc = auc
            self._auc_vec_dev.append(auc)
            return auc
        elif job == TEST_JOB:
            self._print_test_auc = auc
            self._auc_vec_test.append(auc)
            return auc

    # update accuracy after validating
    def _update_accuracy(self, pred, true, job=TRAIN_JOB):
        # calculate acc
        acc = sum([1 if int(i) == int(j) else 0 for i, j in zip(pred, true)]) / len(pred)
        if job == TRAIN_JOB:
            self._print_train_accuracy = acc
            self._accuracy_vec_train.append(acc)
            return acc
        elif job == DEV_JOB:
            self._print_dev_accuracy = acc
            self._accuracy_vec_dev.append(acc)
            return acc
        elif job == TEST_JOB:
            self._print_test_accuracy = acc
            self._accuracy_vec_test.append(acc)
            return acc

    # print progress of a single epoch as a percentage
    def _print_progress(self, batch_index, len_data, job=""):
        prog = int(100 * (batch_index + 1) / len_data)
        stdout.write("\r\r\r\r\r\r\r\r" + job + " %d" % prog + "%")
        print("", end="\n" if prog == 100 else "")
        stdout.flush()

    # print last loss and accuracy
    def _print_info(self, jobs=()):
        if TRAIN_JOB in jobs:
            print("Acc_Train: " + '{:{width}.{prec}f}'.format(self._print_train_accuracy, width=6, prec=4) +
                  " || AUC_Train: " + '{:{width}.{prec}f}'.format(self._print_train_auc, width=6, prec=4) +
                  " || Loss_Train: " + '{:{width}.{prec}f}'.format(self._print_train_loss, width=6, prec=4),
                  end=" || ")
        if DEV_JOB in jobs:
            print("Acc_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_accuracy, width=6, prec=4) +
                  " || AUC_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_auc, width=6, prec=4) +
                  " || Loss_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_loss, width=6, prec=4),
                  end=" || ")
        if TEST_JOB in jobs:
            print("Acc_Test: " + '{:{width}.{prec}f}'.format(self._print_test_accuracy, width=6, prec=4) +
                  " || AUC_Test: " + '{:{width}.{prec}f}'.format(self._print_test_auc, width=6, prec=4) +
                  " || Loss_Test: " + '{:{width}.{prec}f}'.format(self._print_test_loss, width=6, prec=4),
                  end=" || ")
        print("")

    # plot loss / accuracy graph
    def plot_line(self, job=LOSS_PLOT):
        p = figure(plot_width=600, plot_height=250, title=self._dataset + " - Dataset - " + job,
                   x_axis_label="epochs", y_axis_label=job)
        color1, color2, color3 = ("yellow", "orange", "red") if job == LOSS_PLOT else ("black", "green", "blue")
        if job == LOSS_PLOT:
            y_axis_train = self._loss_vec_train
            y_axis_dev = self._loss_vec_dev
            y_axis_test = self._loss_vec_test
        elif job == AUC_PLOT:
            y_axis_train = self._auc_vec_train
            y_axis_dev = self._auc_vec_dev
            y_axis_test = self._auc_vec_test
        elif job == ACCURACY_PLOT:
            y_axis_train = self._accuracy_vec_train
            y_axis_dev = self._accuracy_vec_dev
            y_axis_test = self._accuracy_vec_test

        x_axis = list(range(len(y_axis_dev)))
        p.line(x_axis, y_axis_train, line_color=color1, legend="train")
        p.line(x_axis, y_axis_dev, line_color=color2, legend="dev")
        p.line(x_axis, y_axis_test, line_color=color3, legend="test")
        show(p)

    def _plot_acc_dev(self):
        self.plot_line(LOSS_PLOT)
        self.plot_line(AUC_PLOT)
        self.plot_line(ACCURACY_PLOT)

    @property
    def model(self):
        return self._gcn_model

    @property
    def loss_train_vec(self):
        return self._loss_vec_train

    @property
    def accuracy_train_vec(self):
        return self._accuracy_vec_train

    @property
    def auc_train_vec(self):
        return self._auc_vec_train

    @property
    def loss_dev_vec(self):
        return self._loss_vec_dev

    @property
    def accuracy_dev_vec(self):
        return self._accuracy_vec_dev

    @property
    def auc_dev_vec(self):
        return self._auc_vec_dev

    @property
    def loss_test_vec(self):
        return self._loss_vec_test

    @property
    def accuracy_test_vec(self):
        return self._accuracy_vec_test

    @property
    def auc_test_vec(self):
        return self._auc_vec_test

    # load dataset
    def _load_data(self, dataset, train_split, batch_size):
        self._adjacency = dataset.adjacency_matrix
        self._ftr_vec = dataset.ftr_vec
        # calculate lengths off train and dev according to split ~ (0,1)
        len_train = int(len(dataset) * train_split)
        len_dev = len(dataset) - len_train
        # split dataset
        train, dev = random_split(dataset, (len_train, len_dev))

        # set train loader
        self._balanced_train_loader = DataLoader(
            train.dataset,
            batch_size=batch_size,
            sampler=ImbalancedDatasetSampler(train.dataset, indices=train.indices,
                                             num_samples=len(train.indices))
        )
        # set train loader
        self._unbalanced_train_loader = DataLoader(
            train,
            batch_size=len_train,
            shuffle=True
        )

        # set validation loader
        self._dev_loader = DataLoader(
            dev,
            batch_size=len_dev,
        )

    # train a model, input is the enum of the model type
    def train(self, validation_rate=1, show_plot=True, early_stop=False):
        self._init_loss_and_acc_vec()
        # calc number of iteration in current epoch
        len_data = len(self._balanced_train_loader)
        for epoch_num in range(self._epochs):
            if not self._nni:
                print("epoch:\t" + str(epoch_num))
            # calc number of iteration in current epoch
            for batch_index, (node_idx, labels) in enumerate(self._balanced_train_loader):
                if self._gpu:
                    node_idx, labels = node_idx.cuda(), labels.cuda()
                # print progress
                self._gcn_model.train()

                encode, output = self._gcn_model(self._adjacency, self._ftr_vec, node_idx)  # calc output
                loss = self._loss_func(output, labels)                   # calculate loss
                loss.backward()                                          # back propagation
                self._gcn_model.optimizer.step()                         # update weights
                self._gcn_model.zero_grad()                              # zero gradients
                if not self._nni:
                    self._print_progress(batch_index, len_data, job=TRAIN_JOB)

            # validate and print progress
            if epoch_num % validation_rate == 0 or epoch_num == self._epochs - 1:
                self._validate(self._unbalanced_train_loader, job=TRAIN_JOB)
                self._validate(self._dev_loader, job=DEV_JOB)
                if not self._nni:
                    self._print_info(jobs=[TRAIN_JOB, DEV_JOB])

                # /---------------  FOR NNI - INTERMEDIATE RESULT ---------------
                else:  # if self._nni
                    test_acc = self._print_dev_accuracy
                    nni.report_intermediate_result(test_acc)
                # -------------------------  FOR NNI  --------------------------/

            if early_stop and epoch_num > 10 and self._print_test_loss > np.mean(self._loss_vec_train[-10:]):
                break

        # /-----------------  FOR NNI - FINAL RESULT -----------------------
        if self._nni:
            test_acc = self._print_dev_accuracy
            nni.report_final_result(test_acc)
        # -------------------------  FOR NNI  -----------------------------/

        if show_plot:
            self._plot_acc_dev()

    # validation function only the model and the data are important for input, the others are just for print
    def _validate(self, data_loader, job=""):
        # for calculating total loss and accuracy
        loss_count = 0
        true_labels = []
        pred_labels = []
        pred_auc_labels = []

        self._gcn_model.eval()
        # calc number of iteration in current epoch
        len_data = len(data_loader)
        for batch_index, (node_idx, labels) in enumerate(data_loader):
            if self._gpu:
                node_idx, labels = node_idx.cuda(), labels.cuda()
            # print progress
            if not self._nni:
                self._print_progress(batch_index, len_data, job=VALIDATE_JOB)
            encode, output = self._gcn_model(self._adjacency, self._ftr_vec, node_idx)
            # calculate total loss
            loss_count += self._loss_func(output, labels)

            true_labels += labels.tolist()
            pred_labels += output.squeeze(dim=1).argmax(dim=1).tolist()
            pred_auc_labels += output.squeeze(dim=1).tolist()

        # update loss accuracy
        loss = float(loss_count / len(data_loader))
        # pred_labels = [0 if np.isnan(i) else i for i in pred_labels]
        self._update_loss(loss, job=job)
        self._update_accuracy(pred_labels, true_labels, job=job)
        self._update_auc(pred_auc_labels, true_labels, job=job)
        return loss

    def encode_graph(self):
        self._gcn_model.eval()
        # calc number of iteration in current epoch
        all_nodes = torch.Tensor(list(range(self._adjacency.shape[0]))).long()
        encode, output = self._gcn_model(self._adjacency, self._ftr_vec, all_nodes)
        return encode


if __name__ == '__main__':
    from params.parameters import DatasetParams, MultiLevelGCNParams
    from gcn_model import MultiLevelGCN
    ds = GnxDataset(DatasetParams())
    model_ = MultiLevelGCN(MultiLevelGCNParams(ftr_len=ds.len_features, num_classes=ds.num_classes))
    activator = MultiClassGCNActivator(model_, GCNActivatorParams(), ds, nni=False)
    activator.train(show_plot=False)
    encode_ = activator.encode_graph()

