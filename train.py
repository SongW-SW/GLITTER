import networkx as nx
import numpy as np
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
import sys
import scipy
import sklearn
import json
from collections import defaultdict
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math
import pickle as pkl
import scipy.sparse as sp
import time
import scipy.io as sio
import random
from sklearn import preprocessing
from sklearn.metrics import f1_score
# import contrast_util
import json
import os
from tqdm import tqdm
import csv
from base_model_influence import GCN_dense
from base_model_influence import Linear
from base_model_influence import GCN_emb


# from base_model import GCN

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    f1 = f1_score(labels, preds, average='weighted')
    return f1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def cal_euclidean(input1, input2, normalize=False):
    # input tensor
    # a = input.unsqueeze(0).repeat([input.shape[0], 1, 1])
    # b = input.unsqueeze(1).repeat([1, input.shape[0], 1])
    # distance = (a - b).square().sum(-1)
    if not normalize:
        distance = torch.cdist(input1.unsqueeze(0), input2.unsqueeze(0)).squeeze()
    else:
        input1 = input1 / torch.norm(input1, dim=-1, keepdim=True)
        input2 = input2 / torch.norm(input2, dim=-1, keepdim=True)
        distance = torch.cdist(input1.unsqueeze(0), input2.unsqueeze(0)).squeeze()

    return distance


def load_npz_to_sparse_graph(file_name):
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph : SparseGraph
        Graph in sparse matrix format.
    """
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape'])
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

        node_names = loader.get('node_names')
        attr_names = loader.get('attr_names')
        class_names = loader.get('class_names')
        metadata = loader.get('metadata')

    return adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata


def loadCSV(csvf):
    dictGraphsLabels = {}
    dictLabels = {}
    dictGraphs = {}

    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            filename = row[1]
            g_idx = int(filename.split('_')[0])
            label = row[2]
            # append filename to current label

            if g_idx in dictGraphs.keys():
                dictGraphs[g_idx].append(filename)
            else:
                dictGraphs[g_idx] = [filename]
                dictGraphsLabels[g_idx] = {}

            if label in dictGraphsLabels[g_idx].keys():
                dictGraphsLabels[g_idx][label].append(filename)
            else:
                dictGraphsLabels[g_idx][label] = [filename]

            if label in dictLabels.keys():
                dictLabels[label].append(filename)
            else:
                dictLabels[label] = [filename]
    return dictLabels, dictGraphs, dictGraphsLabels


G_Meta_datasets = ['fold_PPI', 'tissue_PPI']
valid_num_dic = {'dblp': 27}


def load_data_pretrain(dataset_source):
    if dataset_source == 'ogbn-arxiv':

        from ogb.nodeproppred import NodePropPredDataset
        dataset = NodePropPredDataset(name=dataset_source)

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, labels = dataset[0]  # graph: library-agnostic graph object

        n1s = graph['edge_index'][0]
        n2s = graph['edge_index'][1]

        num_nodes = graph['num_nodes']
        print('nodes num', num_nodes)
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                            shape=(num_nodes, num_nodes))

        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        features = torch.FloatTensor(graph['node_feat'])
        labels = torch.LongTensor(labels).squeeze()

        # class_list_test = random.sample(list(range(40)),20)
        # train_class=list(set(list(range(40))).difference(set(class_list_test)))
        # class_list_valid = random.sample(train_class, 5)
        # class_list_train = list(set(train_class).difference(set(class_list_valid)))
        # json.dump([class_list_train,class_list_valid,class_list_test],open('./few_shot_data/{}_class_split.json'.format(dataset_source),'w'))
        class_list_train, class_list_valid, class_list_test = json.load(
            open('./few_shot_data/{}_class_split.json'.format(dataset_source)))

        idx_train, idx_valid, idx_test = [], [], []

        for i in range(labels.shape[0]):
            if labels[i] in class_list_train:
                idx_train.append(i)
            elif labels[i] in class_list_valid:
                idx_valid.append(i)
            else:
                idx_test.append(i)
        print(labels.shape)

    elif dataset_source == 'dblp':
        n1s = []
        n2s = []
        for line in open("./few_shot_data/{}_network".format(dataset_source)):
            n1, n2 = line.strip().split('\t')
            n1s.append(int(n1))
            n2s.append(int(n2))

        data_train = sio.loadmat("./few_shot_data/{}_train.mat".format(dataset_source))
        data_test = sio.loadmat("./few_shot_data/{}_test.mat".format(dataset_source))

        num_nodes = max(max(n1s), max(n2s)) + 1
        labels = np.zeros((num_nodes, 1))
        labels[data_train['Index']] = data_train["Label"]
        labels[data_test['Index']] = data_test["Label"]

        features = np.zeros((num_nodes, data_train["Attributes"].shape[1]))
        features[data_train['Index']] = data_train["Attributes"].toarray()
        features[data_test['Index']] = data_test["Attributes"].toarray()

        print('nodes num', num_nodes)
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                            shape=(num_nodes, num_nodes))

        class_list = []
        for cla in labels:
            if cla[0] not in class_list:
                class_list.append(cla[0])  # unsorted

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels):
            id_by_class[cla[0]].append(id)

        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(labels)

        adj = normalize(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])

        adj = sparse_mx_to_torch_sparse_tensor(adj)

        train_class = list(set(data_train["Label"].reshape((1, len(data_train["Label"])))[0]))
        class_list_test = list(set(data_test["Label"].reshape((1, len(data_test["Label"])))[0]))

        class_list_valid = random.sample(train_class, valid_num_dic[dataset_source])

        class_list_train = list(set(train_class).difference(set(class_list_valid)))

        class_list_train, class_list_valid, class_list_test = json.load(
            open('./few_shot_data/{}_class_split.json'.format(dataset_source)))

        idx_train, idx_valid, idx_test = [], [], []
        for idx_, class_list_ in zip([idx_train, idx_valid, idx_test],
                                     [class_list_train, class_list_valid, class_list_test]):
            for class_ in class_list_:
                idx_.extend(id_by_class[class_])

    elif dataset_source == 'cora-full':
        adj, features, labels, node_names, attr_names, class_names, metadata = load_npz_to_sparse_graph(
            './few_shot_data/cora_full.npz')

        sparse_mx = adj.tocoo().astype(np.float32)
        indices = np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)

        n1s = indices[0].tolist()
        n2s = indices[1].tolist()

        adj = normalize(adj.tocoo() + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        features = features.todense()
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels).squeeze()

        # features=features.todense()

        # class_list_test = random.sample(list(range(70)),25)
        # train_class=list(set(list(range(70))).difference(set(class_list_test)))
        # class_list_valid = random.sample(train_class, 20)
        # class_list_train = list(set(train_class).difference(set(class_list_valid)))
        # json.dump([class_list_train,class_list_valid,class_list_test],open('./few_shot_data/{}_class_split.json'.format(dataset_source),'w'))

        class_list_train, class_list_valid, class_list_test = json.load(
            open('./few_shot_data/{}_class_split.json'.format(dataset_source)))

        idx_train, idx_valid, idx_test = [], [], []

        for i in range(labels.shape[0]):
            if labels[i] in class_list_train:
                idx_train.append(i)
            elif labels[i] in class_list_valid:
                idx_valid.append(i)
            else:
                idx_test.append(i)


    class_train_dict = defaultdict(list)
    for one in class_list_train:
        for i, label in enumerate(labels.numpy().tolist()):
            if label == one:
                class_train_dict[one].append(i)

    class_test_dict = defaultdict(list)
    for one in class_list_test:
        for i, label in enumerate(labels.numpy().tolist()):
            if label == one:
                class_test_dict[one].append(i)

    print(len(idx_train))
    print(len(idx_train) + len(idx_valid))
    print(features.shape[0])

    # for single graph, make it a list

    return [adj], [features], [class_train_dict], [class_test_dict]


def neighborhoods_(adj, n_hops, use_cuda):
    """Returns the n_hops degree adjacency matrix adj."""
    # adj = torch.tensor(adj, dtype=torch.float)
    # adj=adj.to_dense()
    # print(type(adj))
    if use_cuda:
        adj = adj.cuda()
    # hop_adj = power_adj = adj

    # return (adj@(adj.to_dense())+adj).to_dense().cpu().numpy().astype(int)

    hop_adj = adj + torch.sparse.mm(adj, adj)

    hop_adj = hop_adj.to_dense()
    # hop_adj = (hop_adj > 0).to_dense()

    # for i in range(n_hops - 1):
    # power_adj = power_adj @ adj
    # prev_hop_adj = hop_adj
    # hop_adj = hop_adj + power_adj
    # hop_adj = (hop_adj > 0).float()

    hop_adj = hop_adj.cpu().numpy().astype(int)

    return (hop_adj > 0).astype(int)

    # return hop_adj.cpu().numpy().astype(int)


def neighborhoods(adj, n_hops, use_cuda):
    """Returns the n_hops degree adjacency matrix adj."""
    # adj = torch.tensor(adj, dtype=torch.float)
    # adj=adj.to_dense()
    # print(type(adj))
    if n_hops == 1:
        return adj.cpu().numpy().astype(int)

    if use_cuda:
        adj = adj.cuda()
    # hop_adj = power_adj = adj

    # for i in range(n_hops - 1):
    # power_adj = power_adj @ adj
    hop_adj = adj + adj @ adj
    hop_adj = (hop_adj > 0).float()

    np.save(hop_adj.cpu().numpy().astype(int), './neighborhoods_{}.npy'.format(dataset))

    return hop_adj.cpu().numpy().astype(int)


def InforNCE_Loss(anchor, sample, tau, all_negative=False):
    def _similarity(h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return h1 @ h2.t()

    assert anchor.shape[0] == sample.shape[0]

    pos_mask = torch.eye(anchor.shape[0], dtype=torch.float).cuda()
    neg_mask = 1. - pos_mask

    sim = _similarity(anchor, sample) / tau
    exp_sim = torch.exp(sim) * (pos_mask + neg_mask)

    if not all_negative:
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
    else:
        log_prob = - torch.log(exp_sim.sum(dim=1, keepdim=True))

    loss = log_prob * pos_mask
    loss = loss.sum(dim=1) / pos_mask.sum(dim=1)

    return -loss.mean()


class Predictor(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout=0):
        super(Predictor, self).__init__()
        self.linear1 = Linear(nfeat, nhid)
        self.linear2 = Linear(nhid, nout)

    def forward(self, x):
        return self.linear2(self.linear1(x).relu())


def calculate_att(construct_graph_feat, construct_graph_adj, w1, b1, w2, b2, k=10):
    attention = torch.exp(
        -cal_euclidean(F.relu(linear1(construct_graph_feat, w1, b1)), F.relu(linear2(construct_graph_feat, w2, b2)),
                       normalize=True))
    k = 1
    topk_values, topk_indices = torch.topk(attention, k, dim=-1)
    temp = attention - topk_values[:, -1].unsqueeze(-1)
    attention = torch.where(temp > 0, attention, torch.zeros(attention.shape).cuda())
    attention = attention * 0.5 + construct_graph_adj.cuda()
    attention /= (attention.sum(-1, keepdim=True) + 1e-9)
    return attention


parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', action='store_true', default=True, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--test_epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--pretrain_lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,  # 5e-4
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--pretrain_dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', default='dblp',
                    help='Dataset:Amazon_clothing/Amazon_eletronics/dblp')

args = parser.parse_args(args=[])

# args.use_cuda = torch.cuda.is_available()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.use_cuda:
    torch.cuda.manual_seed(args.seed)

# Load data


N = 5
K = 3
query_size = 1
fine_tune_steps=5

fine_tune_lr = 0.1

contrast_loss_weight = 0.2
mi_loss_weight = 0.2

args.epochs = 8000
args.test_epochs = 50


# ['dblp', 'cora-full', 'ogbn-arxiv']
for dataset in ['dblp']:

    adj_sparse, features, class_train_dict, class_test_dict = load_data_pretrain(
        dataset)

    args.hidden1 = features[0].shape[-1]

    adjs = []
    for i in range(len(adj_sparse)):
        adjs.append(adj_sparse[i].to_dense())

    if dataset == 'ogbn-arxiv':
        args.pretrain_dropout = 0
        args.epochs = 5000
        args.pretrain_lr = 0.005
        fine_tune_steps = 40
        fine_tune_lr = 0.1

    if args.dataset == 'dblp' and N == 10:
        fine_tune_steps = 20  # 20

    if args.use_cuda:
        for i in range(len(features)):
            features[i] = features[i].cuda()
            if dataset != 'ogbn-arxiv':
                features[i] -= features[i].min(0, keepdim=True)[0]
                features[i] /= (features[i].max(0, keepdim=True)[0] + 1e-9)
        for i in range(len(adj_sparse)):
            if dataset != 'ogbn-arxiv':
                adjs[i] = adjs[i].cuda()

    if dataset == 'tissue_PPI':
        N = 2
        K = 5
        query_size = 5
        args.pretrain_lr = 0.005
        fine_tune_steps = 20
        fine_tune_lr = 0.01

    if dataset == 'fold_PPI':
        N = 3


    print(dataset)
    print('N={},K={}'.format(N, K))

    model = GCN_dense(nfeat=args.hidden1,
                      nhid=args.hidden2,
                      dropout=args.pretrain_dropout)

    if dataset != 'tissue_PPI':
        loss_f = nn.CrossEntropyLoss()
        classifier = Linear(args.hidden2, N)
    else:
        loss_f = nn.BCELoss(reduction='mean')
        classifier = Linear(args.hidden2, 1)

    linear1 = Linear(args.hidden1, args.hidden2)
    linear2 = Linear(args.hidden1, args.hidden2)

    optimizer = optim.Adam(
        [{'params': model.parameters()}, {'params': classifier.parameters()}, {'params': linear1.parameters()},
         {'params': linear2.parameters()}],
        lr=args.pretrain_lr, weight_decay=args.weight_decay)

    cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    labels = torch.zeros([N * K])

    for i in range(N):
        labels[i * K:(i + 1) * K] = i

    query_labels = []
    for i in range(N):
        query_labels.extend([i] * query_size)
    query_labels = torch.tensor(query_labels)

    if dataset != 'tissue_PPI':
        labels = labels.type(torch.LongTensor)
    else:
        query_labels = query_labels.type(torch.FloatTensor)

    if args.use_cuda:
        model.cuda()
        classifier = classifier.cuda()
        linear1 = linear1.cuda()
        linear2 = linear2.cuda()
        labels = labels.cuda()
        query_labels = query_labels.cuda()


    def pre_train(epoch, mode='train', graph_id=0):
        if len(adjs) > 1:
            if mode == 'train':
                select_graph_dict = class_train_dict
            else:
                select_graph_dict = class_test_dict

            graph_id = np.random.choice(list(select_graph_dict.keys()), 1, replace=False)[0]
            while len(select_graph_dict[graph_id]) < N:
                graph_id = np.random.choice(list(select_graph_dict.keys()), 1, replace=False)[0]

        start_time = time.time()

        adj = adjs[graph_id]
        feature = features[graph_id]

        # emb_features = GCN_model(features, adj_sparse)
        emb_features = feature

        if mode == 'train':
            class_dict = class_train_dict[graph_id]
            for i in class_dict:
                class_dict[i] = class_dict[i]
        else:
            class_dict = class_test_dict[graph_id]

        if dataset != 'tissue_PPI':
            classes = np.random.choice(list(class_dict.keys()), N, replace=False).tolist()
        else:
            if mode == 'test':
                classes = np.random.choice(list(class_dict.keys()), N, replace=False).tolist()
            else:
                k = np.random.choice([8, 2, 5, 6, 3, 1, 0], 1)[0]
                classes = ['0.0' + str(k), '1.0' + str(k)]

        target_idx = []
        pos_node_idx = []

        for i in classes:

            pos_node_idx.extend(np.random.choice(class_dict[i], K, replace=False).tolist())

            ############################################################################################
            # sample query nodes
            for j in range(query_size):
                while True:
                    idx = np.random.choice(class_dict[i], 1, replace=False)[0]
                    if idx in pos_node_idx or idx in target_idx:
                        continue
                    else:
                        target_idx.append(idx)
                        break
            ############################################################################################

        # construct computation graph
        # pos_node_idx is a list containing NK nodes
        construct_graph_neighbors = torch.nonzero(adj[pos_node_idx + target_idx, :].sum(0)).squeeze()

        # including 2-hop neighbors
        if dataset != 'tissue_PPI':
            construct_graph_neighbors = torch.nonzero(adj[construct_graph_neighbors, :].sum(0)).squeeze()

        # including 3-hop neighbors
        # construct_graph_neighbors = torch.nonzero(adj[construct_graph_neighbors, :].sum(0)).squeeze()


        temp = construct_graph_neighbors.cpu().numpy().tolist()
        # make sure the first NK nodes are labeled (support nodes)
        for idx in pos_node_idx:
            while idx in temp:
                temp.remove(idx)
        temp = pos_node_idx + temp

        # put all query nodes to the last several ones
        for idx in target_idx:
            while idx in temp:
                temp.remove(idx)
        temp = temp + target_idx

        construct_graph_nodes = temp

        construct_graph_adj = adj[construct_graph_nodes, :][:, construct_graph_nodes]
        construct_graph_feat = emb_features[construct_graph_nodes]
        # construct_graph_adj_and_feat = [construct_graph_adj, construct_graph_feat]

        # attention=torch.exp(-cal_euclidean(pos_graph_feat)*500)

        if mode == 'train':
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()

        # first round of fine tune
        gc1_w, gc1_b, gc2_w, gc2_b, w, b, w1, b1, w2, b2 = model.gc1.weight, model.gc1.bias, model.gc2.weight, model.gc2.bias, classifier.weight, classifier.bias, linear1.weight, linear1.bias, linear2.weight, linear2.bias

        for j in range(fine_tune_steps):
            W = calculate_att(construct_graph_feat, construct_graph_adj, w1, b1, w2, b2)
            output_feat = model(construct_graph_feat, W, gc1_w, gc1_b, gc2_w, gc2_b)

            loss_contrast = 0
            support_feat = output_feat[:N * K, :].reshape([N, K, -1])
            for i in range(K):
                loss_contrast += InforNCE_Loss(support_feat[:, i, :], support_feat[:, (i + 1) % K, :],
                                               tau=0.1) / K

            if dataset == 'tissue_PPI':
                logits = torch.sigmoid(classifier(output_feat[:N * K], w, b).squeeze())
                loss_supervised = loss_f(logits, labels)
            else:
                loss_supervised = loss_f(classifier(output_feat[:N * K], w, b).squeeze(), labels)

            Q = W[N * K:, N * K:]
            Q_square = Q.matmul(Q)

            A_tilde = W[N * K:, :N * K]

            P = Q.matmul(A_tilde) + Q_square.matmul(A_tilde)

            P_query_to_sup = P[-N * query_size:]  # [query, support]

            P_rearrange = P_query_to_sup.reshape([N * query_size, N, K])
            P_sum = P_rearrange.sum(-1)
            P_norm = P_sum / P_sum.sum(-1, keepdim=True)

            loss_mi = 0

            loss_mi += (P_norm.sum(0) * torch.log(P_norm.sum(0))).sum()
            loss_mi -= (P_norm * torch.log(P_norm)).sum(-1).mean(0)

            loss = loss_supervised + loss_contrast * contrast_loss_weight + loss_mi * mi_loss_weight

            grad = torch.autograd.grad(loss, [gc1_w, gc1_b, gc2_w, gc2_b, w, b, w1, b1, w2, b2],
                                       allow_unused=True)

            # one step of fine tune
            gc1_w, gc1_b, gc2_w, gc2_b, w, b, w1, b1, w2, b2 = list(
                map(lambda p: p[1] - fine_tune_lr * p[0],
                    zip(grad, [gc1_w, gc1_b, gc2_w, gc2_b, w, b, w1, b1, w2, b2])))

            # print(grad)
            if torch.isnan(grad[0]).sum() > 0:
                print('---------------------')
                print(grad)
                print(1 / 0)

        model.eval()

        W = calculate_att(construct_graph_feat, construct_graph_adj, w1, b1, w2, b2)

        output_feat = model(construct_graph_feat, W, gc1_w, gc1_b, gc2_w, gc2_b)

        loss_contrast = 0
        support_feat = output_feat[:N * K, :].reshape([N, K, -1])
        for i in range(K):
            loss_contrast += InforNCE_Loss(support_feat[:, i, :], support_feat[:, (i + 1) % K, :], tau=0.1) / K

        logits = classifier(output_feat[-N * query_size:], w, b).squeeze()

        if dataset == 'tissue_PPI':
            logits = torch.sigmoid(logits)

        loss = loss_f(logits, query_labels) + loss_contrast * contrast_loss_weight
        if mode == 'train':
            loss.backward()
            optimizer.step()

        if dataset == 'tissue_PPI':
            logits = torch.stack([1 - logits, logits], -1)

        if epoch % 50 == 0 and mode == 'train':
            tqdm.write(' '.join(['Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss.item()),  # 'loss_dis: {:.4f}'.format(dis_loss.item()),
                  'loss_contrast: {:.4f}'.format(loss_contrast.item()),
                  'acc_train: {:.4f}'.format((torch.argmax(logits, -1) == query_labels).float().mean().item()),
                  'time: {:.4f}'.format(time.time() - start_time)]))

        return (torch.argmax(logits, -1) == query_labels).float().mean().item()


    t_total = time.time()
    best_acc = 0
    for epoch in tqdm(range(args.epochs)):
        acc_train = pre_train(epoch)
        if epoch % 500 == 499:
            acc = 0
            for epoch_test in range(args.test_epochs):
                acc_ = pre_train(epoch_test, mode='test')
                acc += acc_
                del acc_

            acc /= args.test_epochs

            if acc > best_acc: best_acc = acc

            tqdm.write('Final Test Acc: {:.4f}  Best Acc: {:.4f} '.format(acc, best_acc))

            del acc


    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    torch.save(model.state_dict(),
               './saved_models/{}_{}_epochs.pth'.format(dataset, args.epochs))

    del model
    del adjs

