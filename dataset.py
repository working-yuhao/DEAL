import torch
import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import torch.utils.data
import itertools
from collections import Counter
from random import shuffle
import json
#
from networkx.readwrite import json_graph
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import pdb
import time
import random
import pickle
import os.path
import torch_geometric as tg
import torch_geometric.datasets
import time

from torch_geometric.data import Data, DataLoader

import multiprocessing as mp
import torch.nn.functional as F

def get_pred(args, model, data, edges):
    if args.model == 'G2G':
        pred = -model.module.energy_kl(data,edges)
    else:    
        out = model(data)
        # get_link_mask(data,resplit=False)  # resample negative links
        nodes_first = torch.index_select(out, 0, torch.from_numpy(edges[0,:]).long().to(out.device))
        nodes_second = torch.index_select(out, 0, torch.from_numpy(edges[1,:]).long().to(out.device))
        pred = torch.sum(nodes_first * nodes_second, dim=-1)
    return pred

def tri_loss(data, model, args, device):
        selected_nodes = data.edge_index.unique()
        num_nodes = len(selected_nodes)
        b_size = int(num_nodes*2)
        index = torch.LongTensor(3*b_size).random_(0, len(selected_nodes)).to(selected_nodes.device)
        triplets = selected_nodes[index].view(-1,3)
        sign = torch.sign(data.dists[triplets[:,0],triplets[:,1]] - data.dists[triplets[:,0],triplets[:,2]])
        if args.model == 'G2G':
            pos_edges = triplets[:,[0,1]].t()
            neg_edges = triplets[:,[0,2]].t()
            return F.relu(sign.mul(model.module.energy_kl(data,pos_edges)+torch.exp(-model.module.energy_kl(data,neg_edges)))).sum()
            # return sign.matmul(model.module.energy_kl(data,pos_edges)+torch.exp(-model.module.energy_kl(data,neg_edges)))
        else:    
            emb = model(data)
            first_embs = emb[triplets[:,0]]
            sec_embs = emb[triplets[:,1]]
            third_embs = emb[triplets[:,2]]
            return -sign.matmul(torch.mul(first_embs,sec_embs).sum(dim=1)-torch.mul(first_embs,third_embs).sum(dim=1))

# # approximate
def get_edge_mask_link_negative_approximate(mask_link_positive, num_nodes, num_negtive_edges):
    links_temp = np.zeros((num_nodes, num_nodes)) + np.identity(num_nodes)
    mask_link_positive = duplicate_edges(mask_link_positive)
    links_temp[mask_link_positive[0],mask_link_positive[1]] = 1
    # add random noise
    links_temp += np.random.rand(num_nodes,num_nodes)
    prob = num_negtive_edges / (num_nodes*num_nodes-mask_link_positive.shape[1])
    mask_link_negative = np.stack(np.nonzero(links_temp<prob))
    return mask_link_negative


# exact version, slower
def get_edge_mask_link_negative(mask_link_positive, num_nodes, num_negtive_edges):
    mask_link_positive_set = []
    for i in range(mask_link_positive.shape[1]):
        mask_link_positive_set.append(tuple(mask_link_positive[:,i]))
    mask_link_positive_set = set(mask_link_positive_set)

    mask_link_negative = np.zeros((2,num_negtive_edges), dtype=mask_link_positive.dtype)
    for i in range(num_negtive_edges):
        while True:
            mask_temp = tuple(np.random.choice(num_nodes,size=(2,),replace=False))
            if mask_temp not in mask_link_positive_set:
                mask_link_negative[:,i] = mask_temp
                break
    return mask_link_negative



def resample_edge_mask_link_negative(data):
    data.mask_link_negative_train = get_edge_mask_link_negative(data.mask_link_positive_train, num_nodes=data.num_nodes,
                                                      num_negtive_edges=data.mask_link_positive_train.shape[1])
    data.mask_link_negative_val = get_edge_mask_link_negative(data.mask_link_positive, num_nodes=data.num_nodes,
                                                      num_negtive_edges=data.mask_link_positive_val.shape[1])
    data.mask_link_negative_test = get_edge_mask_link_negative(data.mask_link_positive, num_nodes=data.num_nodes,
                                                     num_negtive_edges=data.mask_link_positive_test.shape[1])
    # nums = [data.mask_link_positive_train.shape[1],data.mask_link_positive_val.shape[1],data.mask_link_positive_test.shape[1]]
    # neg_list = get_edge_mask_hard_neg(data.mask_link_positive,nums, 4, num_nodes=data.num_nodes)
    # data.mask_link_negative_train = neg_list[0]
    # data.mask_link_negative_val = neg_list[1]
    # data.mask_link_negative_test = neg_list[2]

def get_edge_mask_hard_neg(pos_edges, nums, hard, num_nodes): 
    #
    # hard>0: turn on hard mode
    #
    dist_matrix = sp.lil_matrix((num_nodes, num_nodes))
    total_num = pos_edges.shape[1]
    if hard > 0:
        graph = nx.Graph()
        edge_list = pos_edges.transpose(1,0).tolist()
        graph.add_edges_from(edge_list)
        dists_dict = all_pairs_shortest_path_length_parallel(graph,cutoff= min(hard,5))
        for node in dists_dict:
            dist_matrix[node,list(dists_dict[node].keys())]= np.array(list(dists_dict[node].values()))+1
    else:
        dist_matrix[pos_edges[0],pos_edges[1]] = 1
        dist_matrix[np.arange(num_nodes),np.arange(num_nodes)] = 1

    neg_edges = np.zeros((2,total_num), dtype=pos_edges.dtype)
    if hard > 0:
        for i in range(total_num):
            while True:
                mask_temp = tuple(np.random.choice(num_nodes,size=(2,),replace=False))
                if dist_matrix[mask_temp[0],mask_temp[1]]>0:
                    neg_edges[:,i] = mask_temp
                    dist_matrix[mask_temp[0],mask_temp[1]]= 0
                    break
    else:
        for i in range(total_num):
            while True:
                mask_temp = tuple(np.random.choice(num_nodes,size=(2,),replace=False))
                if dist_matrix[mask_temp[0],mask_temp[1]]<1:
                    neg_edges[:,i] = mask_temp
                    dist_matrix[mask_temp[0],mask_temp[1]]= 1
                    break
    results = []
    tmp_s = 0
    for num in nums:
        results.append(neg_edges[:,tmp_s:tmp_s+num])
        tmp_s += num
    return results

def deduplicate_edges(edges):
    edges_new = np.zeros((2,edges.shape[1]//2), dtype=int)
    # add none self edge
    j = 0
    skip_node = set() # node already put into result
    for i in range(edges.shape[1]):
        if edges[0,i]<edges[1,i]:
            edges_new[:,j] = edges[:,i]
            j += 1
        elif edges[0,i]==edges[1,i] and edges[0,i] not in skip_node:
            edges_new[:,j] = edges[:,i]
            skip_node.add(edges[0,i])
            j += 1

    return edges_new

def duplicate_edges(edges):
    return np.concatenate((edges, edges[::-1,:]), axis=-1)


# each node at least remain in the new graph
def split_edges(edges, remove_ratio, connected=False):
    e = edges.shape[1]
    edges = edges[:, np.random.permutation(e)]
    if connected:
        unique, counts = np.unique(edges, return_counts=True)
        node_count = dict(zip(unique, counts))

        index_train = []
        index_val = []
        for i in range(e):
            node1 = edges[0,i]
            node2 = edges[1,i]
            if node_count[node1]>1 and node_count[node2]>1: # if degree>1
                index_val.append(i)
                node_count[node1] -= 1
                node_count[node2] -= 1
                if len(index_val) == int(e * remove_ratio):
                    break
            else:
                index_train.append(i)
        index_train = index_train + list(range(i + 1, e))
        index_test = index_val[:len(index_val)//2]
        index_val = index_val[len(index_val)//2:]

        edges_train = edges[:, index_train]
        edges_val = edges[:, index_val]
        edges_test = edges[:, index_test]
    else:
        split1 = int((1-remove_ratio)*e)
        split2 = int((1-remove_ratio/2)*e)
        edges_train = edges[:,:split1]
        edges_val = edges[:,split1:split2]
        edges_test = edges[:,split2:]

    return edges_train, edges_val, edges_test

def edge_to_set(edges):
    edge_set = []
    for i in range(edges.shape[1]):
        edge_set.append(tuple(edges[:, i]))
    edge_set = set(edge_set)
    return edge_set


def get_link_mask(data, remove_ratio=0.2, resplit=True, infer_link_positive=True):
    if resplit:
        if infer_link_positive:
            data.mask_link_positive = deduplicate_edges(data.edge_index.numpy())
        data.mask_link_positive_train, data.mask_link_positive_val, data.mask_link_positive_test = \
            split_edges(data.mask_link_positive, remove_ratio)
    resample_edge_mask_link_negative(data)


def add_nx_graph(data):
    G = nx.Graph()
    edge_numpy = data.edge_index.numpy()
    edge_list = []
    for i in range(data.num_edges):
        edge_list.append(tuple(edge_numpy[:, i]))
    G.add_edges_from(edge_list)
    data.G = G

def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
    return dists_dict

def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result

def all_pairs_shortest_path_length_parallel(graph,cutoff=None,num_workers=4):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    if len(nodes)<50:
        num_workers = int(num_workers/4)
    elif len(nodes)<400:
        num_workers = int(num_workers/2)

    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(single_source_shortest_path_length_range,
            args=(graph, nodes[int(len(nodes)/num_workers*i):int(len(nodes)/num_workers*(i+1))], cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict


def precompute_dist_data(edge_index, num_nodes, approximate=0):
        '''
        Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
        :return:
        '''
        graph = nx.Graph()
        edge_list = edge_index.transpose(1,0).tolist()
        graph.add_edges_from(edge_list)

        n = num_nodes
        dists_array = np.zeros((n, n))
        # dists_dict = nx.all_pairs_shortest_path_length(graph,cutoff=approximate if approximate>0 else None)
        # dists_dict = {c[0]: c[1] for c in dists_dict}
        dists_dict = all_pairs_shortest_path_length_parallel(graph,cutoff=approximate if approximate>0 else None)
        for i, node_i in enumerate(graph.nodes()):
            shortest_dist = dists_dict[node_i]
            for j, node_j in enumerate(graph.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist!=-1:
                    # dists_array[i, j] = 1 / (dist + 1)
                    dists_array[node_i, node_j] = 1 / (dist + 1)
        return dists_array



def get_random_anchorset(n,c=0.5):
    m = int(np.log2(n))
    copy = int(c*m)
    anchorset_id = []
    for i in range(m):
        anchor_size = int(n/np.exp2(i + 1))
        for j in range(copy):
            anchorset_id.append(np.random.choice(n,size=anchor_size,replace=False))
    return anchorset_id

def get_dist_max(anchorset_id, dist, device):
    dist_max = torch.zeros((dist.shape[0],len(anchorset_id))).to(device)
    dist_argmax = torch.zeros((dist.shape[0],len(anchorset_id))).long().to(device)
    for i in range(len(anchorset_id)):
        temp_id = anchorset_id[i]
        dist_temp = dist[:, temp_id]
        dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
        dist_max[:,i] = dist_max_temp
        # dist_argmax[:,i] = dist_argmax_temp
        dist_argmax[:,i] = torch.LongTensor(temp_id).to(device)[dist_argmax_temp]
    return dist_max, dist_argmax


def preselect_anchor(data, layer_num=1, anchor_num=32, anchor_size_num=4, device='cpu'):

    # data.anchor_size_num = anchor_size_num
    # data.anchor_set = []
    # anchor_num_per_size = anchor_num//anchor_size_num
    # for i in range(anchor_size_num):
    #     anchor_size = 2**(i+1)-1
    #     anchors = np.random.choice(data.num_nodes, size=(layer_num,anchor_num_per_size,anchor_size), replace=True)
    #     data.anchor_set.append(anchors)
    # data.anchor_set_indicator = np.zeros((layer_num, anchor_num, data.num_nodes), dtype=int)

    anchorset_id = get_random_anchorset(data.num_nodes,c=1)
    data.anchorset_id = anchorset_id

    data.dists_max, data.dists_argmax = get_dist_max(anchorset_id, data.dists, device)


def get_tg_dataset(args, dataset_name, use_cache=True, remove_feature=False):
    # "Cora", "CiteSeer" and "PubMed"
    print('Start getting data')
    if dataset_name not in ['grid','communities','protein','email','ppi']:
        if dataset_name in ['Cora','CiteSeer','PubMed']:
            dataset = tg.datasets.Planetoid(root='datasets/' + dataset_name, name=dataset_name)
        elif dataset_name == 'CoraFull':
            dataset = tg.datasets.CoraFull(root='datasets/' + dataset_name)
        elif dataset_name in ['CS','Physics']:
            dataset = tg.datasets.Coauthor(root='datasets/' + dataset_name, name=dataset_name)
        elif dataset_name in ['Photo', 'Computers']:
            dataset = tg.datasets.Amazon(root='datasets/' + dataset_name, name=dataset_name)
        elif dataset_name == 'PPI':
            dataset = tg.datasets.PPI(root='datasets/' + dataset_name)
        elif dataset_name == 'Reddit':
            dataset = tg.datasets.Reddit(root='datasets/' + dataset_name)
        else:
            assert False, 'Error: No dataset'
    else:
        try:
            dataset = load_tg_dataset(dataset_name)
        except:
            raise NotImplementedError

    # precompute shortest path
    if not os.path.isdir('datasets/cache'):
        os.mkdir('datasets/cache')
    f1_name = 'datasets/cache/' + dataset_name + str(args.approximate) + '_dists.dat'
    f2_name = 'datasets/cache/' + dataset_name + str(args.approximate)+ '_dists_removed.dat'
    f3_name = 'datasets/cache/' + dataset_name + str(args.approximate)+ '_links_train.dat'
    f4_name = 'datasets/cache/' + dataset_name + str(args.approximate)+ '_links_val.dat'
    f5_name = 'datasets/cache/' + dataset_name + str(args.approximate)+ '_links_test.dat'
    feature_name = 'datasets/cache/' + dataset_name +'_X.dat'
    neg_links_name = 'datasets/cache/' + dataset_name +'_neg_links.dat'

    if use_cache and ((os.path.isfile(f2_name) and args.task=='link') or (os.path.isfile(f1_name) and args.task!='link')):
        with open(f3_name, 'rb') as f3, \
            open(f4_name, 'rb') as f4, \
            open(f5_name, 'rb') as f5:
            links_train_list = pickle.load(f3)
            links_val_list = pickle.load(f4)
            links_test_list = pickle.load(f5)
        if args.task=='link':
            with open(f2_name, 'rb') as f2:
                dists_removed_list = pickle.load(f2)
        else:
            with open(f1_name, 'rb') as f1:
                dists_list = pickle.load(f1)

        print('Cache loaded!')
        data_list = []
        for i, data in enumerate(dataset):
            if args.task == 'link':
                data.mask_link_positive = deduplicate_edges(data.edge_index.numpy())
            data.mask_link_positive_train = links_train_list[i]
            data.mask_link_positive_val = links_val_list[i]
            data.mask_link_positive_test = links_test_list[i]
            get_link_mask(data, resplit=False)

            if args.task=='link':
                data.dists = torch.from_numpy(dists_removed_list[i]).float()
                data.edge_index = torch.from_numpy(duplicate_edges(data.mask_link_positive_train)).long()
            else:
                data.dists = torch.from_numpy(dists_list[i]).float()
            if remove_feature:
                data.x = torch.ones((data.x.shape[0],1))
            data_list.append(data)
    else:
        data_list = []
        dists_list = []
        dists_removed_list = []
        links_train_list = []
        links_val_list = []
        links_test_list = []
        feature_list = []
        graph_list = []
        neg_links_list = []
                # for i, data in enumerate(dataset):
        data = dataset[0]
        i = 0
        if 'link' in args.task:
            get_link_mask(data, args.remove_link_ratio, resplit=True,
                          infer_link_positive=True if args.task == 'link' else False)
        links_train_list.append(data.mask_link_positive_train)
        links_val_list.append(data.mask_link_positive_val)
        links_test_list.append(data.mask_link_positive_test)

        if args.task=='link':
            dists_removed = precompute_dist_data(data.mask_link_positive_train, data.num_nodes,
                                                 approximate=args.approximate)
            --lossremoved_list.append(dists_removed)
            data.dists = torch.from_numpy(dists_removed).float()
            # data.edge_index = torch.from_numpy(duplicate_edges(data.mask_link_positive_train)).long()
        else:
            dists = precompute_dist_data(data.edge_index.numpy(), data.num_nodes, approximate=args.approximate)
            dists_list.append(dists)
            data.dists = torch.from_numpy(dists).float()

        if remove_feature:
            data.x = torch.ones((data.x.shape[0],1))
        data_list.append(data)
        feature_list.append(data.x)
        neg_links_list.append(np.concatenate([data.mask_link_negative_train,data.mask_link_negative_val,data.mask_link_negative_test],axis=1))
        with open(f3_name, 'wb') as f3, \
            open(f4_name, 'wb') as f4, \
            open(f5_name, 'wb') as f5, \
            open(feature_name, 'wb') as feature_file, \
            open(neg_links_name, 'wb') as neg_file:

            
            if args.task=='link':
                with open(f2_name, 'wb') as f2:
                    pickle.dump(dists_removed_list, f2)
            else:
                with open(f1_name, 'wb') as f1:
                    pickle.dump(dists_list, f1)
            pickle.dump(links_train_list, f3)
            pickle.dump(links_val_list, f4)
            pickle.dump(links_test_list, f5)
            pickle.dump(feature_list, feature_file)
            pickle.dump(neg_links_list, neg_file)
        print('Cache saved!')
    return data_list


def nx_to_tg_data(graphs, features, edge_labels=None):
    data_list = []
    for i in range(len(graphs)):
        feature = features[i]
        graph = graphs[i].copy()
        graph.remove_edges_from(nx.selfloop_edges(graph))

        # relabel graphs
        keys = list(graph.nodes)
        vals = range(graph.number_of_nodes())
        mapping = dict(zip(keys, vals))
        nx.relabel_nodes(graph, mapping, copy=False)

        x = np.zeros(feature.shape)
        graph_nodes = list(graph.nodes)
        for m in range(feature.shape[0]):
            x[graph_nodes[m]] = feature[m]
        x = torch.from_numpy(x).float()

        # get edges
        edge_index = np.array(list(graph.edges))
        edge_index = np.concatenate((edge_index, edge_index[:,::-1]), axis=0)
        edge_index = torch.from_numpy(edge_index).long().permute(1,0)

        data = Data(x=x, edge_index=edge_index)
        # get edge_labels
        if edge_labels[0] is not None:
            edge_label = edge_labels[i]
            mask_link_positive = np.stack(np.nonzero(edge_label))
            data.mask_link_positive = mask_link_positive
        data_list.append(data)
    return data_list



def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def Graph_load_batch(min_num_nodes = 20, max_num_nodes = 1000, name = 'ENZYMES',node_attributes = True,graph_labels=True):
    '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
    print('Loading graph dataset: '+str(name))
    G = nx.Graph()
    # load data
    path = 'data/'+name+'/'
    data_adj = np.loadtxt(path+name+'_A.txt', delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(path+name+'_node_attributes.txt', delimiter=',')
    data_node_label = np.loadtxt(path+name+'_node_labels.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(path+name+'_graph_indicator.txt', delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(path+name+'_graph_labels.txt', delimiter=',').astype(int)


    data_tuple = list(map(tuple, data_adj))

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i+1, feature = data_node_att[i])
        G.add_node(i+1, label = data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0])+1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator==i+1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['label'] = data_graph_labels[i]
        if G_sub.number_of_nodes()>=min_num_nodes and G_sub.number_of_nodes()<=max_num_nodes:
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
    print('Loaded')
    return graphs, data_node_att, data_node_label



# main data load function
def load_graphs(dataset_str):
    edge_labels = [None]

    if dataset_str == 'grid':
        graphs = []
        features = []
        for _ in range(1):
            graph = nx.grid_2d_graph(20, 20)
            graph = nx.convert_node_labels_to_integers(graph)

            feature = np.identity(graph.number_of_nodes())
            graphs.append(graph)
            features.append(feature)

    elif dataset_str == 'communities':
        graphs = []
        features = []
        edge_labels = []
        for i in range(1):
            community_size = 20
            community_num = 20
            p=0.01

            graph = nx.connected_caveman_graph(community_num, community_size)

            count = 0

            for (u, v) in graph.edges():
                if random.random() < p:  # rewire the edge
                    x = random.choice(list(graph.nodes))
                    if graph.has_edge(u, x):
                        continue
                    graph.remove_edge(u, v)
                    graph.add_edge(u, x)
                    count += 1
            print('rewire:', count)

            n = graph.number_of_nodes()
            label = np.zeros((n,n),dtype=int)
            for u in list(graph.nodes):
                for v in list(graph.nodes):
                    if u//community_size == v//community_size and u>v:
                        label[u,v] = 1
            rand_order = np.random.permutation(graph.number_of_nodes())
            feature = np.identity(graph.number_of_nodes())[:,rand_order]
            graphs.append(graph)
            features.append(feature)
            edge_labels.append(label)

    elif dataset_str == 'protein':

        graphs_all, features_all, labels_all = Graph_load_batch(name='PROTEINS_full')
        features_all = (features_all-np.mean(features_all,axis=-1,keepdims=True))/np.std(features_all,axis=-1,keepdims=True)
        graphs = []
        features = []
        edge_labels = []
        for graph in graphs_all:
            n = graph.number_of_nodes()
            label = np.zeros((n, n),dtype=int)
            for i,u in enumerate(graph.nodes()):
                for j,v in enumerate(graph.nodes()):
                    if labels_all[u-1] == labels_all[v-1] and u>v:
                        label[i,j] = 1
            if label.sum() > n*n/4:
                continue

            graphs.append(graph)
            edge_labels.append(label)

            idx = [node-1 for node in graph.nodes()]
            feature = features_all[idx,:]
            features.append(feature)

        print('final num', len(graphs))


    elif dataset_str == 'email':

        with open('data/email.txt', 'rb') as f:
            graph = nx.read_edgelist(f)

        label_all = np.loadtxt('data/email_labels.txt')
        graph_label_all = label_all.copy()
        graph_label_all[:,1] = graph_label_all[:,1]//6


        for edge in list(graph.edges()):
            if graph_label_all[int(edge[0])][1] != graph_label_all[int(edge[1])][1]:
                graph.remove_edge(edge[0], edge[1])

        comps = [comp for comp in nx.connected_components(graph) if len(comp)>10]
        graphs = [graph.subgraph(comp) for comp in comps]

        edge_labels = []
        features = []

##
        total_num = 0
        for g in graphs:
            total_num += g.number_of_nodes()
        total_features = np.identity(total_num)
##
        start = 0
        for g in graphs:
            n = g.number_of_nodes()
            feature = total_features[start:start+n]
            start = start+n
            features.append(feature)

            label = np.zeros((n, n),dtype=int)
            for i, u in enumerate(g.nodes()):
                for j, v in enumerate(g.nodes()):
                    if label_all[int(u)][1] == label_all[int(v)][1] and i>j:
                        label[i, j] = 1
            label = label
            edge_labels.append(label)

        # for g in graphs:
        #     n = g.number_of_nodes()
        #     feature = np.ones((n, 1))
        #     features.append(feature)

        #     label = np.zeros((n, n),dtype=int)
        #     for i, u in enumerate(g.nodes()):
        #         for j, v in enumerate(g.nodes()):
        #             if label_all[int(u)][1] == label_all[int(v)][1] and i>j:
        #                 label[i, j] = 1
        #     label = label
        #     edge_labels.append(label)

    elif dataset_str == 'ppi':
        dataset_dir = 'data/ppi'
        print("Loading data...")
        G = json_graph.node_link_graph(json.load(open(dataset_dir + "/ppi-G.json")))
        edge_labels_internal = json.load(open(dataset_dir + "/ppi-class_map.json"))
        edge_labels_internal = {int(i): l for i, l in edge_labels_internal.items()}

        train_ids = [n for n in G.nodes()]
        train_labels = np.array([edge_labels_internal[i] for i in train_ids])
        if train_labels.ndim == 1:
            train_labels = np.expand_dims(train_labels, 1)

        print("Using only features..")
        feats = np.load(dataset_dir + "/ppi-feats.npy")
        ## Logistic gets thrown off by big counts, so log transform num comments and score
        feats[:, 0] = np.log(feats[:, 0] + 1.0)
        feats[:, 1] = np.log(feats[:, 1] - min(np.min(feats[:, 1]), -1))
        feat_id_map = json.load(open(dataset_dir + "/ppi-id_map.json"))
        feat_id_map = {int(id): val for id, val in feat_id_map.items()}
        train_feats = feats[[feat_id_map[id] for id in train_ids]]

        node_dict = {}
        for id,node in enumerate(G.nodes()):
            node_dict[node] = id

        comps = [comp for comp in nx.connected_components(G) if len(comp)>10]
        graphs = [G.subgraph(comp) for comp in comps]

        id_all = []
        for comp in comps:
            id_temp = []
            for node in comp:
                id = node_dict[node]
                id_temp.append(id)
            id_all.append(np.array(id_temp))

        features = [train_feats[id_temp,:]+0.1 for id_temp in id_all]

    else:
        raise NotImplementedError

    return graphs, features, edge_labels


def load_tg_dataset(name='communities'):
    graphs, features, edge_labels = load_graphs(name)
    return nx_to_tg_data(graphs, features, edge_labels)