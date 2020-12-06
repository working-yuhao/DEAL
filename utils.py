#Some functions may be changed during the extension work of DEAL, I will fix this ASAP.

import torch
import torch.nn as nn 
import matplotlib.pyplot as plt
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datetime import date
import scipy.sparse as sp
import itertools
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
import os 
import pickle


SAMPLE_NUM = 10

class Data:
    x = None
    edge_index =None
    anchorset_id = None
    dists_max = None
    dists_argmax = None
    dists = None
    def __init__(self, x, edge_index, dists_max = None, dists_argmax = None, dists = None):
        self.x = x
        self.edge_index = edge_index
        self.dists_max = dists_max
        self.dists_argmax = dists_argmax
    def copy(self):
        return Data(self.x, self.edge_index,
                    self.dists_max if not self.dists_max is None else None,
                    self.dists_argmax if not self.dists_argmax is None else None,
                    self.dists if not self.dists is None else None)


def get_AdjM(k_hop=1):
    M_old = torch.zeros(nodeNum,nodeNum)
    for i in range(nodeNum):
        if i in adj_dict.keys():
            for neigh in adj_dict[i]:
                M_old[i,neigh] =M_old[i,neigh]+0.3 

    M = M_old.clone()
    for hop in range(k_hop):
        for node in adj_dict.keys():
            for neigh in adj_dict[node]:
                M[node] = torch.max(M_old[neigh]-0.1,M[node])
        M_old = M.clone()
        
    return M*adj_mult+ adj_bias

def edge_index2sp_A(edge_index,node_num):
    A = np.zeros((node_num,node_num))
    A[edge_index[0],edge_index[1]] = 1
    return sp.csr_matrix(A)

def get_STM(adj_dict):
    init_value = 15
    M_old = torch.ones(nodeNum,nodeNum) * init_value
    for node in adj_dict:
        M_old[node,adj_dict[node]] = 1 
        M_old[adj_dict[node],node] = 1
    M = M_old.clone()
    
    for i in range(nodeNum):    
        M[i,i] = 0   
    k_hop = 1
    while M.max().item()==init_value and k_hop<init_value:
        for node in adj_dict.keys():
            for neigh in adj_dict[node]:
                M[node] = torch.min(M[node], M_old[neigh]+1)
        
        M_old = M.clone()
        k_hop+=1

    return M, k_hop
  
def plot_results(data, labels = ['MAP','MRR','F1','P','R']):
    for i in range(len(data)):
        plt.plot(np.arange(len(data[i])).tolist(),data[i].tolist(),label=labels[i] )
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.legend()
    plt.show()
    
USING_CUDA = torch.cuda.is_available()
  
def get_Ms(k_hop=1,alpha = 0.5, mode ='max'):
    M_old = torch.zeros(nodeNum,attriNum)
    for node in attr_dict:
        for a in attr_dict[node]:
            M_old[node,a] = M_old[node,a]+1  # 1 #

    for hop in range(k_hop):
        M = M_old.clone()

        print('hop ', k_hop)
        for node in adj_dict.keys():
            for neigh in adj_dict[node]:
                if mode == 'max':
                    M_old[node] = torch.max(M_old[node], M[neigh]*alpha)
                elif mode == 'sum':
                    M_old[node] = torch.add(M_old[node], M[neigh]*alpha)
                else:
                    print("Error")
                    return
    if USING_CUDA:
        return M_old.cuda()
    return M_old

def get_params_with(string, model):
    return [model.state_dict(keep_vars=True)[key] for key in model.state_dict() if string in key]
    
def get_attr_nodes_dict(attr_dict):
    attr_nodes_dict = {}

    for node in attr_dict:
        attrs = attr_dict[node]
        for a in attrs:
            attr_nodes_dict[a] = attr_nodes_dict.get(a, [])+ [node]
    return attr_nodes_dict
  
def get_inverse(M):
    u,s,v = torch.svd(M)
    return torch.mm(torch.mm(v, torch.diag(s.reciprocal())),u.t())
  
def rprint(s):
    s = str(s)
    print('\r'+s+"",end='')
    
def minmax_scaler(data):
    data_min = data.min(dim=1,keepdim=True)[0]
    data_max = data.max(dim=1,keepdim=True)[0]
    return (data-data_min)/(data_max - data_min)

def normalize(tmp_embs):
    return tmp_embs / tmp_embs.norm(dim=1).unsqueeze(1).repeat(1,tmp_embs.size()[1])
  
def show_counter(counter, start =0, end = -1):
    keys = list(counter.keys())
    keys.sort()
    x = list(range(len(counter)))
    y = [counter[f] for f in counter]
    y.sort()
    plt.scatter(x,y)
    plt.grid(True)
    plt.show()

def counter_filt(counter,t):
    res = []
    for i in counter:
        if counter[i] > t:
            res.append(i)
    return res            

def get_nodes_sp_attr_M(nodes, us_attr_dict):
    index = torch.cat([us_attr_dict[node][0] for node in nodes],dim=1)
    value = torch.cat([us_attr_dict[node][1] for node in nodes])
    if USING_CUDA:
        return torch.sparse.FloatTensor(index, value, torch.Size([nodeNum,attriNum])).cuda()        
    else:
        return torch.sparse.FloatTensor(index, value, torch.Size([nodeNum,attriNum]))

def save_Q(Q,name,folder):
    with open(folder+name, 'w') as f:
        for q in Q:
            f.write(' '.join([str(x) for x in q])+'\n')

def load_Q(name,folder):
    Q = []
    with open(folder+name, 'r') as f:
        for line in f:
            Q.append([int(x) for x in line.strip().split()])
    return Q        

def eval_q(tmp_q, gt, cmodel, tmp_k =20):
    return set(gt).intersection(cmodel.query(tmp_q,k=tmp_k)[1].tolist())

def get_scores(ground_truth,res):
    meanRank = 0
    RR = 0
    F1 = 0
    P = 0
    R = 0


    rank = 0 
    num = 0
    tmpRR = -1
    for j in range(len(res)):
        if res[j] in ground_truth:
            if tmpRR < 0:
                tmpRR = 1/(j+1)
                RR += tmpRR
            num += 1
            rank += num/(j+1)
    tp = len(set(res).intersection(set(ground_truth)))
    P = tp/len(res) 
    R = tp/len(ground_truth)
    F1 += 2*P*R /(P+R+1e-20)
    AP = rank/len(ground_truth)

    print("AP:%.4f\tRR:%.4f\tF1:%.4f\tP:%.4f\tR:%.4f\t"%(AP,RR,F1,P,R))

    return AP,RR,F1,P,R

def eval_model(cmodel,QandN,k=20, use_test = False):
    cmodel.eval()
    scores = cmodel.eval_by_tFile(QandN, k, use_test)

    print("--------RESULT--------")
    print("MAP: %.4f"%scores[0])
    print("MRR: %.4f"%scores[1])
    print("F1: %.4f"%scores[2])
    print("P: %.4f"%scores[3])
    print("R: %.4f"%scores[4])
               
def validation_test(cmodel, v_num = 500, k=20):
    v_nodes = random.sample(list(adj_dict.keys()), v_num)
    v_QandN = [[],[]]
    for n in v_nodes:
        v_QandN[0].append(attr_dict[n])
        v_QandN[1].append(adj_dict[n])

    eval_model(cmodel, QandN = v_QandN, k=k)

def transform_attrM_to_attr_dict(attr_M):
    us_attr_dict = {}
    for node in range(len(attr_M)):
        sp_n = attr_M[node].to_sparse()
        value = sp_n.values()
        index = torch.cat([torch.ones_like(sp_n.indices())*node,sp_n.indices()])
        us_attr_dict[node] = [index, value]
    return us_attr_dict

def nearest_attrs(q_a, attrs, emb):
    q_a = emb[q_a].repeat(len(attrs),1)
    return cmodel.cos(q_a, emb[attrs])

def query(mq,cmodel,gt):
    scores = cmodel.score(mq)
    res = scores.topk(20,largest=False)[1].tolist()
    return set(gt).intersection(res)

def get_inv_adj_dict(adj_dict):
    inverse_adj_dict = {}
    for n in adj_dict:
        for neigh in adj_dict[n]:
            inverse_adj_dict[neigh] = inverse_adj_dict.get(neigh,[]) + [n]
    return inverse_adj_dict

def get_A(adj_dict, nodeNum):
    A = sp.lil_matrix((nodeNum, nodeNum), dtype=np.int8)
    for node in adj_dict:
        # A[adj_dict[node],node] = 1 
        A[node,adj_dict[node]] = 1 
    return A

def save_sp(folder, name, M):
    return sp.save_npz(folder+name+'_sp.npz', M.tocsr())
def load_sp(folder,name):
    return sp.load_npz(folder+name+'_sp.npz')

#
# Please refer to the code of G2G https://github.com/abojchevski/graph2gauss/blob/master/g2g/utils.py#L37
#
# def train_val_test_split_adjacency(A, p_val=0.10, p_test=0.1, seed=0, neg_mul=1,
#                                   every_node=True, connected=False, undirected=False,
#                                  use_edge_cover=True, set_ops=True, asserts=False):


def get_ShortestPathM(A,k_hop):
    tmpA = A.copy().todense()
    for _ in range(k_hop):
        for n in range(len(tmpA)):
            for neigh in A[n].nonzero()[1]:
                neighs = tmpA[neigh].nonzero()[1]
                if(len(neighs)):
                    nz = tmpA[n,neighs].nonzero()
                    nz_values = np.minimum(tmpA[n,neighs], tmpA[neigh,neighs]+1)[nz]
                    tmpA[n,neighs] = tmpA[neigh,neighs]+1
                    tmpA[n,neighs[nz[1]]]= nz_values
    return sp.csr_matrix(tmpA)

def edge_cover(A):
    """
    Approximately compute minimum edge cover.

    Edge cover of a graph is a set of edges such that every vertex of the graph is incident
    to at least one edge of the set. Minimum edge cover is an  edge cover of minimum size.

    Parameters
    ----------
    A : sp.spmatrix
        Sparse adjacency matrix

    Returns
    -------
    edges : array-like, shape [?, 2]
        The edges the form the edge cover
    """

    N = A.shape[0]
    d_in = A.sum(0).A1
    d_out = A.sum(1).A1

    # make sure to include singleton nodes (nodes with one incoming or one outgoing edge)
    one_in = np.where((d_in == 1) & (d_out == 0))[0]
    one_out = np.where((d_in == 0) & (d_out == 1))[0]

    edges = []
    edges.append(np.column_stack((A[:, one_in].argmax(0).A1, one_in)))
    edges.append(np.column_stack((one_out, A[one_out].argmax(1).A1)))
    edges = np.row_stack(edges)

    edge_cover_set = set(map(tuple, edges))
    nodes = set(edges.flatten())

    # greedly add other edges such that both end-point are not yet in the edge_cover_set
    cands = np.column_stack(A.nonzero())
    for u, v in cands[d_in[cands[:, 1]].argsort()]:
        if u not in nodes and v not in nodes and u != v:
            edge_cover_set.add((u, v))
            nodes.add(u)
            nodes.add(v)
        if len(nodes) == N:
            break

    # add a single edge for the rest of the nodes not covered so far
    not_covered = np.setdiff1d(np.arange(N), list(nodes))
    edges = [list(edge_cover_set)]
    not_covered_out = not_covered[d_out[not_covered] > 0]

    if len(not_covered_out) > 0:
        edges.append(np.column_stack((not_covered_out, A[not_covered_out].argmax(1).A1)))

    not_covered_in = not_covered[d_out[not_covered] == 0]
    if len(not_covered_in) > 0:
        edges.append(np.column_stack((A[:, not_covered_in].argmax(0).A1, not_covered_in)))

    edges = np.row_stack(edges)

    # make sure that we've indeed computed an edge_cover
    assert A[edges[:, 0], edges[:, 1]].sum() == len(edges)
    assert len(set(map(tuple, edges))) == len(edges)
    assert len(np.unique(edges)) == N

    return edges

def edges_to_sparse(edges, N, values=None):
    """
    Create a sparse adjacency matrix from an array of edge indices and (optionally) values.

    Parameters
    ----------
    edges : array-like, shape [n_edges, 2]
        Edge indices
    N : int
        Number of nodes
    values : array_like, shape [n_edges]
        The values to put at the specified edge indices. Optional, default: np.ones(.)

    Returns
    -------
    A : scipy.sparse.csr.csr_matrix
        Sparse adjacency matrix

    """
    if values is None:
        values = np.ones(edges.shape[0])

    return sp.coo_matrix((values, (edges[:, 0], edges[:, 1])), shape=(N, N)).tocsr()

def get_hops(A, K):
    """
    Calculates the K-hop neighborhoods of the nodes in a graph.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        The graph represented as a sparse matrix
    K : int
        The maximum hopness to consider.

    Returns
    -------
    hops : dict
        A dictionary where each 1, 2, ... K, neighborhoods are saved as sparse matrices
    """
    hops = {1: A.tolil(), -1: A.tolil()}
    hops[1].setdiag(0)

    for h in range(2, K + 1):
        # compute the next ring
        next_hop = hops[h - 1].dot(A)
        next_hop[next_hop > 0] = 1

        for prev_h in range(1, h):
            next_hop -= next_hop.multiply(hops[prev_h])

        next_hop = next_hop.tolil()
        next_hop.setdiag(0)

        hops[h] = next_hop
        hops[-1] += next_hop

    return hops

def sample_all_hops(hops, nodes=None):
    """
    For each node in nodes samples a single node from all of their neighborhoods.

    Parameters
    ----------
    hops : dict
        A dictionary where each 1, 2, ... K, neighborhoods are saved as sparse matrices
    nodes : array-like, shape [N]
        The nodes to consider

    Returns
    -------
    sampled_nodes : array-like, shape [N, K]
        The sampled nodes.
    """

    N = hops[1].shape[0]

    if nodes is None:
        nodes = np.arange(N)

    return np.vstack((nodes,
                      np.array([[-1 if len(x) == 0 else np.random.choice(x) for x in hops[h].rows[nodes]]
                                for h in hops.keys() if h != -1]),
                      sample_last_hop(hops[-1], nodes)
                      )).T

def to_triplets(sampled_hops, scale_terms):
    """
    Form all valid triplets (pairwise constraints) from a set of sampled nodes in triplets

    Parameters
    ----------
    sampled_hops : array-like, shape [N, K]
       The sampled nodes.
    scale_terms : dict
        The appropriate up-scaling terms to ensure unbiased estimates for each neighbourhood

    Returns
    -------
    triplets : array-like, shape [?, 3]
       The transformed triplets.
    """
    triplets = []
    triplet_scale_terms = []

    for i, j in itertools.combinations(np.arange(1, sampled_hops.shape[1]), 2):
        triplet = sampled_hops[:, [0] + [i, j]]
        triplet = triplet[(triplet[:, 1] != -1) & (triplet[:, 2] != -1)]
        triplet = triplet[(triplet[:, 0] != triplet[:, 1]) & (triplet[:, 0] != triplet[:, 2])]
        triplets.append(triplet)
        triplet_scale_terms.append(scale_terms[i][triplet[:, 1]] * scale_terms[j][triplet[:, 2]])
        # triplet_scale_terms.append(scale_terms[i][triplet[:, 0]] * scale_terms[j][triplet[:, 0]])
    return np.row_stack(triplets), np.concatenate(triplet_scale_terms)

def sample_last_hop(A, nodes):
    """
    For each node in nodes samples a single node from their last (K-th) neighborhood.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse matrix encoding which nodes belong to any of the 1, 2, ..., K-1, neighborhoods of every node
    nodes : array-like, shape [N]
        The nodes to consider

    Returns
    -------
    sampled_nodes : array-like, shape [N]
        The sampled nodes.
    """
    N = A.shape[0]

    sampled = np.random.randint(0, N, len(nodes))

    nnz = A[nodes, sampled].nonzero()[1]
    while len(nnz) != 0:
        new_sample = np.random.randint(0, N, len(nnz))
        sampled[nnz] = new_sample
        nnz = A[nnz, new_sample].nonzero()[1]

    return sampled

def convert_sSp_tSp(x):
    coo = x.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def score_link_prediction(labels, scores):
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def load_data_arrays(folder, file = 'data_arrays.npz'):
    data_arrays = np.load(folder+file,allow_pickle=True)
    return data_arrays.values()


def load_dataset(file_name):
    """Load a graph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'z' : The ground truth class labels
            * Further dictionaries mapping node, class and attribute IDs

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        z = loader.get('labels')

        graph = {
            'A': A,
            'X': X,
            'z': z
        }

        idx_to_node = loader.get('idx_to_node')
        if idx_to_node:
            idx_to_node = idx_to_node.tolist()
            graph['idx_to_node'] = idx_to_node

        idx_to_attr = loader.get('idx_to_attr')
        if idx_to_attr:
            idx_to_attr = idx_to_attr.tolist()
            graph['idx_to_attr'] = idx_to_attr

        idx_to_class = loader.get('idx_to_class')
        if idx_to_class:
            idx_to_class = idx_to_class.tolist()
            graph['idx_to_class'] = idx_to_class

        return graph

def score_node_classification(features, z,train_set,test_set, norm=False):

    if norm:
        features = normalize(features)

    lrcv = LogisticRegressionCV(max_iter= 2000, multi_class='auto')
    lrcv.fit(features[train_set], z[train_set])
    predicted = lrcv.predict(features[test_set])

    f1_micro = f1_score(z[test_set], predicted, average='micro')
    f1_macro = f1_score(z[test_set], predicted, average='macro')

    return f1_micro, f1_macro

def get_delta(edge_index,A):
    a,b= np.unique(edge_index,return_counts=True)
    order = a[np.argsort(b)[::-1]]
    delta = torch.zeros(A.shape[0])
    delta[a] = torch.FloatTensor(b/b.max())
    return delta

def get_train_data(A_train, batch_size,tv_edges,inductive):
    nodes = []
    labels = []
    tmp_A = A_train.tolil()
    nodeNum = A_train.shape[0]
    
    if not inductive:
        forbidden_Matrix = sp.lil_matrix(A_train.shape)
        forbidden_Matrix[tv_edges[:,0],tv_edges[:,1]] = 1
        while True:
            a = random.randint(0,nodeNum-1)
            b = random.randint(0,nodeNum-1)
            if not(forbidden_Matrix[a,b]):
                nodes.append([a,b])
                if tmp_A[a,b]:
                    labels.append(1)
                else:
                    labels.append(0)

            if len(tmp_A.rows[a]):    
                neigh = np.random.choice(tmp_A.rows[a])
                if not(forbidden_Matrix[a,neigh]):
                    nodes.append([a,neigh])
                    labels.append(1)

            if len(labels) >= batch_size:
                yield torch.LongTensor(nodes), torch.LongTensor(labels)
                del nodes[:]
                del labels[:]
    else:
        while True:
            a = random.randint(0,nodeNum-1)
            b = random.randint(0,nodeNum-1)
            nodes.append([a,b])
            if tmp_A[a,b]:
                labels.append(1)
            else:
                labels.append(0)

            if len(tmp_A.rows[a]):    
                neigh = np.random.choice(tmp_A.rows[a])
                nodes.append([a,neigh])
                labels.append(1)

            if len(labels) >= batch_size:
                yield torch.LongTensor(nodes), torch.LongTensor(labels)
                del nodes[:]
                del labels[:]


def convert_triplets(triplets,A):
    final = np.hstack((triplets,A[triplets[:,0],triplets[:,1]].T,np.zeros((len(triplets),1),dtype=np.int8)))
    final = np.random.permutation(final)
    return final

def test_train_data(A_train,batch_size):
    hops = get_hops(A_train, 1)
    scale_terms = {h if h != -1 else max(hops.keys()) + 1:
                           hops[h].sum(1).A1 if h != -1 else hops[1].shape[0] - hops[h].sum(1).A1
                       for h in hops}
    # triplets: node node label
    triplets = convert_triplets(to_triplets(sample_all_hops(hops), scale_terms)[0], A_train)
    while True:
        if len(triplets)> batch_size:
            yield triplets[:batch_size].astype(int)
            triplets = triplets[batch_size:]
        else:
            triplets = np.vstack((triplets,convert_triplets(to_triplets(sample_all_hops(hops), scale_terms)[0], A_train)))
    
def get_us_attr_dict(X):
    us_attr_dict = {}
    for node in range(X.shape[0]):
        coo = X[node].tocoo()
        values = coo.data
        indices = np.vstack((coo.row+node, coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        us_attr_dict[node] = [i, v]
    return us_attr_dict

#inductive
def ind_eval(cmodel, nodes, gt_labels,X,nodes_keep, lambdas = (0,1,1)):

    # anode_emb = torch.sparse.mm(data.x, cmodel.attr_emb(torch.arange(data.x.shape[1]).to(cmodel.device)))
    test_data = Data(X, None)
    anode_emb = cmodel.attr_emb(test_data)

    first_embs = anode_emb[nodes[:,0]]

    sec_embs = anode_emb[nodes[:,1]]
    res = cmodel.attr_layer(first_embs,sec_embs) * lambdas[1]

    node_emb = anode_emb.clone()

    res = res + cmodel.inter_layer(first_embs,node_emb[nodes[:,1]]) * lambdas[2]
    
    if len(res.shape)>1:
        res = res.softmax(dim=1)[:,1]
    res = res.detach().cpu().numpy()
    return score_link_prediction(gt_labels, res)

def tran_eval(cmodel, test_data, gt_labels,data, lambdas = (1,1,1)):
    res = cmodel.evaluate(test_data, data, lambdas)
    if len(res.shape)>1:
        res = res.softmax(dim=1)[:,1]
    res = res.detach().cpu().numpy()
    return score_link_prediction(gt_labels, res)
    
def detailed_eval(model,test_data,gt_labels,sp_M, evaluate,nodes_keep=None, verbose=False, lambdas=(1,1,1)):
    setting = {}

    setting['Full '] = lambdas
    setting['Inter'] = (0,0,1)
    if lambdas[1]:
        setting['Attr '] = (0,1,0)
    if lambdas[0]:
        setting['Node '] = (1,0,0)
    
    res = {}
    for s in setting:
        if not nodes_keep is None:
            if s != 'Node ':
                res[s] = evaluate(model, test_data, gt_labels,sp_M,nodes_keep,setting[s])
                if verbose:
                    print( s+' ROC-AUC:%.4f AP:%.4f'%res[s])
        else:            
            res[s] = evaluate(model, test_data, gt_labels,sp_M,setting[s])
            if verbose:
                print( s+' ROC-AUC:%.4f AP:%.4f'%res[s])
    return res

def load_datafile(args):
    nodes_keep = None
    device = torch.device('cuda:'+str(args.cuda) if args.gpu else 'cpu')

    folder = './data/'+args.dataset+'/'
    
    if args.inductive:
        data_array_file =  folder+'pv0.10_pt0.00_pn0.10_arrays.npz'
    else:
        data_array_file = folder+'data_arrays_'+args.task+'.npz'
        
    if os.path.exists(folder+'A_sp.npz'):
        A = load_sp(folder,'A')
        X = load_sp(folder,'X')
        z = np.load(folder+'z.npy')
        with np.load(data_array_file, allow_pickle=True) as data_arrays:
            train_ones,val_ones, val_zeros, test_ones, test_zeros = data_arrays.values()
    else:
        assert False, "No data file"
 
    if args.task == 'node':
        # symmetrical
        A = A.maximum(A.T)
        #~#
        A_train = A
        X_train = X

    if args.inductive:
        A_train = sp.load_npz(folder+'ind_train_A.npz')
        X_train = sp.load_npz(folder+'ind_train_X.npz')
        nodes_keep = np.unique(np.load(folder+'nodes_keep.npy'))
    else:
        A_train = edges_to_sparse(train_ones,X.shape[0])
        X_train = X

    if not os.path.exists(folder+'trained_models/'):
        os.mkdir(folder+'trained_models/')

    hops = get_hops(A_train, 1)

    scale_terms = {h if h != -1 else max(hops.keys()) + 1:
                           hops[h].sum(1).A1 if h != -1 else hops[1].shape[0] - hops[h].sum(1).A1
                       for h in hops}

    test_edges = np.row_stack((test_ones, test_zeros))
    val_edges = np.row_stack((val_ones, val_zeros))

    gt_labels = A[test_edges[:, 0], test_edges[:, 1]].A1
    ## test_ground_truth = torch.LongTensor(1-gt_labels) * 2 

    if args.inductive:
        sp_X = convert_sSp_tSp(X).to(device).to_dense()
        sp_attrM = convert_sSp_tSp(X_train).to(device)
        # us_attr_dict = get_us_attr_dict(X_train)
        val_labels = A_train[val_edges[:, 0], val_edges[:, 1]].A1
    else:
        # sp_attrM = convert_sSp_tSp(X).to(device)
        # us_attr_dict = get_us_attr_dict(X)
        val_labels = A[val_edges[:, 0], val_edges[:, 1]].A1
    
    data = Data(convert_sSp_tSp(X_train).to_dense().to(device), torch.LongTensor(train_ones.T).to(device))

    data.dists = load_dists(args.dataset)
     
    if not data.dists is None:
        data.dists = data.dists.to(device)
        preselect_anchor(data, layer_num=args.layer_num, anchor_num=64, device=device)

    return A, X, A_train, X_train, data, train_ones, val_edges, test_edges, folder, val_labels, gt_labels, nodes_keep

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

    anchorset_id = get_random_anchorset(data.x.shape[0],c=1)
    data.anchorset_id = anchorset_id
    data.dists_max, data.dists_argmax = get_dist_max(anchorset_id, data.dists, device)


def load_dists(dataset):
    dists_file_name = 'data/' + dataset +'/dists-1.dat'
    if os.path.exists(dists_file_name):
        dists_file = open(dists_file_name, 'rb')
        return pickle.load(dists_file)
    else:
        dists_file_name = 'data/' + dataset +'/dists2.dat'
        if os.path.exists(dists_file_name):
            return pickle.load(dists_file)
        else:
            return None

