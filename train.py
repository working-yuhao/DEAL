import os
import torch.nn as nn 
import time
from tqdm import tqdm
from utils import *
from model import *

from args import *
from torch_geometric.nn import Node2Vec
from torch.utils.data import DataLoader

# args
args = make_args()

device = torch.device('cuda:'+str(args.cuda) if args.gpu else 'cpu')

print("Device: using ", device)

day_str = date.today().strftime("%d_%b")

neg_num = 1

# node / attr /  inter
theta_list = (0.1,0.85,0.05)
lambda_list = (0.1,0.85,0.05)

print(f'theta_list:{theta_list}')


A, X, A_train, X_train, data, train_ones, val_edges, test_edges, folder, val_labels, gt_labels, nodes_keep = load_datafile(args)


if args.inductive:
    sp_X = convert_sSp_tSp(X).to(device).to_dense()
    sp_attrM = convert_sSp_tSp(X_train).to(device)
    val_labels = A_train[val_edges[:, 0], val_edges[:, 1]].A1
else:
    val_labels = A[val_edges[:, 0], val_edges[:, 1]].A1


init_delta = get_delta(np.stack(A_train.nonzero()),A_train)

def get_train_inputs(data,test_edges,val_edges,batch_size,neg_sample_num=10,undirected=True,inductive=False):
    test_mask = (1-torch.eye(data.dists.shape[0])).bool()
    if not inductive:
        test_mask[test_edges[:,0],test_edges[:,1]]=0
        test_mask[val_edges[:,0],val_edges[:,1]]=0
        if undirected:
            test_mask[test_edges[:,1],test_edges[:,0]]=0
            test_mask[val_edges[:,1],val_edges[:,0]]=0
    test_mask=test_mask.to(data.dists.device)
    filter_dists = data.dists * test_mask
    pos = (filter_dists == 0.5).nonzero()
    filter_dists[pos[:,0],pos[:,1]]=0
    pos = pos.cpu().tolist()
    pos_dict = {}
    for i,j in pos:
        pos_dict[i] = pos_dict.get(i,[])+[j]
        
    neg_dict = {}
    neg = (filter_dists>0.12).nonzero().cpu().tolist()
    for i,j in neg:
        neg_dict[i] = neg_dict.get(i,[])+[j]
    nodes = list(pos_dict.keys())
    random.shuffle(nodes)
    inputs = []
    while True:
        for node in nodes:
            tmp_imput = [node, pos_dict[node], random.sample(neg,neg_sample_num) if len(neg)>neg_sample_num else neg]
            inputs.append(tmp_imput)
            if len(inputs) >= batch_size:
                yield np.array(inputs)
                del inputs[:]
        random.shuffle(nodes)

data_loader = iter(get_train_data(A_train, int(X_train.shape[0] *args.train_ratio),np.vstack((test_edges,val_edges)),args.inductive)) #,neg_num
inputs, labels = next(data_loader)




result_list = []
margin_dict = {}
margin_pairs = {}
best_state_dict = None

print(args)
for repeat in tqdm(range(args.repeat_num)):
    for d in margin_dict:
        margin_dict[d].append([])

    deal = DEAL(args.output_dim, X_train.shape[1], X_train.shape[0], device, args, locals()[args.attr_model])

    optimizer = torch.optim.Adam(deal.parameters(), lr=args.lr) 

    max_val_score = np.zeros(1)
    val_result = np.zeros(2)

    running_loss = 0.0

    time1 = time.time()

    # for epoch in tqdm(range(args.epoch_num)):
    for epoch in range(args.epoch_num):
        # get the inputs; data is a list of [inputs, labels]
        
        inputs, labels = next(data_loader)
        labels = labels.to(device)   

        # zero the parameter gradients
        optimizer.zero_grad()

        #
        # forward + backward + optimize
        #

        loss = deal.default_loss(inputs, labels, data, thetas=theta_list, train_num=int(X_train.shape[0] *args.train_ratio)*2)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        b_num = 5
        if epoch% b_num == b_num-1:   
            avg_loss = running_loss / b_num
            val_scores = tran_eval(deal, val_edges, val_labels,data ,lambdas=lambda_list)
            
            running_loss = 0.0
            val_result = np.vstack((val_result,np.array(val_scores)))
            tmp_max = np.maximum(np.mean(val_scores), max_val_score)
            rprint('[%8d]  val %.4f %.4f' % (epoch + 1, *val_scores))
            if tmp_max > max_val_score:
                max_val_score = tmp_max
                if args.inductive:
                    final_scores = avg_loss, *ind_eval(deal, test_edges, gt_labels,sp_X,nodes_keep ,lambdas=lambda_list)
                else:
                    final_scores = avg_loss, *tran_eval(deal, test_edges, gt_labels,data ,lambdas=lambda_list)
            for tmp_d in margin_dict:
                pairs = margin_pairs[tmp_d]
                margin_dict[tmp_d][repeat].append([deal.node_forward(pairs).mean().item(),deal.attr_forward(pairs,data).mean().item()])
    
    time2 = time.time()
    print()
    print('\033[93mTime used: %.2f\033[0m'%(time2-time1))

    print(f'ROC-AUC:{final_scores[1]:.4f} AP:{final_scores[2]:.4f}')

    # if args.inductive:
    #     print()
    #     print('Evaluate Validation Dataset')
    #     detailed_eval(deal, val_edges, val_labels, sp_X,ind_eval,nodes_keep)
    #     print()
    #     print('Evaluate Test Dataset')
    #     detailed_eval(deal, test_edges,gt_labels,sp_X,ind_eval,nodes_keep)
    # else: 
    #     print()
    #     print('Evaluate Validation Dataset')
    #     detailed_eval(deal, val_edges, val_labels, data,tran_eval,verbose=True)
    #     print()
    #     print('Evaluate Test Dataset')
    #     detailed_eval(deal,test_edges,gt_labels,data,tran_eval,verbose=True)


