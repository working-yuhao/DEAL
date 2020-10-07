import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
import torch_geometric as tg
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import init

####################### Basic Ops #############################

# # PGNN layer, only pick closest node for message passing
class PGNN_layer(nn.Module):
    def __init__(self, input_dim, output_dim,dist_trainable=True):
        super(PGNN_layer, self).__init__()
        self.input_dim = input_dim
        self.dist_trainable = dist_trainable

        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, output_dim, 1)

        self.linear_hidden = nn.Linear(input_dim*2, output_dim)
        self.linear_out_position = nn.Linear(output_dim,1)
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, feature, dists_max, dists_argmax):
        if self.dist_trainable:
            dists_max = self.dist_compute(dists_max.unsqueeze(-1)).squeeze()

        subset_features = feature[dists_argmax.flatten(), :]
        subset_features = subset_features.reshape((dists_argmax.shape[0], dists_argmax.shape[1],
                                                   feature.shape[1]))
        messages = subset_features * dists_max.unsqueeze(-1)

        self_feature = feature.unsqueeze(1).repeat(1, dists_max.shape[1], 1)
        messages = torch.cat((messages, self_feature), dim=-1)

        messages = self.linear_hidden(messages).squeeze()
        messages = self.act(messages) # n*m*d 

        out_position = self.linear_out_position(messages).squeeze(-1)  # n*m_out
        out_structure = torch.mean(messages, dim=1)  # n*d

        return out_position, out_structure


### Non linearity
class Nonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Nonlinear, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x




####################### NNs #############################

class MLP(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(MLP, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.linear_first = nn.Linear(feature_dim, hidden_dim)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out = nn.Linear(hidden_dim, output_dim)


    def forward(self, data):
        x = data.x
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.linear_first(x)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class GCN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GCN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.GCNConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.GCNConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.GCNConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x

class SAGE(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(SAGE, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.SAGEConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.SAGEConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.SAGEConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.SAGEConv(hidden_dim, output_dim)

    def forward(self, data):
        device = self.conv_out.weight.device
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x

class GAT(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GAT, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.GATConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.GATConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.GATConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.GATConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x

class GIN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GIN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first_nn = nn.Linear(feature_dim, hidden_dim)
            self.conv_first = tg.nn.GINConv(self.conv_first_nn)
        else:
            self.conv_first_nn = nn.Linear(input_dim, hidden_dim)
            self.conv_first = tg.nn.GINConv(self.conv_first_nn)
        self.conv_hidden_nn = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_hidden = nn.ModuleList([tg.nn.GINConv(self.conv_hidden_nn[i]) for i in range(layer_num - 2)])

        self.conv_out_nn = nn.Linear(hidden_dim, output_dim)
        self.conv_out = tg.nn.GINConv(self.conv_out_nn)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)            
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x



class PGNN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(PGNN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if layer_num == 1:
            hidden_dim = output_dim
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = PGNN_layer(feature_dim, hidden_dim)
        else:
            self.conv_first = PGNN_layer(input_dim, hidden_dim)
        if layer_num>1:
            self.conv_hidden = nn.ModuleList([PGNN_layer(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
            self.conv_out = PGNN_layer(hidden_dim, output_dim)

    def forward(self, data):
        x = data.x
        if self.feature_pre:
            x = self.linear_pre(x)
        x_position, x = self.conv_first(x, data.dists_max, data.dists_argmax)
        if self.layer_num == 1:
            return x #x_position
        # x = F.relu(x) # Note: optional!
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            _, x = self.conv_hidden[i](x, data.dists_max, data.dists_argmax)
            # x = F.relu(x) # Note: optional!
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x_position, x = self.conv_out(x, data.dists_max, data.dists_argmax)
        x_position = F.normalize(x_position, p=2, dim=-1)
        return x #x_position


def pearsonr(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val
    
class Hidden_Layer(nn.Module): #Hidden Layer, Binary classification
        
    def __init__(self, emb_dim, device,BCE_mode, mode='all', dropout_p = 0.3):
        super(Hidden_Layer, self).__init__()
        self.emb_dim = emb_dim
        self.mode = mode
        self.device = device
        self.BCE_mode = BCE_mode
        self.Linear1 = nn.Linear(self.emb_dim*2, self.emb_dim).to(self.device)
        self.Linear2 = nn.Linear(self.emb_dim, 32).to(self.device)
        x_dim = 1
        self.Linear3 = nn.Linear(32, x_dim).to(self.device)
        if self.mode == 'all':
            if self.BCE_mode:
                self.linear_output = nn.Linear(x_dim+ 3, 1).to(self.device)
            else:
                self.linear_output = nn.Linear(x_dim+ 3, 2).to(self.device)
        else:
            self.linear_output = nn.Linear(1, 2).to(self.device) 
            self.linear_output.weight.data[1,:] = 1
            self.linear_output.weight.data[0,:] = -1

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.pdist = nn.PairwiseDistance(p=2,keepdim=True)       
        self.softmax = nn.Softmax(dim=1)
        self.elu = nn.ELU()
        assert (self.mode in ['all','cos','dot','pdist']),"Wrong mode type"


    def forward(self, f_embs, s_embs):

        if self.mode == 'all':
            x = torch.cat([f_embs,s_embs],dim=1)
            x = F.rrelu(self.Linear1(x))
            x = F.rrelu(self.Linear2(x))
            x = F.rrelu(self.Linear3(x))
            cos_x = self.cos(f_embs,s_embs).unsqueeze(1)
            dot_x = torch.mul(f_embs,s_embs).sum(dim=1,keepdim=True)
            pdist_x = self.pdist(f_embs,s_embs)
            x = torch.cat([x,cos_x,dot_x,pdist_x],dim=1)
        elif self.mode == 'cos':
            x = self.cos(f_embs,s_embs).unsqueeze(1)
        elif self.mode == 'dot':
            x = torch.mul(f_embs,s_embs).sum(dim=1,keepdim=True)
        elif self.mode == 'pdist':
            x = self.pdist(f_embs,s_embs)

        if self.BCE_mode:
            return x.squeeze()
            # return (x/x.max()).squeeze()
        else:
            x = self.linear_output(x)
            x = F.rrelu(x)
            # x = torch.cat((x,-x),dim=1)
            return x
    
    def evaluate(self, f_embs, s_embs):
        if self.mode == 'all':
            x = torch.cat([f_embs,s_embs],dim=1)
            x = F.rrelu(self.Linear1(x))
            x = F.rrelu(self.Linear2(x))
            x = F.rrelu(self.Linear3(x))
            cos_x = self.cos(f_embs,s_embs).unsqueeze(1)
            dot_x = torch.mul(f_embs,s_embs).sum(dim=1,keepdim=True)
            pdist_x = self.pdist(f_embs,s_embs)
            x = torch.cat([x,cos_x,dot_x,pdist_x],dim=1)
        elif self.mode == 'cos':
            x = self.cos(f_embs,s_embs)
        elif self.mode == 'dot':
            x = torch.mul(f_embs,s_embs).sum(dim=1)
        elif self.mode == 'pdist':
            x = -self.pdist(f_embs,s_embs).squeeze()
        return x


class Emb(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=False, layer_num=2, dropout=0, **kwargs):
        super(Emb, self).__init__()
        self.attr_emb = nn.Embedding(input_dim , output_dim)
        self.attr_num = input_dim

    def forward(self, data):
        x = data.x
        x = torch.mm(x, self.attr_emb(torch.arange(self.attr_num).to(self.attr_emb.weight.device)))
        return x


class DEAL(nn.Module):

    def __init__(self, emb_dim, attr_num, node_num,device, args,attr_emb_model ,h_layer=Hidden_Layer, num_classes=0 ,feature_dim=64,dropout_p = 0.3, verbose=False):
        super(DEAL, self).__init__()
        n_hidden=args.layer_num
        self.device = device
        self.mode = args.train_mode
        self.node_num = node_num
        self.attr_num = attr_num
        self.emb_dim = emb_dim
        self.verbose = verbose
        self.BCE_mode = args.BCE_mode
        self.gamma = args.gamma
        self.s_a = args.strong_A

        self.num_classes = num_classes
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.pdist = nn.PairwiseDistance(p=2,keepdim=True)       
        self.softmax = nn.Softmax(dim=1)


        self.dropout = nn.Dropout(p=dropout_p)
        if self.BCE_mode:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        if self.num_classes:
            self.nc_Linear = nn.Linear(self.emb_dim,self.num_classes).to(self.device)
            nn.init.xavier_uniform_(self.nc_Linear.weight)
            
        self.nc_W = nn.Linear(2 * self.emb_dim,self.emb_dim).to(self.device)
        nn.init.xavier_uniform_(self.nc_W.weight)
        self.inter_W = nn.Linear(self.emb_dim,self.emb_dim, bias=False).to(self.device)

        self.node_emb = nn.Embedding(node_num, emb_dim).to(self.device)

        self.attr_emb = attr_emb_model(input_dim=attr_num, feature_dim= emb_dim,
                                hidden_dim=emb_dim, output_dim=emb_dim,
                                feature_pre=True, layer_num=0 if n_hidden is None else n_hidden,
                                dropout=dropout_p).to(device)

        self.node_layer = h_layer(self.emb_dim,self.device,self.BCE_mode, mode=self.mode)
        # self.attr_layer = self.node_layer
        # self.inter_layer = self.node_layer
        self.attr_layer = h_layer(self.emb_dim,self.device,self.BCE_mode, mode=self.mode)
        self.inter_layer = h_layer(self.emb_dim,self.device,self.BCE_mode, mode=self.mode)
        
    def node_forward(self, nodes):
        first_embs = self.node_emb(nodes[:,0])
        sec_embs = self.node_emb(nodes[:,1])
        return self.node_layer(first_embs,sec_embs)
    
    def attr_forward(self, nodes,data):
        node_emb = self.dropout(self.attr_emb(data))
        attr_res = self.attr_layer(node_emb[nodes[:,0]],node_emb[nodes[:,1]])
        return attr_res
    
    def inter_forward(self, nodes,data):
        first_nodes = nodes[:,0]
        first_embs = self.attr_emb(data)
        # first_embs = self.inter_W(first_embs)
        first_embs = self.dropout(first_embs)[first_nodes]
        sec_embs = self.node_emb(nodes[:,1])
        return self.inter_layer(first_embs,sec_embs)


    def RLL_loss(self,scores,dists,labels,alpha=0.2, mode='cos'):

        gamma_1 = self.gamma
        gamma_2 = self.gamma
        b_1 = 0.1
        b_2 = 0.1

        return torch.mean(labels*(torch.log(1+torch.exp(-scores*gamma_1+b_1)))/gamma_1+ torch.exp(dists)*(1-labels)*torch.log(1+torch.exp(scores*gamma_2+b_2))/gamma_2)

    def default_loss(self,inputs, labels, data,thetas=(1,1,1), train_num = 1330,c_nodes=None, c_labels=None):
        if self.BCE_mode:
            labels = labels.float()
        nodes = inputs.to(self.device)
        labels = labels.to(self.device)
        dists = data.dists[nodes[:,0],nodes[:,1]] 

        loss_list = []

        scores = self.node_forward(nodes) 
        node_loss = self.RLL_loss(scores,dists,labels)
        loss_list.append(node_loss*thetas[0])

        scores = self.attr_forward(nodes,data)
        attr_loss = self.RLL_loss(scores,dists,labels)
        loss_list.append(attr_loss*thetas[1])
         
        unique_nodes = torch.unique(nodes)
        first_embs = self.attr_emb(data)[unique_nodes]

        sec_embs = self.node_emb(unique_nodes)
        loss_list.append(-self.cos(first_embs,sec_embs).mean()*thetas[2])

        losses = torch.stack(loss_list)
        self.losses = losses.data
        return losses.sum()

    def evaluate(self, nodes,data, lambdas=(1,1,1)):
       
        node_emb = self.node_emb(torch.arange(self.node_num).to(self.device)) 
        first_embs = node_emb[nodes[:,0]]
        sec_embs = node_emb[nodes[:,1]]
        res = self.node_layer(first_embs,sec_embs) * lambdas[0]

        node_emb = self.attr_emb(data)
        first_embs = node_emb[nodes[:,0]]
        sec_embs = node_emb[nodes[:,1]]
        res = res + self.attr_layer(first_embs,sec_embs)* lambdas[1]

        
        first_nodes = nodes[:,0]
        first_embs = self.attr_emb(data)[first_nodes]
        sec_embs = self.node_emb(torch.LongTensor(nodes[:,1]).to(self.device))
        res = res + self.inter_layer(first_embs,sec_embs)* lambdas[2]

        return res
    
