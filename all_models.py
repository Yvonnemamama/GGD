import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F

import dgl
import torch.optim as optim
from dgl.nn.pytorch import SAGEConv
from train import *


class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCNClassifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True) 
        self.conv2 = GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)   
        self.conv3 = GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        self.conv4 = GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)   
        self.classify = nn.Linear(hidden_dim, n_classes)  # 定义分类器

    def forward(self, g):
        """g表示批处理后的大图，N表示大图的所有节点数量，n表示图的数量 
        """
        #ß
        # 我们用节点的度作为初始节点特征。对于无向图，入度 = 出度
        h = g.in_degrees().view(-1, 1).float() # [N, 1]
        # 执行图卷积和激活函数
        h = F.relu(self.conv1(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv2(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv3(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv4(g, h))  # [N, hidden_dim]
        g.ndata['h'] = h    # 将特征赋予到图的节点
        # 通过平均池化每个节点的表示得到图表示
        hg = dgl.mean_nodes(g, 'h')   # [n, hidden_dim]
        # g.ndata['h'] = torch.mm(h) # 通过平均池化每个节点的表示得到图表示  # [n, hidden_dim]
        return hg, self.classify(hg)  # [n, n_classes]

class GCNClassifier_for_dml(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCNClassifier_for_dml, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True) 
        self.conv2 = GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)   
        self.conv3 = GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        self.conv4 = GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)    
        self.classify = nn.Linear(hidden_dim, n_classes) 

    def forward(self, g1, g2):
        """g表示批处理后的大图，N表示大图的所有节点数量，n表示图的数量 
        """
        #ß
        # use the degree of nodes as initial node feature
        # g1 = list(g)[0]
        # g2 = list(g)[1]
        h1 = g1.in_degrees().view(-1, 1).float() # [N, 1]
        # graph convolution and activate function
        h1 = F.relu(self.conv1(g1, h1))  # [N, hidden_dim]
        h1 = F.relu(self.conv2(g1, h1))  # [N, hidden_dim]
        h1 = F.relu(self.conv3(g1, h1))  # [N, hidden_dim]
        h1 = F.relu(self.conv4(g1, h1))  # [N, hidden_dim]
        # h1 = self.conv1(g1, h1)
        # h1 = self.conv2(g1, h1)
        g1.ndata['h1'] = h1    # give the features to graph nodes
        # 通过平均池化每个节点的表示得到图表示
        hg1 = dgl.mean_nodes(g1, 'h1')   # [n, hidden_dim]
    
        h2 = g2.in_degrees().view(-1, 1).float() # [N, 1]
        h2 = F.relu(self.conv1(g2, h2))  # [N, hidden_dim]
        h2 = F.relu(self.conv2(g2, h2))  # [N, hidden_dim]
        h2 = F.relu(self.conv3(g2, h2))  # [N, hidden_dim]
        h2 = F.relu(self.conv4(g2, h2))  # [N, hidden_dim]
        # h2 = self.conv1(g2, h2)
        # h2 = self.conv2(g2, h2)
        g2.ndata['h2'] = h2
        hg2 = dgl.mean_nodes(g2, 'h2')
        # print(hg2.shape)   # [n, hidden_dim]
        hg = element_l1(hg1, hg2)
        # hg = torch.cat([hg1,hg2],1)
        # print(hg)
        return hg1, hg2, self.classify(hg)  # [n, n_classes]

class GCNClassifier_for_dml_2_layers(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCNClassifier_for_dml_2_layers, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True) 
        self.conv2 = GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)     
        self.classify = nn.Linear(hidden_dim, n_classes) 

    def forward(self, g1, g2):
        """g表示批处理后的大图，N表示大图的所有节点数量，n表示图的数量 
        """
        #ß
        # use the degree of nodes as initial node feature
        # g1 = list(g)[0]
        # g2 = list(g)[1]
        h1 = g1.in_degrees().view(-1, 1).float() # [N, 1]
        # graph convolution and activate function
        h1 = F.relu(self.conv1(g1, h1))  # [N, hidden_dim]
        h1 = F.relu(self.conv2(g1, h1))  # [N, hidden_dim]
        # h1 = self.conv1(g1, h1)
        # h1 = self.conv2(g1, h1)
        g1.ndata['h1'] = h1    # give the features to graph nodes
        # 通过平均池化每个节点的表示得到图表示
        hg1 = dgl.mean_nodes(g1, 'h1')   # [n, hidden_dim]
    
        h2 = g2.in_degrees().view(-1, 1).float() # [N, 1]
        h2 = F.relu(self.conv1(g2, h2))  # [N, hidden_dim]
        h2 = F.relu(self.conv2(g2, h2))  # [N, hidden_dim]
        # h2 = self.conv1(g2, h2)
        # h2 = self.conv2(g2, h2)
        g2.ndata['h2'] = h2
        hg2 = dgl.mean_nodes(g2, 'h2')
        # print(hg2.shape)   # [n, hidden_dim]
        hg = element_l1(hg1, hg2)
        # hg = torch.cat([hg1,hg2],1)
        # print(hg)
        return hg1, hg2, self.classify(hg)  # [n, n_classes]

class GCNClassifier_for_dml_3_layers(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCNClassifier_for_dml_3_layers, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True) 
        self.conv2 = GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)   
        self.conv3 = GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)  
        self.classify = nn.Linear(hidden_dim, n_classes) 

    def forward(self, g1, g2):
        """g表示批处理后的大图，N表示大图的所有节点数量，n表示图的数量 
        """
        #ß
        # use the degree of nodes as initial node feature
        # g1 = list(g)[0]
        # g2 = list(g)[1]
        h1 = g1.in_degrees().view(-1, 1).float() # [N, 1]
        # graph convolution and activate function
        h1 = F.relu(self.conv1(g1, h1))  # [N, hidden_dim]
        h1 = F.relu(self.conv2(g1, h1))  # [N, hidden_dim]
        h1 = F.relu(self.conv3(g1, h1))  # [N, hidden_dim]
        # h1 = self.conv1(g1, h1)
        # h1 = self.conv2(g1, h1)
        g1.ndata['h1'] = h1    # give the features to graph nodes
        # 通过平均池化每个节点的表示得到图表示
        hg1 = dgl.mean_nodes(g1, 'h1')   # [n, hidden_dim]
    
        h2 = g2.in_degrees().view(-1, 1).float() # [N, 1]
        h2 = F.relu(self.conv1(g2, h2))  # [N, hidden_dim]
        h2 = F.relu(self.conv2(g2, h2))  # [N, hidden_dim]
        h2 = F.relu(self.conv3(g2, h2))  # [N, hidden_dim]
        # h2 = self.conv1(g2, h2)
        # h2 = self.conv2(g2, h2)
        g2.ndata['h2'] = h2
        hg2 = dgl.mean_nodes(g2, 'h2')
        # print(hg2.shape)   # [n, hidden_dim]
        hg = element_l1(hg1, hg2)
        # hg = torch.cat([hg1,hg2],1)
        # print(hg)
        return hg1, hg2, self.classify(hg)  # [n, n_classes]
