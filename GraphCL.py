import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import torch.optim as optim
# from torch.utils.data import DataLoader as DL2
from torch_geometric.datasets import TUDataset
from torch_geometric.data import InMemoryDataset, download_url
import torch.nn.functional as Func
import os
import time
import sys
import numpy as np
import random
sys.path.append("../")
from preprocessing import *


def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))

def element_l1(embedding_1, embedding_2):
    # print(embedding_1-embedding_2)
    return torch.abs(embedding_1 - embedding_2).float()

    
class linear_for_binary(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(linear_for_binary, self).__init__()
        # self.conv1 = GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True) 
        # self.conv2 = GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)     
        # self.hidden = torch.nn.Linear(in_dim, hidden_dim)
        self.classify = nn.Linear(in_dim, n_classes)

    def forward(self, emb):
        """g表示批处理后的大图，N表示大图的所有节点数量，n表示图的数量 
        """# [n, hidden_dim]
        # emb = Func.relu(self.hidden(emb)).view(-1)
        return self.classify(emb)

class linear_for_dml(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(linear_for_dml, self).__init__()
        # self.conv1 = GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True) 
        # self.conv2 = GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)     
        # self.hidden = torch.nn.Linear(in_dim, hidden_dim)
        self.hidden = nn.Linear(in_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, emb1, emb2):
        # [n, hidden_dim]
        # emb = Func.relu(self.hidden(emb)).view(-1)
        emb1 = self.hidden(emb1)
        emb2 = self.hidden(emb2)
        hg = element_l1(emb1, emb2)
        return emb1, emb2, self.classify(hg)

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g
'''
class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GConv, self).__init__()
        self.conv1 = GConv(input_dim, hidden_dim) 
        self.conv2 = GConv(hidden_dim, hidden_dim)   
        self.conv3 = GConv(hidden_dim, hidden_dim)
        self.conv4 = GConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)

        return x, self.classify(x)
'''

class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        return z, g, z1, z2, g1, g2


def train(datasetname, encoder_model, contrast_model, dataloader, optimizer, device):
    # device = torch.device('cuda:'+str(device))
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        # print(data)
        data = data.to(device)
        optimizer.zero_grad()
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        # print("data[0].x: ", data[0].x)
        # print("data.x: ", data.x)
        _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    return epoch_loss

def train_with_metric(datasetname, encoder_model, contrast_model, dataset, optimizer, device):
    # device = torch.device('cuda:'+str(device))
    encoder_model.train()
    epoch_loss = 0
    dataset_dict = {}
    for data in dataset:
        # print(data.y)
        if data.y not in dataset_dict:
            dataset_dict[data.y] = [data]
        else:
            dataset_dict[data.y].append(data)
    labels = [0,1,2,3,4]
    final_data = get_paired_samples_for_finetune(dataset_dict,labels,200000)
    for data in dataloader:
        print(data)
        data = data.to(device)
        optimizer.zero_grad()
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        # print("data[0].x: ", data[0].x)
        # print("data.x: ", data.x)
        _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    return epoch_loss


def get_paired_samples(x, y, threshold):
    dataset = {}
    labels = []
    for i in range(len(x)):
        new_label = int(y[i].cpu().numpy())
        if new_label not in dataset:
            labels.append(new_label)
            dataset[new_label] = [x[i]]
        else:
            dataset[new_label].append(x[i])
    print(labels)
    print('length of dataset:', len(dataset))
    positive_samples = [] # label=1, x1 and x2 belongs to the same label
    negative_samples = [] # label=0, x1 and x2 belongs to different labels
    i = 0
    while i < threshold:
        posi_label = random.choice(labels)
        # print(posi_label)
        # print(posi_label in dataset)
        # print(dataset[posi_label])
        # print(dataset)
        num1 = random.randint(0, len(dataset[posi_label])-1)
        num2 = random.randint(0, len(dataset[posi_label])-1)
        if num1 != num2:
            try:
                positive_samples.append((dataset[posi_label][num1], dataset[posi_label][num2], 1))
            except:
                print("posi_label", posi_label)
                print("len(dataset[posi_label])", len(dataset[posi_label]))
                print("num1", num1)
                print("num2", num2)
            i += 1
        if i % 10000 == 0:
            print(i, time.asctime(time.localtime(time.time())))
    i = 0
    while i < threshold:
        neg_label1 = random.choice(labels)
        neg_label2 = random.choice(labels)
        if neg_label1 != neg_label2:
            graph1 = random.choice(dataset[neg_label1])
            graph2 = random.choice(dataset[neg_label2])
            negative_samples.append((graph1, graph2, 0))
            i += 1
        if i % 10000 == 0:
            print(i, time.asctime(time.localtime(time.time())))
    return  positive_samples+negative_samples

def get_paired_samples_for_finetune(dataset, labels, threshold):
    print(labels)
    print('length of dataset:', len(dataset))
    positive_samples = [] # label=1, x1 and x2 belongs to the same label
    negative_samples = [] # label=0, x1 and x2 belongs to different labels
    i = 0
    while i < threshold:
        posi_label = random.choice(labels)
        # print(posi_label)
        # print(posi_label in dataset)
        # print(dataset[posi_label])
        # print(dataset)
        num1 = random.randint(0, len(dataset[posi_label])-1)
        num2 = random.randint(0, len(dataset[posi_label])-1)
        if num1 != num2:
            try:
                positive_samples.append((dataset[posi_label][num1], dataset[posi_label][num2], 1))
            except:
                print("posi_label", posi_label)
                print("len(dataset[posi_label])", len(dataset[posi_label]))
                print("num1", num1)
                print("num2", num2)
            i += 1
        if i % 1000 == 0:
            print(i, time.asctime(time.localtime(time.time())))
    i = 0
    while i < threshold:
        neg_label1 = random.choice(labels)
        neg_label2 = random.choice(labels)
        if neg_label1 != neg_label2:
            graph1 = random.choice(dataset[neg_label1])
            graph2 = random.choice(dataset[neg_label2])
            negative_samples.append((graph1, graph2, 0))
            i += 1
        if i % 1000 == 0:
            print(i, time.asctime(time.localtime(time.time())))
    return  positive_samples+negative_samples

def predict_dml(datasetname, x, y, number_of_samples, threshold, device):
    dataset = {}
    labels = []
    for i in range(len(x)):
        new_label = int(y[i].cpu().numpy())
        if new_label not in dataset:
            labels.append(new_label)
            dataset[new_label] = [x[i]]
        else:
            dataset[new_label].append(x[i])
    # device = torch.device('cuda:'+str(device))
    # path = './models/metriclearning/'+datasetname+"/"+str(2.1)+'_cross_entropy_' + str(dataset_size) + "_" + str(depth) + "_2ndOrder.pkl"
    path = '../models/constrative/'+datasetname+"_metric_"+str(threshold)+"_.pkl"
    # if datasetname == 'github_stargazers':
    #     model = torch.load(path, map_location={'cuda:7':'cuda:6'})
    # else:
    #     model = torch.load(path, map_location={'cuda:2':'cuda:6'})
    model = torch.load(path, map_location=lambda storage, loc: storage.cuda(device))
    real_label = []
    pred_label = []
    time_flag = 0
    print(len(dataset[0]))
    for i in range(len(dataset)):
        graphs = dataset[i]
        # graphs.to(device)
        for one_graph in graphs:
            real_label.append(i)
            pred_all = []
            one_graph = one_graph.to(device)
            for j in range(len(trainset)):
                if time_flag%1000 == 0:
                    print(time_flag, time.asctime(time.localtime(time.time())))
                time_flag += 1
                pred = 0
                ref_graphs = random.sample(dataset[j],number_of_samples)
                # ref_graphs = trainset[j]
                for ref_graph in ref_graphs:
                    ref_graph = ref_graph.to(device)
                    emb1, emb2, prediction = model(one_graph, ref_graph)
                    # prediction = torch.unsqueeze(prediction, 0)
                    # print(prediction.shape)
                    # return
                    prediction = torch.softmax(prediction, dim = 0)
                    # print(prediction)
                    # prediction = torch.max(prediction, 1)[1].view(-1)
                    pred += prediction.detach().cpu().numpy().tolist()[1]
                pred_all.append(pred/number_of_samples)
                # pred_all.append(pred/len(trainset[j]))
            pred_label.append(pred_all.index(max(pred_all)))
    print(real_label)
    print(pred_label)
    print("accuracy: ", accuracy_score(real_label, pred_label))
    print("classification report: ", classification_report(real_label, pred_label))
    cm = confusion_matrix(real_label, pred_label)
    print("confusion matrix: ")
    print(cm)

def test_metric(datasetname, x,y,epoch,threshold,device):
    # if os.path.exists('../models/constrative/'+datasetname+"_metric.pkl"):
    #     return
    # paired_samples = get_paired_samples(x,y,200)
    dataset = list(zip(x,y))
    random.shuffle(dataset)
    trainset = dataset[:int(0.8*len(dataset))]
    testset = dataset[int(0.8*len(dataset)):]
    trainset_x, trainset_y = zip(*trainset)
    testset_x, testset_y = zip(*testset)
    train_paired_samples = get_paired_samples(trainset_x,trainset_y,threshold)
    train_loader = DataLoader(train_paired_samples, 128, shuffle=True)
    # print(len(x[0]))
    model = linear_for_dml(len(x[0]), 32, 2)
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    loss_func.to(device)  
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #train model
    model.train()
    epoch_losses = []
    for epoch in range(epoch):   
        epoch_loss = 0
        for iter, (graphs1, graphs2, label) in enumerate(train_loader):
            graphs1 = graphs1.to(device)
            graphs2 = graphs2.to(device)
            label = label.to(device)
            emb1, emb2, prediction = model(graphs1, graphs2)
            pred = torch.softmax(prediction, 1)
            pred = torch.max(pred, 1)[1].view(-1)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            epoch_loss += loss.item()
            #epoch_loss += loss.depatch().item
        epoch_loss /= (iter+1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss), time.asctime(time.localtime(time.time())))
        # print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss), time.asctime(time.localtime(time.time())), accuracy_score(label.cpu().numpy().tolist(), pred.detach().cpu().numpy().tolist()))
        epoch_losses.append(epoch_loss)
    # test model
    test_paired_samples = get_paired_samples(testset_x,testset_y,threshold/10)
    test_loader = DataLoader(test_paired_samples, 128, shuffle=True)
    model.eval()
    test_pred, test_label = [], []
    with torch.no_grad():
        for it, (graphs1, graphs2, label) in enumerate(test_loader):
            # batchdgl = Variable(batchdgl.tensor()).cuda()
            # label = Variable(label.tensor()).cuda()
            graphs1 = graphs1.to(device)
            graphs2 = graphs2.to(device)
            label = label.to(device)
            emb1, emb2, prediction = model(graphs1, graphs2)
            pred = torch.softmax(prediction, 1)
            pred = torch.max(pred, 1)[1].view(-1)
            test_pred += pred.detach().cpu().numpy().tolist()
            test_label += label.cpu().numpy().tolist()
        # labels = ["real", "ER", "BA"]
    print("accuracy: ", accuracy_score(test_label, test_pred))
    print("classification report: ", classification_report(test_label, test_pred))
    cm = confusion_matrix(test_label, test_pred)
    print("confusion matrix: ")
    print(cm)
    # if not os.path.exists('/home/c01yima/ggnn/PyGCL/models/constrative/'+datasetname):
    #     os.mkdir('/home/c01yima/ggnn/PyGCL/models/constrative/'+datasetname)
    path = '../models/constrative/'+datasetname+"_metric_"+str(threshold)+"_.pkl"
    torch.save(model, path)

def test_linear(datasetname, x, y, epoch, device):
    dataset = zip(x,y)
    # dataloader = DataLoader(dataset, 128, shuffle=True)
    # label_real = []
    # label_pred = []
    model = linear_for_binary(len(x[0]), 32, 6)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()
    loss_func.to(device)
    model.train()
    epoch_losses = []
    print(len(x), len(y))
    train_x = x[:int(0.8*len(x))]
    test_x = x[int(0.8*len(x)):]
    print(test_x)
    train_y = y[:int(0.8*len(y))]
    test_y = y[int(0.8*len(y)):]
    print(test_y)
    print(len(train_x), len(train_y), len(test_x), len(test_y))
    for epoch in range(epoch):
        train_x = train_x.to(device)
        pred = model(train_x)
        loss = loss_func(pred, train_y)
        optimizer.zero_grad()   # 清空上一步的残余更新参数值
        loss.backward(retain_graph=True)         # 误差反向传播, 计算参数更新值
        optimizer.step()
        print('Epoch {}, loss {:.4f}'.format(epoch, loss.item()), time.asctime(time.localtime(time.time())))
        epoch_losses.append(loss)
        # pred = torch.max(Func.softmax(pred), 1)[1]
        # label_pred = pred.detach().cpu().numpy().tolist()
    with torch.no_grad():
        test_x = test_x.to(device)
        pred = model(test_x)
        prediction = torch.max(Func.softmax(pred), 1)[1] # 1表示维度1，列，[0]表示概率值，[1]表示标签
        pred_y = prediction.detach().cpu().numpy()
        label_real = test_y.cpu().numpy()
    print("accuracy: ", accuracy_score(pred_y, label_real))
    print("classification report: ", classification_report(pred_y, label_real))
    cm = confusion_matrix(pred_y, label_real)
    print("confusion matrix: ")
    print(cm)
    # if not os.path.exists('../models/contrastive/'+datasetname):
    #     os.mkdir('../models/contrastive/'+datasetname)
    path = '../models/constrative/'+datasetname+"_linear.pkl"
    torch.save(model, path)

from sklearn import manifold 
import seaborn as sns
from pandas.core.frame import DataFrame
import json
def tsne_plot_using_embeddings(datasetname, embeddings, labels):
    print(labels)
    print("start printing tsne result")
    tsne = manifold.TSNE(n_components=2, init="pca", random_state=2)
    # final_embeddings = {"x":[], "y":[], "labels":labels}
    flag = 0
    # all_embeddings = all_embeddings.cpu().detach().numpy()
    embedded = tsne.fit_transform(embeddings)
    final_embeddings = {"x":embedded[:,0].tolist(), "y":embedded[:,1].tolist(), "labels":labels}
    print(final_embeddings)
    with open("../results/tsne_contrastive_"+datasetname+".json", "w") as f:
        json.dump(final_embeddings, f, indent=4, ensure_ascii=True)
    data = DataFrame(final_embeddings)
    # sns.set(rc = {'figure.figsize':(15,15)})
    plt.figure()
    fig_dims = (8, 8)
    fig, ax = plt.subplots(figsize=fig_dims)
    current_palette = sns.color_palette("muted", n_colors=2)
    color_list = sns.color_palette(current_palette).as_hex()
    fig = sns.scatterplot(ax=ax, x ="x", y="y", hue="labels", legend='full', data=data, s=15, palette=color_list)
    scatter_fig = fig.get_figure()
    scatter_fig.savefig("../figures/"+datasetname+"_contrastive_all_tsne.jpg")

def test(datasetname, encoder_model, dataloader, way, epoch, device):
    # device = torch.device('cuda:'+str(device))
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        # if len(data.y) != 128:
        #     continue
        print("len(data): ", data.size())
        data = data.to(device)
        # print("data.edge_index: ", data.edge_index)
        # print("data: ", data)
        if data.x is None:
            num_nodes = data.batch.size(0)
            # print(num_nodes)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
            # print("data.x: ", data.x)
        # print("data[0].x: ", data[0].x)
        # print("data.x: ", data.x)
        # print("data.x.shape(): ", len(data.x))
        # print("data.edge_index.shape(): ", data.edge_index.size())
        # print("data.batch: ", len(data.batch))
        _, g, _, _, _, _ = encoder_model(data.x, data.edge_index, data.batch)
        if len(g) != 128:
            continue
        x.append(g)
        # print("shape of g: ", g.size())
        y.append(data.y)
        # print("shape of y: ", data.y.size())
    # print(x[0])
    # print(y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    x_final = []
    y_final = []
    for i in range(len(x)):
        x_final.append(x[i].cpu().detach().numpy())
        y_final.append(int(y[i].cpu().detach()))
    finaldata = {x:x_final, y:y_final}
    print(x_final[0])
    print(y_final[0])
    # with open("./figures/"+datasetname+"_embedding.json", "w") as f:
    #     json.dump(finaldata, fp=f, indent=4)
    # print(x_final[0])
    # print(y_final[0])
    # tsne_plot_using_embeddings(datasetname, x_final,y_final)

    # return x, y
    # print(torch.Tensor(x).size())
    # print(torch.Tensor(y).size())
    # print(len(x[0]),len(y[0]))
    print("x.shape(): ", x.size())
    print("y.shape(): ", y.size())
    print(len(x), len(y))
    if way == "svm":
        print("svm")
        split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
        result = SVMEvaluator(linear=True)(x, y, split)
        print(f'(E): Best test F1Mi={result["micro_f1"]:.4f}, F1Ma={result["macro_f1"]:.4f}')
        print("accuracy: ", result['accuracy'])
        print('classification report: ')
        print(result["classification_report"])
        print("confusion matrix: ")
        print(result['cm'])
    elif way == "linear":
        test_linear(datasetname, x,y,epoch,device)
    elif way == "metric":
        train_x = x[:int(0.8*len(x))]
        test_x = x[int(0.8*len(x)):]
        # print(test_x)
        train_y = y[:int(0.8*len(y))]
        test_y = y[int(0.8*len(y)):]
        threshold = 20000
        test_metric(datasetname, train_x,train_y,epoch, threshold, device)
        reference_number = [10]
        for number_of_samples in reference_number:
            print("number of samples: ", number_of_samples)
            # print("test_x[0]", test_x[0])
            predict_dml(datasetname, test_x, test_y, number_of_samples, threshold, device)
    # return x,y

        
    # split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    # result = SVMEvaluator(linear=True)(x, y, split)
    # print(f'(E): Best test F1Mi={result["micro_f1"]:.4f}, F1Ma={result["macro_f1"]:.4f}')
    # print("accuracy: ", result['accuracy'])
    # print('classification report: ')
    # print(result["classification_report"])

    return

def create_my_own_data(dataset):
    dataset_final = []
    flag = 0
    print(len(dataset))
    for i in range(len(dataset)):
        # if i != 0:
        #     continue
        for graph in dataset[i]:
            flag += 1
            if flag % 1000 == 0:
                print(flag)
            edges = graph.edges()
            a = edges[0].numpy().tolist()
            b = edges[1].numpy().tolist()
            edges = torch.Tensor([a,b])
            edges = edges.long()
            degrees = graph.in_degree(graph.nodes())
            degree = degrees.numpy().reshape(len(degrees.numpy()), 1)
            # print(edges)
            # print(edges)
            # a, b = zip(*edges)
            # a, b = list(a), list(b)
            # edge_index = torch.tensor([a,b])
            data = Data()
            data.edge_index = edges
            # data.x = degree
            data.y = i
            # print(data.y)
            # print(data.edge_index)
            # print(data)
            # break
            dataset_final.append(data)
    # print(dataset_final[0])
    return dataset_final


def main(scenario, datasetname, trainset, testset, device):
    if not os.path.exists("./models"):
        os.mkdir("./models")
    if not os.path.exists("./models/constrative"):
        os.mkdir("./models/constrative")
    if not os.path.exists('./models/constrative/'+datasetname):
        os.mkdir('./models/constrative/'+datasetname)
    path = "./models/constrative/"+datasetname+"_sampled_"+scenario+".pkl"
    # path = "/home/c01yima/ggnn/models/constrative/"+datasetname+"_sampled.pkl"
    # path = "/home/c01yima/ggnn/models/constrative/"+datasetname+"_lr_0.001_finetune.pkl"
    # device = torch.device('cuda:1')
    # path = osp.join(osp.expanduser('~'), 'datasets')
    # trainset, testset = get_dataset_from_saved_file(datasetname, algorithm_list)
    # dataset_final = {}
    # for i in range(len(trainset)):
    #     dataset_final[i] = trainset[i] + testset[i]
    # dataset_final[0] = trainset[0]+testset[0]
    # dataset = {}
    # print("trainset", len(trainset))
    # print("testset", len(testset))
    # dataset[0] = testset[0]
    # dataset[1] = []
    # for i in range(6, len(testset)):
    # # for i in range(6, 11):
    #     dataset[1] += random.sample(testset[i], int(len(testset[0])/3))
    # for i in range(1, len(trainset)):
    #     dataset[1] += random.sample(dataset_final[i], int(len(dataset_final[0])/5))
        # dataset_final[i] = testset[i]
    # dataset_final = trainset + testset
    dataset = {}
    for i in range(len(trainset)):
        dataset[i] = trainset[i]+testset[i]
    dataset = create_my_own_data(dataset)
    # print(dataset[0])
    # print(len(dataset))
    # print(dataset[1234])
    # print(dataset[0].num_features())
    # return
    # dataset = TUDataset(path, name='PTC_MR')
    # print(dataset[0].edge_index)
    # print(dataset[0].edge_index.dtype)
    # return
    # print(dataset[0].x)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    input_dim = 1
    # input_dim = max(dataset.num_features, 1)
    # aug1 = A.Identity()
    aug1 = A.NodeDropping(pn=0.2)
    aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=5),
                        A.NodeDropping(pn=0.1),
                        A.EdgeRemoving(pe=0.1)], 1)
    # print(aug2)
    gconv = GConv(input_dim=input_dim, hidden_dim=32, num_layers=2).to(device)
    # gconv = GConv(input_dim=input_dim, hidden_dim=64).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
    optimizer = Adam(encoder_model.parameters(), lr=0.001)
    epoch = 200
    if not os.path.exists(path):
        with tqdm(total=epoch, desc='(T)') as pbar:
            for oneepoch in range(1, 1+epoch):
                loss = train(datasetname, encoder_model, contrast_model, dataloader, optimizer, device)
                # loss = train_with_metric(datasetname, encoder_model, contrast_model, dataset, optimizer, device)
                pbar.set_postfix({'loss': loss})
                pbar.update()
                # print('Epoch {}, loss {:.4f}'.format(epoch, loss), time.asctime(time.localtime(time.time())))
        torch.save(encoder_model.state_dict(), path)
    else:
        # encoder_model = Encoder(encoder = gconv, augmentor = (aug1,aug2))
        encoder_model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage.cuda(3)))
        # print(encoder_model)
        # encoder_model.load_state_dict(torch.load(path))
    # way = ["svm", "linear", "metric"]
    # way = ["svm", "linear"]
    way = ["svm"]
    for oneway in way:
        print("evaluation method:", oneway)
        test(datasetname, encoder_model, dataloader, oneway, epoch, device)
    # try:
    #     test(encoder_model, dataloader, device)
    # except Exception as e:
    #     # print(e)
    #     print("test error")
    # print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
    # print("accuracy: ", test_result['accuracy'])
    # print('classification report: ')
    # print(test_result["classification_report"])

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    device = 6
    device = torch.device('cuda:'+str(device))
    # device = torch.device('cpu')
    datasetname1 = ["COLLAB", "twitch_egos"]
    datasetname2 = ["AIDS", "alchemy_full", "deezer_ego_nets", "DBLP_v1", "github_stargazers"]
    # datasetname2 = ["alchemy_full"]
    datasetname3 = ["all", "all_minus2"]
    # datasetname3 = ["all_minus2"]
    # for onedataset in datasetname2+datasetname3:
    all_dataset = datasetname1+datasetname2
    # scenario = ["s1","s2", "s3", "s4"]
    scenario = ["s1","s2"]
    for onedataset in datasetname2+datasetname1:
        # if onedataset == "COLLAB":
        #     scenario = ["s4"]
        algorithm_list = ['real', 'ER', 'BA', "graphite","vgae", 'GraphRNN_RNN']
        algorithm_list_openworld = ["GRAN","GraphRNN_VAE_conditional", "sbmgnn"]
        print("training contrastive model of "+onedataset)
        for onescenario in scenario:
            print(onescenario)
            trainset, testset, trainset_final, testset_final = get_sampled_dataset(onedataset, all_dataset, onescenario)
        # if onedataset == "all":
        #     trainset, testset = get_dataset_for_binary_all_openworld(datasetname1+datasetname2, algorithm_list, algorithm_list_openworld)
        # elif onedataset == "all_minus2":
        #     trainset, testset = get_dataset_for_binary_all_minus2_openworld(datasetname1+datasetname2, algorithm_list, algorithm_list_openworld)
        # else:
        #     trainset, testset = get_dataset_from_saved_file_openworld(onedataset, algorithm_list, algorithm_list_openworld)
            main(onescenario, onedataset, trainset_final, testset_final, device)
    # for onedataset in datasetname1:
    #     algorithm_list = ['real', 'ER', 'BA', "graphite","vgae", "GraphRNN_RNN"]
    #     algorithm_list_openworld = ['GRAN', "GraphRNN_VAE_conditional", "sbmgnn"]
    #     # if onedataset == "COLLAB":
    #     #     continue
    #     print("training contrastive model of "+onedataset)
    #     trainset, testset = get_dataset_from_saved_file_openworld(onedataset, algorithm_list, algorithm_list_openworld)
    #     main(onedataset, trainset, testset, device)
    # onedataset = datasetname2[4]
    # algorithm_list = ['real', 'ER', 'BA', "graphite","vgae", "GRAN"]
    # algorithm_list_openworld = ['GraphRNN_RNN',"GraphRNN_VAE_conditional", "sbmgnn"]
    # print("training contrastive model of "+onedataset)
    # main(onedataset, device)