import math
from operator import neg
from networkx.algorithms.clique import number_of_cliques
from networkx.algorithms.lowest_common_ancestors import tree_all_pairs_lowest_common_ancestor
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier
from preprocessing import *
from all_models import *
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
import dgl
import torch.optim as optim
from torch.utils.data import DataLoader
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import os
import time
import json
# from graph-generation.GRAN.utils.dist_helper import compute_mmd
from torch.autograd import Variable
from dgl.nn.pytorch import SAGEConv
import random
import networkx as nx
# from GraphEmbedding.ge.models import node2vec
# from pytorch_metric_learning import losses, miners, distances, reducers, testers
# from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.linear_model import SGDClassifier
# import evaluate
from sklearn.inspection import permutation_importance
from all_models import *
def collate(samples):
    # samples is a list, [(graph1, label1), (graph2, label2), ...]
    graphs, labels = map(list, zip(*samples))
    return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)

def collate_2(samples):
    # samples are a list, [(graph1, graph2, label1), (graph3, graph4, label2), ...]
    graphs1, graphs2, labels = map(list, zip(*samples))
    return dgl.batch(graphs1), dgl.batch(graphs2), torch.tensor(labels, dtype=torch.long)

def feature_classifier_sampled(datasetname, dataset_list, scenario):
    # featureset_train, featureset_test = evaluate.get_features(datasetname, algorithm_list)
    if scenario == "s1":
        with open("./datasets/processed/"+datasetname+"/closedworld_feature_sampled_norm.json", "r") as f:
            featureset = json.load(f)
        x = []
        y = []
        for i in range(6):
            if i == 0:
                for onedata in featureset[str(i)]:
                    # train_x.append(onedata[:-2])
                    x.append(onedata)
                    y.append(0)
            if i != 0 and i < 6:
                # data = random.sample(featureset_train[str(i)], int(len(featureset_train['0'])/5))
                for onedata in featureset[str(i)]:
                    # train_x.append(onedata[:-2])
                    x.append(onedata)
                    y.append(1)
        x = np.nan_to_num(x)
        train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.2,random_state=42)
    if scenario == "s2":
        with open("./datasets/processed/"+datasetname+"/closedworld_feature_sampled_norm.json", "r") as f:
            featureset = json.load(f)
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        print(len(featureset))
        for i in range(len(featureset)):
            if i == 0:
                random.shuffle(featureset[str(i)])
                trainset = featureset[str(i)][:int(0.8*len(featureset[str(i)]))]
                testset = featureset[str(i)][int(0.8*len(featureset[str(i)])):]
                train_x += trainset
                train_y += [0 for j in range(len(trainset))]
                test_x += testset
                test_y += [0 for j in range(len(testset))]
            if i != 0 and i < 6:
                random.shuffle(featureset[str(i)])
                train_x += featureset[str(i)]
                train_y += [1 for j in range(len(featureset[str(i)]))]
            if i != 0 and i > 5:
                random.shuffle(featureset[str(i)])
                test_x += featureset[str(i)]
                test_y += [1 for j in range(len(featureset[str(i)]))]
        # x = x.fillna(x.mean())
        # x = np.nan_to_num(x)
        train_x = np.nan_to_num(train_x)
        test_x = np.nan_to_num(test_x)
        # print(train_y)
        # print(test_y)
        print(len(train_x),len(train_y),len(test_x),len(test_y))
    if scenario == "s3":
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for dataset in dataset_list[:-2]:
            with open("./datasets/processed/"+dataset+"/closedworld_feature_sampled.json", "r") as f:
                featureset = json.load(f)
            for i in range(len(featureset)):
                if i == 0:
                    random.shuffle(featureset[str(i)])
                    train_x += featureset[str(i)]
                    train_y += [0 for j in range(len(featureset[str(i)]))]
                if i != 0 and i < 6:
                    random.shuffle(featureset[str(i)])
                    train_x += featureset[str(i)]
                    train_y += [1 for j in range(len(featureset[str(i)]))]
        for dataset in dataset_list[-2:]:
            with open("./datasets/processed/"+dataset+"/closedworld_feature_sampled.json", "r") as f:
                featureset = json.load(f)
            for i in range(len(featureset)):
                if i == 0:
                    random.shuffle(featureset[str(i)])
                    test_x += featureset[str(i)]
                    test_y += [0 for j in range(len(featureset[str(i)]))]
                if i != 0 and i < 6:
                    random.shuffle(featureset[str(i)])
                    test_x += featureset[str(i)]
                    test_y += [1 for j in range(len(featureset[str(i)]))]
            # x = x.fillna(x.mean())
        train_x = np.nan_to_num(train_x)
        test_x = np.nan_to_num(test_x)
    if scenario == "s4":
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for dataset in dataset_list[:-2]:
            with open("./datasets/processed/"+dataset+"/closedworld_feature_sampled.json", "r") as f:
                featureset = json.load(f)
            for i in range(len(featureset)):
                if i == 0:
                    random.shuffle(featureset[str(i)])
                    train_x += featureset[str(i)]
                    train_y += [0 for j in range(len(featureset[str(i)]))]
                if i != 0 and i < 6:
                    random.shuffle(featureset[str(i)])
                    train_x += featureset[str(i)]
                    train_y += [1 for j in range(len(featureset[str(i)]))]
        for dataset in dataset_list[-2:]:
            with open("./datasets/processed/"+dataset+"/closedworld_feature_sampled.json", "r") as f:
                featureset = json.load(f)
            for i in range(len(featureset)):
                if i == 0:
                    random.shuffle(featureset[str(i)])
                    test_x += featureset[str(i)]
                    test_y += [0 for j in range(len(featureset[str(i)]))]
                if i != 0 and i > 5:
                    random.shuffle(featureset[str(i)])
                    test_x += featureset[str(i)]
                    test_y += [1 for j in range(len(featureset[str(i)]))]
            # x = x.fillna(x.mean())
        train_x = np.nan_to_num(train_x)
        test_x = np.nan_to_num(test_x)
    
    # train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.2,random_state=42)
    clf1 = MLPRegressor(hidden_layer_sizes=130,learning_rate="adaptive") 
    clf2 = svm.SVC(gamma=0.001, C=100.)
    clf3 = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    clf4 = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    # print("DT")
    for i in range(len(train_x)):
        rows = train_x[i]
        for j in range(len(rows)):
            if math.isinf(rows[j]) or math.isnan(rows[j]):
                print(1)
                print(train_x[i])
                rows[j] = 1
                print(train_x[i])
        train_x[i] = rows
    for i in range(len(test_x)):
        rows = test_x[i]
        for j in range(len(rows)):
            if math.isinf(rows[j]) or math.isnan(rows[j]):
                print(1)
                print(test_x[i])
                rows[j] = 1
                print(test_x[i])    
        test_x[i] = rows
    train_x = train_x.astype(np.float)
    test_x = test_x.astype(np.float)
    print(np.any(np.isnan(train_x)))
    print(np.all(np.isfinite(train_x)))
    clf1.fit(train_x, train_y)
    r = permutation_importance(clf1, train_x, train_y, n_repeats=30,random_state=0)
    print("r.importances_mean: ", r.importances_mean)
    print("r.importances_std: ", r.importances_std)
    # print(test_y)
    pred_y = []
    pred = clf1.predict(test_x)
    for item in pred:
        if item >= 0.5:
            pred_y.append(1)
        else:
            pred_y.append(0)
    test_macro = f1_score(test_y, pred_y, average='macro')
    test_micro = f1_score(test_y, pred_y, average='micro')
    test_accuracy = accuracy_score(test_y, pred_y)
    test_classification_report = classification_report(test_y, pred_y)
    cm = confusion_matrix(test_y, pred_y)
    print(test_classification_report)
    print("confusion matrix:")
    print(cm)
    print("Accuracy: ", test_accuracy)
    # scores = cross_val_score(clf, )


def GCN_sampled(trainset, testset, datasetname, epoch, batch_size, algorithm_list, output, device):
    device = torch.device('cuda:'+str(device))
    # algorithm_list = ['real', 'ER', 'BA', 'GraphRNN_RNN',"GraphRNN_VAE_conditional", "GRAN"]
    # for i in range(1, len(trainset)):
    # graph_train = []
    # graph_test = []
    # # print("testset: ", algorithm_list[i])
    # label_real_train = [1 for length in range(len(trainset[0]))]
    # label_real_test = [1 for length in range(len(testset[0]))]
    # # print(len(testset[i]))
    # graph_train += list(zip(trainset[0],label_real_train))
    # graph_test += list(zip(testset[0],label_real_test))
    # # label_real_test = [0 for length in range(len(testset[i]))]
    # # graph_test += list(zip(testset[i], label_real_test))
    # for j in range(1, len(trainset)):
    #     label_real_train = [0 for length in range(len(trainset[j]))]
    #     label_real_test = [0 for length in range(len(testset[j]))]
    #     graphs_train = list(zip(trainset[j],label_real_train))
    #     graph_train += graphs_train
    #     # graph_train += random.sample(graphs_train, int(len(trainset[0])/5)+1)
    #     # print(int(len(trainset[0])/5))
    #     graphs_test = list(zip(testset[j],label_real_test))
    #     # graph_test += random.sample(graphs_test, int(len(testset[0])/5)+1)
    #     graph_test += graphs_test
    label_real_train = [0 for length in range(len(trainset[0]))]
    label_fake_train = [1 for length in range(len(trainset[1]))]
    graph_train_real = list(zip(trainset[0], label_real_train))
    graph_train_fake = list(zip(trainset[1], label_fake_train))
    label_real_test = [0 for length in range(len(testset[0]))]
    label_fake_test = [1 for length in range(len(testset[1]))]
    graph_test_real = list(zip(testset[0], label_real_test))
    graph_test_fake = list(zip(testset[1], label_fake_test))
    graph_train = graph_train_real+graph_train_fake
    graph_test = graph_test_real+graph_test_fake
    train_loader = DataLoader(graph_train, batch_size, shuffle=True, collate_fn=collate)
    model = GCNClassifier(1, 256, 2)
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    loss_func.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # patience = 20	
    # early_stopping = EarlyStopping(patience, verbose=True)
    #train model
    model.train()
    epoch_losses = []

    for epoch in range(epoch):
        epoch_loss = 0
        for iter, (batchdgl, label) in enumerate(train_loader):
            batchdgl = batchdgl.to(device)
            label = label.to(device)
            emb, prediction = model(batchdgl)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            #epoch_loss += loss.depatch().item
        epoch_loss /= (iter+1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss), time.asctime(time.localtime(time.time())))
        epoch_losses.append(epoch_loss)
        # early_stopping(valid_loss, model)   
    #test the model
    test_loader = DataLoader(graph_test, batch_size, shuffle=True, collate_fn=collate)
    model.eval()
    test_pred, test_label = [], []
    with torch.no_grad():
        for it, (batchdgl, label) in enumerate(test_loader):
            # batchdgl = Variable(batchdgl.tensor()).cuda()
            # label = Variable(label.tensor()).cuda()
            batchdgl = batchdgl.to(device)
            label = label.to(device)
            emb, prediction = model(batchdgl)
            pred = torch.softmax(prediction, 1)
            pred = torch.max(pred, 1)[1].view(-1)
            test_pred += pred.detach().cpu().numpy().tolist()
            test_label += label.cpu().numpy().tolist()
    # labels = ["real", "ER", "BA"]
    if not os.path.exists("./models"):
        os.mkdir("./models")
    if not os.path.exists("./models/binaryclassifier"):
        os.mkdir("./models/binaryclassifier")
    if not os.path.exists('./models/binaryclassifier/'+datasetname):
        os.mkdir('./models/binaryclassifier/'+datasetname)
    path = './models/binaryclassifier/'+datasetname+output
    torch.save(model, path)
    print("accuracy: ", accuracy_score(test_label, test_pred))
    # print("auc: ", roc_auc_score(test_label, test_pred))
    print("classification report: ", classification_report(test_label, test_pred))
    cm = confusion_matrix(test_label, test_pred)
    print("confusion matrix: ")
    print(cm)

def get_paired_samples(labels, dataset, threshold):
    print('length of dataset:', len(dataset))
    positive_samples = [] # label=1, x1 and x2 belongs to the same label
    negative_samples = [] # label=0, x1 and x2 belongs to different labels
    # num = 0
    # for i in trainset:
    #     for a in range(len(trainset[i])):
    #         for b in range(a, len(trainset[i])):
    #             train_positive_samples.append((trainset[i][a], trainset[i][b], 1))
    #             num += 1
    #         for c in trainset:
    #             if c != i:
    #                 for onedata in trainset[c]:
    #                     train_negative_samples.append((trainset[i][a], onedata, 0))
    #                     num += 1
    #     if num % 100 == 0:
    #         print(num)
    # train_negative_samples = random.sample(train_negative_samples, len(train_positive_samples))
    # trainset_final = train_positive_samples + train_negative_samples
    # randomly choose 2 figures
    # dataset_all = []
    # positive_label_all = []
    # negative_label_all = []
    # for i in range(len(dataset)):
    #     label_real = [i for length in range(len(dataset[i]))]
    #     dataset_all += list(zip(dataset[i],label_real))
    # while True:
    #     graph1 = random.choice(dataset_all)
    #     graph2 = random.choice(dataset_all)
    #     if graph1 != graph2:
    #         if list(graph1)[1] == list(graph2)[1]:
    #             if (list(graph1)[0], list(graph2)[0], 1) and (list(graph2)[0], list(graph1)[0], 1) not in positive_samples:
    #                 positive_samples.append((list(graph1)[0], list(graph2)[0], 1))
    #                 positive_label_all.append((list(graph1)[1], list(graph2)[1]))
    #         else:
    #             if (list(graph1)[0], list(graph2)[0], 0) and (list(graph2)[0], list(graph1)[0], 0) not in negative_samples:
    #                 negative_samples.append((list(graph1)[0], list(graph2)[0], 0))
    #                 negative_label_all.append((list(graph1)[1], list(graph2)[1]))
    #     if len(positive_samples) % 1000 == 0:
    #         print(len(positive_samples), time.asctime(time.localtime(time.time())))
    #     if len(positive_samples) >= threshold:
    #         break
    # negative_samples = random.sample(negative_samples, len(positive_samples))
    # first randomly choose labels, then choose samples with corresponding labels
    i = 0
    while i < threshold:
        posi_label = random.choice(labels)
        # print(posi_label)
        # print(posi_label in dataset)
        # print(dataset[posi_label])
        # print(dataset)
        graph1 = random.choice(dataset[posi_label])
        graph2 = random.choice(dataset[posi_label])
        if graph1 != graph2:
            positive_samples.append((graph1, graph2, 1))
            i += 1
        # if i % 1000 == 0:
        #     print(i, time.asctime(time.localtime(time.time())))
    i = 0
    while i < threshold:
        neg_label1 = random.choice(labels)
        neg_label2 = random.choice(labels)
        if neg_label1 != neg_label2:
            graph1 = random.choice(dataset[neg_label1])
            graph2 = random.choice(dataset[neg_label2])
            negative_samples.append((graph1, graph2, 0))
            i += 1
        # if i % 1000 == 0:
        #     print(i, time.asctime(time.localtime(time.time())))
    return  positive_samples+negative_samples

def deep_metric_learning_2(scenario, labels, trainset, testset, datasetname, dataset_size, batch_size, epoch, depth, loss_func_name, output, device):
    if scenario in ["s1", "s2"]:
        model_path = './models/metriclearning/'+datasetname+"/_" +str(dataset_size) + output
    else:
        model_path = './models/metriclearning/mixed_' + output
    if os.path.exists(model_path):
        print("model already exists")
        return
    device = torch.device('cuda:'+str(device))
    trainset_final = get_paired_samples(labels, trainset, dataset_size)
    print(len(trainset_final))
    # print(trainset_final[:10])
    train_loader = DataLoader(trainset_final, batch_size, shuffle=True, collate_fn=collate_2)
    # train_loader = GraphDataLoader(trainset_final, batch_size, shuffle=True)
    # print(train_loader)
    if depth == 4:
        model = GCNClassifier_for_dml(1, 256, 2)
    if depth == 3:
        model = GCNClassifier_for_dml_3_layers(1, 256, 2)
    if depth == 2:
        model = GCNClassifier_for_dml_2_layers(1, 256, 2)
    model.to(device)
    if loss_func_name == "cross_entropy":
        loss_func = nn.CrossEntropyLoss()
    # if loss_func_name == "contrastive_loss":
    #     loss_func = ContrastiveLoss()
    # loss_func = torch.nn.BCEWithLogitsLoss(size_average=True)
    # loss_func_name = 'cross_entropy'
    # loss_func_name = 'BCE'
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
                if loss_func_name == "BCE":
                    one_hot = torch.zeros(len(label), 2).long()
                    label = label.cpu()
                    one_hot.scatter_(dim=1,index=label.unsqueeze(dim=1),src=torch.ones(len(label), 2).long())
                    label = one_hot.float().to(device)
                if loss_func_name == "contrastive_loss":
                    loss = loss_func(emb1, emb2, label)
                else:
                    loss = loss_func(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                #epoch_loss += loss.depatch().item
            epoch_loss /= (iter+1)
            print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss), time.asctime(time.localtime(time.time())))
            # print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss), time.asctime(time.localtime(time.time())), accuracy_score(label.cpu().numpy().tolist(), pred.detach().cpu().numpy().tolist()))
            epoch_losses.append(epoch_loss)
    
    # test model
    testset_final = get_paired_samples(labels, testset, dataset_size/10)
    test_loader = DataLoader(testset_final, batch_size, shuffle=True, collate_fn=collate_2)
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
    if not os.path.exists("./models"):
        os.mkdir("./models")
    if not os.path.exists("./models/metriclearning"):
        os.mkdir("./models/metriclearning")
    if not os.path.exists('./models/metriclearning/'+datasetname):
        os.mkdir('./models/metriclearning/'+datasetname)
    # path = './models/metriclearning/'+datasetname+"/" + output
    torch.save(model, model_path)

def predict_dml_binary(datasetname, trainset, testset, number_of_samples, dataset_size, output, device):
    device = torch.device('cuda:'+str(device))
    path = './models/metriclearning/'+datasetname+"/_" +str(dataset_size) + output
    # path = './models/metriclearning/'+datasetname+"/"+str(2.1)+'_cross_entropy_' + str(dataset_size) + "_" + str(depth) + "_2ndOrder.pkl"
    # path = './models/metriclearning/'+datasetname+"/"+str(2.1)+'_cross_entropy_' + str(dataset_size) + "_" + str(depth) + ".pkl"
    # if datasetname == 'github_stargazers':
    #     model = torch.load(path, map_location={'cuda:7':'cuda:6'})
    # else:
    #     model = torch.load(path, map_location={'cuda:2':'cuda:6'})
    model = torch.load(path, map_location=lambda storage, loc: storage.cuda(device))
    real_label = []
    pred_label = []
    time_flag = 0
    # lengths = []
    # testset_final = {}
    # if scenario == "s1":
    #     # for i in range(6):
    #     #     lengths.append(len(testset[i]))
    #     # final_length = min(lengths)
    #     # testset_final = {}
    #     # testset_final[0] = random.sample(testset[0], final_length)
    #     # testset_final[0] = testset[0]
    #     for i in range(1,6):
    #         onetestset = random.sample(testset[i], int(len(testset[0])/5))
    #         testset_final[1] += onetestset
    #     testset = testset_final
    # if scenario == "s2":
    #     testset_final[0] = testset[0]
    #     testset[1] = []
    #     for i in range(6,9):
    #         onetestset = random.sample(testset[i], int(len(testset[0])/3))
    #         testset_final[1] += onetestset
    #     testset_final = testset
    for i in range(len(testset)):
        graphs = testset[i]
        # graphs.to(device)
        for one_graph in graphs:
            real_label.append(i)
            one_graph = one_graph.to(device)
            pred = 0
            ref_graphs = random.sample(trainset[0],number_of_samples)
            # ref_graphs = trainset[j]
            for ref_graph in ref_graphs:
                if time_flag%100 == 0:
                    print(time_flag, time.asctime(time.localtime(time.time())))
                time_flag += 1
                ref_graph = ref_graph.to(device)
                emb1, emb2, prediction = model(one_graph, ref_graph)
                pred += torch.softmax(prediction, 1).detach().cpu().numpy().tolist()[0][1]
            if pred/number_of_samples > 0.5:
                # pred_all.append(pred/len(trainset[j]))
                pred_label.append(0) #real graphs
            else:
                pred_label.append(1) # generated graph
    print(real_label)
    print(pred_label)
    print("accuracy: ", accuracy_score(real_label, pred_label))
    print("classification report: ", classification_report(real_label, pred_label))
    cm = confusion_matrix(real_label, pred_label)
    print("confusion matrix: ")
    print(cm)


