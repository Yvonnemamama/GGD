import math
from posixpath import join
import networkx as nx
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import random
import dgl
from functools import reduce
import pickle
from numpy.core.fromnumeric import _choose_dispatcher
from torch.autograd import Variable
import torch
# from compute_mmd import compute_mmd, gaussian_tv
import time

#use networkx to generate tudataset(excluding node featues and edge features)
def tu2networkx(datasetname): 
    #first make sure if the dataset has been proessed
    print("******Start extract infomation from origin dataset******")
    if not os.path.exists("./datasets/processed/"+datasetname):
        os.mkdir("./datasets/processed/"+datasetname)
    if os.path.exists("./datasets/processed/"+datasetname+"/"+datasetname+".json"):
        with open("./datasets/processed/"+datasetname+"/"+datasetname+".json", "r") as fp:
            node2graph = json.load(fp)
        return node2graph
    node2graph = {} #match about node and graph
    path = "./datasets/"+ datasetname + "/" +datasetname + "_"
    f = open(path+"graph_indicator.txt")
    line = f.readline()
    i = 0
    node_graph_match = {}
    while line:
        if i % 1000 == 0:
            print(i)
        i += 1
        line = int(line.strip("\n"))
        node_graph_match[i] = line
        if line not in node2graph:
            node2graph[line] = {"node":[i]}
        else:
            node2graph[line]["node"].append(i)
        line = f.readline()
    f.close()
    f1 = open(path+"A.txt")
    line = f1.readline()
    i = 0
    while line:
        if i % 1000 == 0:
            print(i)
        i += 1
        line = line.strip("\n")
        line = [int(line.split(",")[0]),int(line.split(",")[1])]

        # too slow, try to optimize it
        # for item in node2graph:
        #     if line[0] in node2graph[item]["node"] and line[1] in node2graph[item]["node"]:
        #         if "edge" not in node2graph[item]:
        #             node2graph[item]["edge"] = [line]
        #         else:
        #             line_reverse = [line[1], line[0]]
        #             if line not in node2graph[item]["edge"] and line_reverse not in node2graph[item]["edge"]:
        #                 node2graph[item]["edge"].append(line)

        # optimizing:  try method 1 ---not working
        # for key, value in node2graph.items():
        #     if line[0] in value["node"] and line[1] in value["node"]:
        #         if value.__contains__("edge"):
        #             line_reverse = [line[1], line[0]]
        #             if line not in value["edge"] and line_reverse not in value["edge"]:
        #                 value["edge"].append(line)
        #         else:
        #             value["edge"] = [line]

        # optimizing:  try method 2 --reduce “for” loop works!
        #first get the corresponding graph of the 2 nodes
        g1 = node_graph_match[line[0]]
        g2 = node_graph_match[line[1]]
        if g1 == g2:
            if line[0] == line[1]:
                continue
            if "edge" not in node2graph[g1]:
                node2graph[g1]["edge"] = [line]
            else:
                line_reverse = [line[1], line[0]]
                if line not in node2graph[g1]["edge"] and line_reverse not in node2graph[g1]["edge"]:
                    node2graph[g1]["edge"].append(line)
        line = f1.readline()
    f1.close()
    with open("./datasets/processed/"+datasetname+"/"+datasetname+".json", "w") as fp:
        json.dump(node2graph, fp, indent=4)
    return node2graph
# save a list of graphs
def save_graph_list(G_list, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(G_list, f)
def gen_avg(expected_avg, n, a, b, class_name, distribution):
    print('******Start getting random list******')
    print(n)
    print("expected_avg:", expected_avg)
    print("graph_number:", n)
    # print("iter_number:", iter_number)
    print("a:", a)
    print("b:", b)
    # if the density of graph varies a lot, there will be a mistick, then try another method to get random list
    try:
        mid = (8*expected_avg-3*a-b)/4
        if class_name == "int":
            random.seed(2)
            half_list = [random.randint(a, int(mid)) for i in range(int(n*3/4))]
            random.seed(2)
            other_half = [random.randint(int(mid), b) for i in range(int(n/4))]
            target = half_list + other_half
            random.seed(2)
            random.shuffle(target)
            return target
        if class_name == "float":
            random.seed(2)
            half_list = [random.uniform(a, mid) for i in range(int(n/2))]
            random.seed(2)
            other_half = [random.uniform(mid, b) for i in range(int(n/2))]
            target = half_list + other_half
            random.seed(2)
            random.shuffle(target)
            return target
    except:
        target = []
        for item in distribution:
            if class_name == "int":
                target += [random.randint((item-1)*10, item*10) for i in range(distribution[item])]
            if class_name == "float":
                target += [random.uniform((item-1)*0.1, item*0.1) for i in range(distribution[item])]
        random.shuffle(target)
        return target
def getaverageoflist(list):
    a = 0
    for item in list:
        a += item
    average = a/len(list)
    return average
#generate er and ba graphs based on the disributions of number of nodes and density
def get_generateddataset(graphtype, datasetname):
    print("******Start generating dataset******")
    dataset = []
    dataset_store = []
    nodeinfo = [] #the max, min, average and standard deviation number of nodes of all the graphes
    densityinfo = [] #the max, min and average density of all the graphes
    node_number_list = [] #the numbers of nodes of all graphes
    density_list = [] #the densities of all graphes 
    node2graph = tu2networkx(datasetname)
    graph_number = len(node2graph)
    for item in node2graph:
        node_number_list.append(len(node2graph[item]["node"]))
        n = len(node2graph[item]["node"])
        N = n*(n-1)/2
        density_list.append(len(node2graph[item]["edge"])/N)
    iter_number = int(graph_number/10) + 1
    node_distribution = {}
    for i in range(1, iter_number+1):
        node_distribution[i] = 0
    for item in node_number_list:
        node_distribution[int(item/10)+1] += 1
    density_distribution = {}
    for j in range(1,11):
        density_distribution[j] = 0
    for item in density_list:
        if int(item) == 1:
            density_distribution[int(item/0.1)] +=1
        else:
            density_distribution[int(item/0.1)+1] +=1
    nodeinfo = [max(node_number_list), min(node_number_list), getaverageoflist(node_number_list), np.std(node_number_list, ddof=1), node_distribution]
    densityinfo = [max(density_list), min(density_list), getaverageoflist(density_list), np.std(density_list, ddof=1), density_distribution]
    if os.path.exists("./datasets/processed/"+datasetname+"/"+datasetname+"_nodeinfo.json"):
        with open("./datasets/processed/"+datasetname+"/"+datasetname+"_nodeinfo.json", "r") as fp:
            nodenumber = json.load(fp)
    else:
        nodenumber = gen_avg(nodeinfo[2], graph_number, nodeinfo[1], nodeinfo[0], "int", nodeinfo[4])
        with open("./datasets/processed/"+datasetname+"/"+datasetname+"_nodeinfo.json", "w") as fp:
            json.dump(nodenumber, fp, indent=4)
    if os.path.exists("./datasets/processed/"+datasetname+"/"+datasetname+"_densityinfo.json"):
        with open("./datasets/processed/"+datasetname+"/"+datasetname+"_densityinfo.json", "r") as fp:
            densitynumber = json.load(fp)
    else:
        densitynumber = gen_avg(densityinfo[2], graph_number, densityinfo[1], densityinfo[0], "float", densityinfo[4])
        with open("./datasets/processed/"+datasetname+"/"+datasetname+"_densityinfo.json", "w") as fp:
            json.dump(densitynumber, fp, indent=4)
    iter_number = min([len(nodenumber), len(densitynumber)])
    for i in range(iter_number):
        node = nodenumber[i]
        density = densitynumber[i]
        if i % 10000 == 0:
            print(i)
        if graphtype == "ER":
            random.seed(2)
            G = nx.random_graphs.erdos_renyi_graph(node, density)
            G = dgl.DGLGraph(G)
            dataset_store.append(G)
        else:
            if int(density*node*(node-1)/2/node) == 0:
                m = 1
            else:
                m = int(density*node*(node-1)/2/node)

            random.seed(2)
            G = nx.random_graphs.barabasi_albert_graph(node, m)
            G = dgl.DGLGraph(G)
            dataset_store.append(G)
        #     if class_num >= 3:
        #         dataset.append((G, 2))
        #     if class_num == 2:
        #         dataset.append((G, 1))
    random.shuffle(dataset_store)
    save_graph_list(dataset_store, "./datasets/processed/"+datasetname+"/"+datasetname+"_"+graphtype+".dat")
    print("******Dataset generated******")
    return dataset_store

def get_features(graphs):
    featureset = []
    flag = 0
    for graph in graphs:
        graph = nx.Graph(graph)
        if flag % 1000 == 0:
            print(flag, time.asctime(time.localtime(time.time())))
        flag += 1
        one_featureset = []
        number_of_nodes = nx.number_of_nodes(graph)
        number_of_edges = nx.number_of_edges(graph)
        density = nx.density(graph)
        # radius = nx.radius(graph)
        # diameter = nx.diameter(graph)
        average_clustering = nx.average_clustering(graph)
        transitivity = nx.transitivity(graph)
        diameter = max([max(j.values()) for (n,j) in nx.shortest_path_length(graph)])
        one_featureset = [number_of_nodes, number_of_edges, density, diameter, average_clustering, transitivity]
        featureset.append(one_featureset)
    return featureset
def standardization(mean, std, data):
    return (data-mean) / std
def eucliDist(A,B):
    return math.sqrt(sum([(a - b)* (a-b) for (a,b) in zip(A,B)]))
# sample dataset
# "./datasets/processed/"+datasetname+"/all_for_sample.dat" contains 
# a dict where key "0" stores real graphs and keys "1", "2",... store 
# graphs generated by different generators
def sample_dataset(datasetname):
    with open("./datasets/processed/"+datasetname+"/all_for_sample.dat", "rb") as f:
        dataset = pickle.load(f)
    real_graphs = dataset[0]
    dataset_length = []
    for i in range(1, len(dataset)):
        # print(len(dataset[i]))
        dataset_length.append(len(dataset[i]))
    print(dataset_length)
    if min(dataset_length) >= len(dataset[0]):
        min_length = len(dataset[0])
    else:
        min_length = min(dataset_length)
    featureset = {}
    for i in range(len(dataset)):
        print(i)
        dataset[i] = random.sample(dataset[i], min_length)
        print(len(dataset[i]))
        featureset[i] = get_features(dataset[i])
    mean = np.mean(featureset[0], axis=0)    
    std = np.std(featureset[0], axis=0)
    distance = {}
    flag = 0
    featureset_copy = featureset.copy()
    # print(featureset[0])
    featureset[0] = standardization(mean, std, featureset[0]).tolist()
    featureset[0] = np.nan_to_num(featureset[0]).tolist()
    print(featureset[0][:10])
    # print(featureset[0])
    for i in range(1,len(dataset)):
        featureset[i] = standardization(mean, std, featureset[i]).tolist()
        featureset[i] = np.nan_to_num(featureset[i]).tolist()
        print(featureset[i][:10])
        distance[i] = []
        for onefeature1 in featureset[i]:
            distance_list = []
            for onefeature2 in featureset[0]:
                if flag % 1000000 == 0:
                    print(flag, time.asctime(time.localtime(time.time())))
                flag += 1
                distance_list.append(eucliDist(onefeature1, onefeature2))
            distance[i].append(min(distance_list))
        one_distance = distance[i].copy()
        one_distance.sort()
        split = int(len(featureset[i])/5)
        max_distance = one_distance[split]
        dataset_i = dataset[i].copy()
        dataset[i] = []
        featureset_copy_i = featureset_copy[i]
        featureset_copy[i] = []
        featureset_i = featureset[i]
        featureset[i] = []
        # print(featureset_i)
        for j in range(len(dataset_i)):
            if distance[i][j] <= max_distance:
                dataset[i].append(dataset_i[j])
                featureset_copy[i].append(featureset_copy_i[j])
                featureset[i].append(featureset_i[j])
    # featureset[0].to_list()
    # featureset_copy[0].to_list()
    with open("./datasets/processed/"+datasetname+"/closedworld_sampled.dat", "wb") as f:
        pickle.dump(dataset, f)
    with open("./datasets/processed/"+datasetname+"/closedworld_feature_sampled_norm.json", "w") as f:
        json.dump(featureset, f, indent=4)
    with open("./datasets/processed/"+datasetname+"/closedworld_feature_sampled.json", "w") as f:
        json.dump(featureset_copy, f, indent=4)
    return dataset, featureset, featureset_copy

def get_sampled_dataset(datasetname, dataset, scenario):
    if scenario == "s1":
        with open("./datasets/processed/"+datasetname+"/closedworld_sampled.dat", "rb") as f:
            graphset = pickle.load(f)
        trainset = {}
        testset = {}
        trainset_final = {}
        testset_final = {}
        for i in range(len(graphset)):
            random.shuffle(graphset[i])
            for j in range(len(graphset[i])):
                graphset[i][j] = dgl.DGLGraph(graphset[i][j])
            trainset[i] = graphset[i][:int(len(graphset[i])*0.8)]
            testset[i] = graphset[i][int(len(graphset[i])*0.8):]
            print(len(trainset[i]))
            print(len(testset[i]))
        trainset_final[0] = trainset[0]
        testset_final[0] = testset[0]
        trainset_final[1] = []
        testset_final[1] = []
        for i in range(1,len(graphset)):
            trainset_final[1] += trainset[i]
            testset_final[1] += testset[i]
            # try:
            #     trainset_final[1] += random.sample(trainset[i],int(len(trainset[0])/5))
            # except:
            #     trainset_final[1] += trainset[i]
            # try:
            #     testset_final[1] += random.sample(testset[i],int(len(testset[0])/5))
            # except:
            #     testset_final[1] += testset[i]
        return trainset, testset, trainset_final, testset_final
    if scenario == "s2":
        with open("./datasets/processed/"+datasetname+"/closedworld_sampled.dat", "rb") as f:
            graphset = pickle.load(f)
        trainset = {}
        testset = {}
        trainset_final = {}
        testset_final = {}
        for i in range(len(graphset)):
            random.shuffle(graphset[i])
            for j in range(len(graphset[i])):
                graphset[i][j] = dgl.DGLGraph(graphset[i][j])
            trainset[i] = graphset[i][:int(len(graphset[i])*0.8)]
            testset[i] = graphset[i][int(len(graphset[i])*0.8):]
        trainset_final[0] = trainset[0]
        trainset_final[1] = []
        testset_final[0] = trainset[0]
        testset_final[1] = []
        for i in range(1,6):
            trainset_final[1] += trainset[i]
            # try:
            #     trainset_final[1] += random.sample(trainset[i],int(len(graphset[0])/5))
            # except:
            #     trainset_final[1] += trainset[i]
        for i in range(6,9):
            # testset_final[1] += testset[i]
            try:
                testset_final[1] += random.sample(testset[i],int(len(graphset[0])/3))
            except:
                testset_final[1] += testset[i]
        return trainset, testset, trainset_final, testset_final
    if scenario == "s3":
        trainset = {}
        testset = {}
        trainset_final = {}
        testset_final = {}
        for onedataset in dataset[:-2]:
            with open("../datasets/processed/"+onedataset+"/closedworld_sampled.dat", "rb") as f:
                one_graphset = pickle.load(f)
            for i in range(len(one_graphset)):
                for j in range(len(one_graphset[i])):
                    one_graphset[i][j] = dgl.DGLGraph(one_graphset[i][j])
                if i not in trainset:
                    try:
                        trainset[i] = random.sample(one_graphset[i], 1600)
                    except:
                        trainset[i] = one_graphset[i]
                else:
                    try:
                        trainset[i] += random.sample(one_graphset[i], 1600)
                    except:
                        trainset[i] += one_graphset[i]
        for onedataset in dataset[-2:]:
            with open("./datasets/processed/"+onedataset+"/closedworld_sampled.dat", "rb") as f:
                one_graphset = pickle.load(f)
            for i in range(len(one_graphset)):
                for j in range(len(one_graphset[i])):
                    one_graphset[i][j] = dgl.DGLGraph(one_graphset[i][j])
                if i not in testset:
                    try:
                        testset[i] = random.sample(one_graphset[i], 1000)
                    except:
                        testset[i] = one_graphset[i]
                else:
                    try:
                        testset[i] += random.sample(one_graphset[i], 1000)
                    except:
                        testset[i] += one_graphset[i]
            trainset_final[0] = trainset[0]
            trainset_final[1] = []
            testset_final[0] = testset[0]
            testset_final[1] = []
            for i in range(1,6):
                trainset_final[1] += trainset[i]
                testset_final[1] += testset[i]
                # try:
                #     trainset_final[1] += random.sample(trainset[i],int(len(trainset[0])/5))
                # except:
                #     trainset_final[1] += trainset[i]
                # try:
                #     testset_final[1] += random.sample(testset[i],int(len(testset[0])/5))
                # except:
                #     testset_final[1] += testset[i]
        return trainset, testset, trainset_final, testset_final
    if scenario == "s4":
        graphset_final = {}
        for onedataset in dataset[:-2]:
            with open("./datasets/processed/"+onedataset+"/closedworld_sampled.dat", "rb") as f:
                one_graphset = pickle.load(f)
            for i in range(len(one_graphset)):
                for j in range(len(one_graphset[i])):
                    one_graphset[i][j] = dgl.DGLGraph(one_graphset[i][j])
                if i not in graphset_final:
                    try:
                        graphset_final[i] = random.sample(one_graphset[i], 1600)
                    except:
                        graphset_final[i] = one_graphset[i]
                else:
                    try:
                        graphset_final[i] += random.sample(one_graphset[i], 1600)
                    except:
                        graphset_final[i] += one_graphset[i]
        trainset = {}
        testset = {}
        trainset_final = {}
        testset_final = {}
        for onedataset in dataset[:-2]:
            with open("./datasets/processed/"+onedataset+"/closedworld_sampled.dat", "rb") as f:
                one_graphset = pickle.load(f)
            for i in range(len(one_graphset)):
                for j in range(len(one_graphset[i])):
                    one_graphset[i][j] = dgl.DGLGraph(one_graphset[i][j])
                if i not in trainset:
                    try:
                        trainset[i] = random.sample(one_graphset[i], 1600)
                    except:
                        trainset[i] = one_graphset[i]
                else:
                    try:
                        trainset[i] += random.sample(one_graphset[i], 1600)
                    except:
                        trainset[i] += one_graphset[i]
        for onedataset in dataset[-2:]:
            with open("./datasets/processed/"+onedataset+"/closedworld_sampled.dat", "rb") as f:
                one_graphset = pickle.load(f)
            for i in range(len(one_graphset)):
                for j in range(len(one_graphset[i])):
                    one_graphset[i][j] = dgl.DGLGraph(one_graphset[i][j])
                if i not in testset:
                    try:
                        testset[i] = random.sample(one_graphset[i], 1000)
                    except:
                        testset[i] = one_graphset[i]
                else:
                    try:
                        testset[i] += random.sample(one_graphset[i], 1000)
                    except:
                        testset[i] += one_graphset[i]
        trainset_final[0] = trainset[0]
        trainset_final[1] = []
        testset_final[0] = trainset[0]
        testset_final[1] = []
        for i in range(1,6):
            trainset_final[1] += trainset[i]
            # try:
            #     trainset_final[1] += random.sample(trainset[i],int(len(trainset[0])/5))
            # except:
            #     trainset_final[1] += trainset[i]
        for i in range(6,9):
            testset_final[1] += testset[i]
            # try:
            #     testset_final[1] += random.sample(testset[i],int(len(trainset[0])/3))
            # except:
            #     testset_final[1] += testset[i]
        return trainset, testset, trainset_final, testset_final
    