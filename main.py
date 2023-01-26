import torch
import argparse
from train import *
# import GraphCL

if __name__ == "__main__":
    torch.set_num_threads(200)
    datasetname1 = ["AIDS", "alchemy_full", "deezer_ego_nets", "DBLP_v1", "github_stargazers"]
    datasetname2 = ["COLLAB", "twitch_egos"]
    datasetname3 = ["all"]
    labels = [0,1,2,3,4,5]
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--device", help="choose GPU", default=0)
    parser.add_argument("-e", "--experiment", help="choose experiment type", default="train_models", type=str, choices=["sample_dataset", "train_models"], required='True')
    parser.add_argument("-o", "--output", help="choose outputfile", default=".pkl", required='True')
    parser.add_argument("-d", "--dataset", help="choose dataset", default="AIDS", required='True')
    parser.add_argument("-s", "--scenario", help="choose scenario", default="s1", choices=["s1", "s2", "s3", "s4"], required='True')
    args = parser.parse_args()
    device = 0
    if args.experiment == "Generate_ER":
        get_generateddataset["ER", args.dataset]
    if args.experiment == "Generate_BA":
        get_generateddataset["BA", args.dataset]
    if args.experiment == "sample_dataset":
        all_dataset = datasetname1+datasetname2+datasetname3
        # print("dataset: ", dataset)
        sample_dataset(args.dataset)
    if args.experiment == "train_models":
        algorithm_list = ['real', 'ER', 'BA', "graphite","vgae", 'GraphRNN_RNN']
        algorithm_list_openworld = ["GRAN", "GraphRNN_VAE_conditional", "sbmgnn"]
        all_dataset = datasetname1+datasetname2
        if args.scenario == "s3" or args.scenario == "s4":
            args.dataset = "all"
        paired_samples = 200000
        reference_sample = 10
        print("dataset: ", args.dataset)
        print("feature classification:")
        print(args.scenario)
        feature_classifier_sampled(args.dataset, all_dataset, args.scenario)
        print("GCN")
        print(args.scenario)
        trainset, testset, trainset_final, testset_final = get_sampled_dataset(args.dataset, all_dataset, args.scenario)
        GCN_sampled(trainset_final, testset_final, args.dataset, 200, 64, algorithm_list, args.output, device)
        print("metric learning")
        paired_samples = 200000
        reference_sample = 10
        print(args.scenario)
        trainset, testset, trainset_final, testset_final = get_sampled_dataset(args.dataset, all_dataset, args.scenario)
        deep_metric_learning_2(args.scenario, labels, trainset, testset, args.dataset, paired_samples, 64, 200, 4, "cross_entropy", args.output, device)
            # for reference in reference_sample:
            #     print("reference_sample: ", reference)
        predict_dml_binary(args.dataset, trainset_final, testset_final, 10, paired_samples, args.output, device)
        print("Contrastive learning")
        print(args.scenario)
        trainset, testset, trainset_final, testset_final = get_sampled_dataset(args.dataset, all_dataset, args.scenario)
        # GraphCL.main(args.scenario, args.dataset, trainset_final, testset_final, device)