# Assumes Python 3.7 or higher and avoids OrderedDict

import argparse
import csv
import math

import networkx as nx

from functools import partial
import itertools
import os
from pathlib import Path
from unittest import result
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print

from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.manifold import TSNE
import tqdm
import pickle

from typing import Callable, Mapping, Union

from random_sampling.topo_sort_random import (
    topo_sort_random,
    topo_sort_random_reverse,
    topo_sort_middle,
    topo_sort_random_start_node,
)
from random_sampling.other_topo_sorts import (
    topo_sort_list_schedule_asap,
    topo_sort_list_schedule_alap,
    topo_sort_cost_guided,
)

import m5
from m5 import find_optimal_partitions_dp, critical_path_mapping

def simple_partial(func: Callable, *args, **kwargs) -> Callable:
    def func_new(*args_new, **kwargs_new):
        return func(*args_new, *args, **kwargs_new, **kwargs)

    return func_new

def gen_topo_orders(G: nx.DiGraph, topo_order_functions: dict[str, Union[Callable, partial]]) -> dict[str, list]:
    topo_orders_data: dict[str, list] = {}
    for name, func in topo_order_functions.items():
        print(f"Generating topo orders: {name}")
        topo_orders_data[name] = func(G)
    return topo_orders_data

def compute_optimal_mappings(G: nx.DiGraph, topo_orders_data: dict[str, list], II, DSP, acc_config_enable, T, n_jobs=8) -> dict[str, list]:
    optimal_mappings: dict[str, list] = {}
    for ordering in topo_orders_data.keys():
        print(f"Computing optimal mappings: {ordering}")
        # optimal_mappings[ordering] = []
        # for topo_order in topo_orders_data[ordering]:
        #     optimal_mappings[ordering].append(find_optimal_partitions_dp(G, topo_order, II, DSP, acc_config_enable))
        optimal_mappings[ordering] = Parallel(n_jobs=n_jobs)(
            delayed(find_optimal_partitions_dp)(G, topo_order, DSP, II, acc_config_enable, T)
            for topo_order in tqdm.tqdm(topo_orders_data[ordering])
        )
    return optimal_mappings

def filter_best_mappings(optimal_mappings: dict[str, list]) -> dict[str, int]:
    best_mappings: dict[str, int] = {}
    for ordering in optimal_mappings.keys():
        sort_lengths = [len(sort.keys()) for sort in optimal_mappings[ordering]]
        best_mappings[ordering] = min(sort_lengths)
    return best_mappings

def find_best_mappings(graph: nx.DiGraph, topo_order_functions, II, DSP, acc_config_enable, T, n_jobs=1):
    topo_orders = gen_topo_orders(graph, topo_order_functions)
    optimal_mappings = compute_optimal_mappings(graph, topo_orders, II, DSP, acc_config_enable, T, n_jobs=n_jobs)
    best_mappings = filter_best_mappings(optimal_mappings)
    return best_mappings

if(__name__ == "__main__"):
    
    #----------------------------------------------------------------
    # Parse input arguments
    #----------------------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument("--DSP", type=int, required=True,
                        help="Number of available DSP units in the target FPGA")
    parser.add_argument("--II", type=int, required=True,
                        help="Initiation interval period in milli-seconds")
    parser.add_argument("--mmmt_model", type=str, required=True,
                        help="Supported models: ResNet50, VLocNet, QDTrack, MoCap, CNN_LSTM, FaceBagNet, VFS, CASUA_SURF")
    
    parser.add_argument("--clock_period", type=float, default=5, 
                        help="Clock period in ns. Default is 5 ns (200 MHz)")
    #parser.add_argument("--acc_config_enable", type=bool, default=False, 
    #                    help="Enable accelerator configuration search")

    args = parser.parse_args()

    print(args)

    DSP_total  = args.DSP
    II         = args.II
    mmmt_model = args.mmmt_model
    
    T                 = args.clock_period if (args.clock_period) else 5
    #acc_config_enable = args.acc_config_enable if (args.acc_config_enable) else False

    supported_models = ["ResNet50", "VLocNet", "QDTrack", "MoCap", "CNN_LSTM", "FaceBagNet", "VFS", "CASUA_SURF"]

    #----------------------------------------------------------------
    # Read adjacency list of chosen MMMT model
    #----------------------------------------------------------------
    if(mmmt_model in supported_models):
        DAG_file = mmmt_model + "/" + mmmt_model + ".adjlist"
    else:
        print("Unsupported model. Exiting...")
        exit()

    DG = nx.read_adjlist(DAG_file, create_using=nx.DiGraph, nodetype=int)

    #----------------------------------------------------------------
    # Read node attributes
    #----------------------------------------------------------------
    config_filename = mmmt_model + "/config_" + mmmt_model + ".csv"
    with open(config_filename,'r') as f:
        f_reader = csv.reader(f)
        i = 1
        for row in f_reader:
            DG.nodes[i]['cost'] = int(row[0])
            DG.nodes[i]['type'] = row[1]
            #DG.nodes[i]['level'] = 0
            for successor in DG.successors(i):
                DG[i][successor]['comm_delay'] = float(row[2])

            i+=1

    # Add edge weights TODO
    #for node_idx in DG.nodes:
    #	for successor in DG.successors(node_idx):
    #        DG[node_idx][successor]['weight'] = 5

    #----------------------------------------------------------------
    # Generate DSP-delay pairs for each node
    # 
    # Assumption: Computation cost is in MMAC units and 
    #             clock period is in nano-seconds. 
    #             The computed delay is in ms, same units as II
    #----------------------------------------------------------------
    for node_idx in DG.nodes:
        node = DG.nodes[node_idx]
        node_cost = node['cost']
        node['DSP-delay'] = {}
        
        # Generate valid range of DSP factors
        if(DSP_total < 256):
            base = 32 # base parallelism factor
            max_DSP = DSP_total
        elif(DSP_total < 512):
            base = 64
            max_DSP = DSP_total
        else:
            base = 128
            max_DSP = 1024 if (DSP_total > 1024) else DSP_total
            #max_DSP = 2048 if (DSP_total > 2048) else DSP_total

        if('conv' in node['type'] or 'fc' in node['type'] or 'lstm' in node['type']):
            for i in range(math.floor(max_DSP/base)):
                node['DSP-delay'][base*(i+1)] = (node_cost/(base*(i+1))) * (T/10**6)
            node['DSP-delay'][max_DSP] = (node_cost/max_DSP) * (T/10**6)
            
            if(node['DSP-delay'][max_DSP] > II):
                print("Node", node, "cannot fit on board.")
                print("Consider splitting the node or using a bigger FPGA or faster clock.")
                print("Exiting...")
                exit()    

        else: # Layers not involving DSP usage
            node['DSP-delay'][0] = node_cost * (T/10**6)
        
    
    best_mappings = {}
    s = 1

    # Disable accelerator configuration
    #acc_config_enable = False

    #print("No. of optimal partitions in various settings are as follows:");

##    #----------------------------------------------------------------
##    # Critical path based mapping (baseline)
##    #
##    # Node ordering: Critical-path
##    # Mapping: Dynamic Programming
##    #----------------------------------------------------------------
##    #description = "Critical-path-based mapping using DP"
##    description = "CP w/o Acc. Config."
##
##    best_mappings[s] = m5.critical_path_mapping(DG, DSP_total, II, acc_config_enable, T)    
##
##    #print("\nSetting " + str(s) + ": " + description)
##    #print("\nBest mapping of setting " + str(s) + ":")
##    #print(best_mappings[s])
##    #print("Number of optimal partitions =", len(best_mappings[s]))
##    print("Setting " + str(s) + ") " + description + ": \t" + str(len(best_mappings[s])))
##    s+=1
##    
##    #----------------------------------------------------------------
##    # ALAP list-schedule-based mapping
##    #
##    # Node ordering: ALAP (level-wise ordering)
##    # Mapping: Level-wise Dynamic Programming 
##    #----------------------------------------------------------------
##    #description = "ALAP list-scheduling-based mapping using DP"
##    description = "ALAP w/o Acc. Config."
##    list_schedule_type = 'ALAP'
##
##    best_mappings[s] = m5.list_schedule_mapping(DG, DSP_total, II, 
##                                                acc_config_enable, 
##                                                list_schedule_type, T)    
##
##    #print("\nSetting " + str(s) + ": " + description)
##    #print("\nBest mapping of setting " + str(s) + ":")
##    #print(best_mappings[s])
##    #print("Number of optimal partitions =", len(best_mappings[s]))
##    print("Setting " + str(s) + ") " + description + ": \t" + str(len(best_mappings[s])))
##    s+=1
##    
##    #----------------------------------------------------------------
##    # ASAP list-schedule-based mapping
##    #
##    # Node ordering: ASAP (level-wise ordering)
##    # Mapping: Level-wise Dynamic Programming 
##    #----------------------------------------------------------------
##    #description = "ASAP list-scheduling-based mapping using DP"
##    description = "ASAP w/o Acc. Config."
##    list_schedule_type = 'ASAP'
##
##    best_mappings[s] = m5.list_schedule_mapping(DG, DSP_total, II, 
##                                                acc_config_enable, 
##                                                list_schedule_type, T)    
##
##    #print("\nSetting " + str(s) + ": " + description)
##    #print("\nBest mapping of setting " + str(s) + ":")
##    #print(best_mappings[s])
##    #print("Number of optimal partitions =", len(best_mappings[s]))
##    print("Setting " + str(s) + ") " + description + ": \t" + str(len(best_mappings[s])))
##    s+=1
##    
####    #----------------------------------------------------------------
####    # Topological order based mapping using DP 
####    #
####    # Node ordering: Topological sort
####    # Mapping: Dynamic Programming
####    #----------------------------------------------------------------
####    description       = "Topological-order-based mapping using DP"
####    topo_order        = list(nx.topological_sort(DG))
####    
####    best_mappings[s] = m5.find_optimal_partitions_dp(DG, topo_order, DSP_total, II, acc_config_enable)
####    
####    print("\nSetting " + str(s) + ": " + description)
####    #print("\nBest mapping of setting " + str(s) + ":")
####    #print(best_mappings[s])
####    print("Number of optimal partitions =", len(best_mappings[s]))
####    s+=1
##
##    #----------------------------------------------------------------
##    # Cost-guided Topological order based mapping using DP 
##    #
##    # Node ordering: Cost-guided greedy Topological sort
##    # Mapping: Dynamic Programming
##    #----------------------------------------------------------------
##    #description       = "Cost-guided topological-order-based mapping using DP"
##    description = "CG w/o Acc. Config."
##    #topo_order        = m5.get_cost_guided_topo_sort(DG)
##    #best_mappings[s] = m5.find_optimal_partitions_dp(DG, topo_order, DSP_total, II, acc_config_enable)
##    best_mappings[s] = m5.cost_guided_mapping(DG, DSP_total, II, T, acc_config_enable)
##    
##    #print("\nSetting " + str(s) + ": " + description)
##    #print("\nBest mapping of setting " + str(s) + ":")
##    #print(best_mappings[s])
##    #print("Number of optimal partitions =", len(best_mappings[s]))
##    print("Setting " + str(s) + ") " + description + ": \t" + str(len(best_mappings[s])))
##    s+=1
##
##    #----------------------------------------------------------------
##    # Repeat with accelerator configuration enabled
##    #----------------------------------------------------------------
##    acc_config_enable = True
##
##    #----------------------------------------------------------------
##    # Baseline with Accelerator Configuration
##    #----------------------------------------------------------------
##    #description = "Critical-path-based mapping using DP with accelerator configuration"
##    description = "CP with Acc. Config."
##
##    best_mappings[s] = m5.critical_path_mapping(DG, DSP_total, II, acc_config_enable, T)    
##
##    #print("\nSetting " + str(s) + ": " + description)
##    #print("\nBest mapping of setting " + str(s) + ":")
##    #print(best_mappings[s])
##    #print("Number of optimal partitions =", len(best_mappings[s]))
##    print("Setting " + str(s) + ") " + description + ": \t" + str(len(best_mappings[s])))
##    s+=1
##    
##    #----------------------------------------------------------------
##    # ALAP list-schedule-based mapping with accelerator configuration
##    #----------------------------------------------------------------
##    #description = "ALAP list-schedule-based mapping using DP with accelerator configuration"
##    description = "ALAP with Acc. Config."
##    list_schedule_type = 'ALAP'
##
##    best_mappings[s] = m5.list_schedule_mapping(DG, DSP_total, II, 
##                                                acc_config_enable, 
##                                                list_schedule_type, T)    
##
##    #print("\nSetting " + str(s) + ": " + description)
##    #print("\nBest mapping of setting " + str(s) + ":")
##    #print(best_mappings[s])
##    #print("Number of optimal partitions =", len(best_mappings[s]))
##    print("Setting " + str(s) + ") " + description + ": \t" + str(len(best_mappings[s])))
##    s+=1
##    
##    #----------------------------------------------------------------
##    # ASAP list-schedule-based mapping with accelator config
##    #----------------------------------------------------------------
##    #description = "ASAP list-scheduling-based mapping using DP with accelerator configuration"
##    description = "ASAP with Acc. Config."
##    list_schedule_type = 'ASAP'
##
##    best_mappings[s] = m5.list_schedule_mapping(DG, DSP_total, II, 
##                                                acc_config_enable, 
##                                                list_schedule_type, T)    
##
##    #print("\nSetting " + str(s) + ": " + description)
##    #print("\nBest mapping of setting " + str(s) + ":")
##    #print(best_mappings[s])
##    #print("Number of optimal partitions =", len(best_mappings[s]))
##    print("Setting " + str(s) + ") " + description + ": \t" + str(len(best_mappings[s])))
##    s+=1
##
####    #----------------------------------------------------------------
####    # Topological order based mapping using DP with accelerator configuration 
####    #----------------------------------------------------------------
####    description       = "Topological-order-based mapping using DP with accelerator configuration"
####    acc_config_enable = True
####    topo_order        = list(nx.topological_sort(DG))
####    
####    best_mappings[s] = m5.find_optimal_partitions_dp(DG, topo_order, DSP_total, II, acc_config_enable)
####    
####    print("\nSetting " + str(s) + ": " + description)
####    #print("\nBest mapping of setting " + str(s) + ":")
####    #print(best_mappings[s])
####    print("Number of optimal partitions =", len(best_mappings[s]))
####    s+=1
##
##    #print(topo_order)
##    #----------------------------------------------------------------
##    # Cost-guided Topological order based mapping using DP with acc config 
##    #
##    # Node ordering: Cost-guided greedy Topological sort
##    # Mapping: Dynamic Programming
##    #----------------------------------------------------------------
##    #description       = "Cost-guided topological-order-based mapping using DP with acc config"
##    description = "CG with Acc. Config."
##    #topo_order        = m5.get_cost_guided_topo_sort(DG)
##    #best_mappings[s] = m5.find_optimal_partitions_dp(DG, topo_order, DSP_total, II, acc_config_enable)
##    best_mappings[s] = m5.cost_guided_mapping(DG, DSP_total, II, T, acc_config_enable)
##    
##
##    #print("\nSetting " + str(s) + ": " + description)
##    #print("\nBest mapping of setting " + str(s) + ":")
##    #print(best_mappings[s])
##    #print("Number of optimal partitions =", len(best_mappings[s]))
##    print("Setting " + str(s) + ") " + description + ": \t" + str(len(best_mappings[s])))
##    s+=1

    results_data = []
    N = 64
    N_JOBS = 16
#    topo_order_functions = {
#        "random_sample": simple_partial(topo_sort_random, n=N, seed=0, as_ndarray=True, n_jobs=N_JOBS),
#        "random_sample_reverse": simple_partial(topo_sort_random_reverse, n=N, seed=0, as_ndarray=True, n_jobs=N_JOBS),
#        "topo_sort_middle": simple_partial(topo_sort_middle, n=N, seed=0, as_ndarray=True, n_jobs=N_JOBS),
#        "topo_sort_random_start_node": simple_partial(topo_sort_random_start_node, n=N, seed=0, as_ndarray=True, n_jobs=N_JOBS),
#        "list_schedule_asap": simple_partial(topo_sort_list_schedule_asap, n=N, seed=0, as_ndarray=True, n_jobs=N_JOBS),
#        "list_schedule_alap": simple_partial(topo_sort_list_schedule_alap, n=N, seed=0, as_ndarray=True, n_jobs=N_JOBS),
#        "cost_guided": simple_partial(topo_sort_cost_guided, as_ndarray=True),
#    }
    
    topo_order_functions = {
        "list_schedule_alap": simple_partial(topo_sort_list_schedule_alap, n=N, seed=0, as_ndarray=True, n_jobs=N_JOBS),
        "list_schedule_asap": simple_partial(topo_sort_list_schedule_asap, n=N, seed=0, as_ndarray=True, n_jobs=N_JOBS),
        "topo_sort_random_start_node": simple_partial(topo_sort_random_start_node, n=N, seed=0, as_ndarray=True, n_jobs=N_JOBS),
    }

    acc_configs = [False, True]
    for acc_config in acc_configs:
        acc_config_enable = acc_config
        best_mappings = find_best_mappings(DG, topo_order_functions, II, DSP_total, acc_config_enable, T, n_jobs=N_JOBS)
        critical_path_mapping_len = len(critical_path_mapping(DG, DSP_total, II, acc_config_enable, T))

        results_data.append({
            "model": mmmt_model,
            "topo_order": "critical_path",
            "II": II,
            "T": T,
            "DSP": DSP_total,
            "acc_config": acc_config_enable,
            "best_solution": critical_path_mapping_len,
        })
        for sort_name, best_solution in best_mappings.items():
            results_data.append({
                "model": mmmt_model,
                "topo_order": sort_name,
                "II": II,
                "T": T,
                "DSP": DSP_total,
                "acc_config": acc_config_enable,
                "best_solution": best_solution,
            })
        # Save all mappings
        pkl_filename = mmmt_model + "/mappings_" + mmmt_model + "_" + str(acc_config) + ".pkl"
        with open(pkl_filename, 'wb') as f:
    	    pickle.dump(best_mappings, f)        

    df = pd.DataFrame(results_data)
    os.makedirs("./results", exist_ok=True)
    output_file_name = "results_" + mmmt_model + "_" + str(DSP_total) + ".csv"
    df.to_csv("./results/" + output_file_name, index=False)
    print(df)

#    # Save all mappings
#    import pickle
#
#    pkl_filename = mmmt_model + "/mappings_" + mmmt_model + ".pkl"
#    with open(pkl_filename, 'wb') as f:
#    	pickle.dump(best_mappings, f)
