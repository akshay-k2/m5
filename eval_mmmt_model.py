# Assumes Python 3.7 or higher and avoids OrderedDict

import argparse
import csv
import math

import networkx as nx

import m5

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
            i+=1

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
    acc_config_enable = False

    #----------------------------------------------------------------
    # Critical path based mapping (baseline)
    #
    # Node ordering: Critical-path
    # Mapping: Dynamic Programming
    #----------------------------------------------------------------
    description = "Critical-path-based mapping using DP"

    best_mappings[s] = m5.critical_path_mapping(DG, DSP_total, II, acc_config_enable)    

    print("\nSetting " + str(s) + ": " + description)
    #print("\nBest mapping of setting " + str(s) + ":")
    #print(best_mappings[s])
    print("Number of optimal partitions =", len(best_mappings[s]))
    s+=1
    
    #----------------------------------------------------------------
    # ALAP list-schedule-based mapping
    #
    # Node ordering: ALAP (level-wise ordering)
    # Mapping: Level-wise Dynamic Programming 
    #----------------------------------------------------------------
    description = "ALAP list-scheduling-based mapping using DP"
    list_schedule_type = 'ALAP'

    best_mappings[s] = m5.list_schedule_mapping(DG, DSP_total, II, 
                                                acc_config_enable, 
                                                list_schedule_type)    

    print("\nSetting " + str(s) + ": " + description)
    #print("\nBest mapping of setting " + str(s) + ":")
    #print(best_mappings[s])
    print("Number of optimal partitions =", len(best_mappings[s]))
    s+=1
    
    #----------------------------------------------------------------
    # ASAP list-schedule-based mapping
    #
    # Node ordering: ASAP (level-wise ordering)
    # Mapping: Level-wise Dynamic Programming 
    #----------------------------------------------------------------
    description = "ASAP list-scheduling-based mapping using DP"
    acc_config_enable = False
    list_schedule_type = 'ASAP'

    best_mappings[s] = m5.list_schedule_mapping(DG, DSP_total, II, 
                                                acc_config_enable, 
                                                list_schedule_type)    

    print("\nSetting " + str(s) + ": " + description)
    #print("\nBest mapping of setting " + str(s) + ":")
    #print(best_mappings[s])
    print("Number of optimal partitions =", len(best_mappings[s]))
    s+=1
    
    #----------------------------------------------------------------
    # Topological order based mapping using DP 
    #
    # Node ordering: Topological sort
    # Mapping: Dynamic Programming
    #----------------------------------------------------------------
    description       = "Topological-order-based mapping using DP"
    topo_order        = list(nx.topological_sort(DG))
    
    best_mappings[s] = m5.find_optimal_partitions_dp(DG, topo_order, DSP_total, II, acc_config_enable)
    
    print("\nSetting " + str(s) + ": " + description)
    #print("\nBest mapping of setting " + str(s) + ":")
    #print(best_mappings[s])
    print("Number of optimal partitions =", len(best_mappings[s]))
    s+=1

    #----------------------------------------------------------------
    # Cost-guided Topological order based mapping using DP 
    #
    # Node ordering: Cost-guided greedy Topological sort
    # Mapping: Dynamic Programming
    #----------------------------------------------------------------
    description       = "Cost-guided topological-order-based mapping using DP"
    #topo_order        = m5.get_cost_guided_topo_sort(DG)
    #best_mappings[s] = m5.find_optimal_partitions_dp(DG, topo_order, DSP_total, II, acc_config_enable)
    best_mappings[s] = m5.cost_guided_mapping(DG, DSP_total, II, T, acc_config_enable)
    
    print("\nSetting " + str(s) + ": " + description)
    #print("\nBest mapping of setting " + str(s) + ":")
    #print(best_mappings[s])
    print("Number of optimal partitions =", len(best_mappings[s]))
    s+=1

    #----------------------------------------------------------------
    # Repeat with accelerator configuration enabled
    #----------------------------------------------------------------
    acc_config_enable = True

    #----------------------------------------------------------------
    # Baseline with Accelerator Configuration
    #----------------------------------------------------------------
    description = "Critical-path-based mapping using DP with accelerator configuration"

    best_mappings[s] = m5.critical_path_mapping(DG, DSP_total, II, acc_config_enable)    

    print("\nSetting " + str(s) + ": " + description)
    #print("\nBest mapping of setting " + str(s) + ":")
    #print(best_mappings[s])
    print("Number of optimal partitions =", len(best_mappings[s]))
    s+=1
    
    #----------------------------------------------------------------
    # ALAP list-schedule-based mapping with accelerator configuration
    #----------------------------------------------------------------
    description = "ALAP list-schedule-based mapping using DP with accelerator configuration"
    list_schedule_type = 'ALAP'

    best_mappings[s] = m5.list_schedule_mapping(DG, DSP_total, II, 
                                                acc_config_enable, 
                                                list_schedule_type)    

    print("\nSetting " + str(s) + ": " + description)
    #print("\nBest mapping of setting " + str(s) + ":")
    #print(best_mappings[s])
    print("Number of optimal partitions =", len(best_mappings[s]))
    s+=1
    
    #----------------------------------------------------------------
    # ASAP list-schedule-based mapping with accelator config
    #----------------------------------------------------------------
    description = "ASAP list-scheduling-based mapping using DP with accelerator configuration"
    list_schedule_type = 'ASAP'

    best_mappings[s] = m5.list_schedule_mapping(DG, DSP_total, II, 
                                                acc_config_enable, 
                                                list_schedule_type)    

    print("\nSetting " + str(s) + ": " + description)
    #print("\nBest mapping of setting " + str(s) + ":")
    #print(best_mappings[s])
    print("Number of optimal partitions =", len(best_mappings[s]))
    s+=1

    #----------------------------------------------------------------
    # Topological order based mapping using DP with accelerator configuration 
    #----------------------------------------------------------------
    description       = "Topological-order-based mapping using DP with accelerator configuration"
    acc_config_enable = True
    topo_order        = list(nx.topological_sort(DG))
    
    best_mappings[s] = m5.find_optimal_partitions_dp(DG, topo_order, DSP_total, II, acc_config_enable)
    
    print("\nSetting " + str(s) + ": " + description)
    #print("\nBest mapping of setting " + str(s) + ":")
    #print(best_mappings[s])
    print("Number of optimal partitions =", len(best_mappings[s]))
    s+=1

    #print(topo_order)
    #----------------------------------------------------------------
    # Cost-guided Topological order based mapping using DP with acc config 
    #
    # Node ordering: Cost-guided greedy Topological sort
    # Mapping: Dynamic Programming
    #----------------------------------------------------------------
    description       = "Cost-guided topological-order-based mapping using DP with acc config"
    #topo_order        = m5.get_cost_guided_topo_sort(DG)
    #best_mappings[s] = m5.find_optimal_partitions_dp(DG, topo_order, DSP_total, II, acc_config_enable)
    best_mappings[s] = m5.cost_guided_mapping(DG, DSP_total, II, T, acc_config_enable)
    

    print("\nSetting " + str(s) + ": " + description)
    #print("\nBest mapping of setting " + str(s) + ":")
    #print(best_mappings[s])
    print("Number of optimal partitions =", len(best_mappings[s]))
    s+=1

    # Save all mappings
    import pickle

    pkl_filename = mmmt_model + "/mappings_" + mmmt_model + ".pkl"
    with open(pkl_filename, 'wb') as f:
    	pickle.dump(best_mappings, f)
