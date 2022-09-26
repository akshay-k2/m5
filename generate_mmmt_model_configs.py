import pandas as pd
import networkx as nx

import math
import csv

import argparse

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--mmmt_model", type=str, required=True,
                        help="Supported models: CASUA_SURF, FaceBagNet, MoCap, QDTrack, ResNet50, VFS, VLocNet")
    
    args       = parser.parse_args()
    mmmt_model = args.mmmt_model
    filename   = "../DNN_Model_Specifications/" + mmmt_model + ".xlsx"
    
    if(mmmt_model == "CASUA_SURF"):
        input_depth  = [   3,   3,   3]
        input_height = [ 360, 360, 360]
        input_width  = [ 360, 360, 360]

    elif(mmmt_model == "CNN_LSTM"):
        input_depth  = [  1,  1, 25]
        input_height = [256, 64, 48]
        input_width  = [256, 64, 36]

    elif(mmmt_model == "FaceBagNet"):
        input_depth  = [   3,   3,   3]
        input_height = [ 360, 360, 360]
        input_width  = [ 360, 360, 360]

    elif(mmmt_model == "MoCap"):
        input_depth  = [0, 0, 3]
        input_height = [100, 500, 200]
        input_width  = [34, 300, 189]

    elif(mmmt_model == "QDTrack"):
        input_depth  = [3]
        input_height = [480]
        input_width  = [640]
        #input_height = [368]
        #input_width  = [640]

    elif(mmmt_model == "ResNet50"):
        input_depth  = [3]
        input_height = [480]
        input_width  = [640]

    elif(mmmt_model == "VFS"):
        input_depth  = [   3,   64,   3,   64]
        input_height = [ 150,   21, 150,   21]
        input_width  = [ 150, 1014, 150, 1014]

    elif(mmmt_model == "VLocNet"):
        input_depth  = [3, 3]
        input_height = [480, 480]
        input_width  = [640, 640]

    else:
        print("MMMT Model not supported. Exiting...")
        exit(1)
    
    mmmt_df = pd.read_excel(open(filename, 'rb'), sheet_name='CNN')
    
    mmmt = mmmt_df.to_dict()
    
    header_row = list(mmmt.keys())
    successor_start_col = header_row.index('Successor_Layers')
    
    # Create adjacency list
    num_of_layers = len(mmmt['Layer_Name'])
    layer_list    = list(mmmt['Layer_Name'].values())
    adjlist       = {}
    
    for i in range(num_of_layers):
        adjlist[i] = []
        for j in range(successor_start_col, len(header_row)):
            child_layer     = mmmt[header_row[j]][i]
            
            #if(not math.isnan(child_layer)):
            if(type(child_layer) == str):
                if(child_layer != 'output'):
            	    child_layer_idx = layer_list.index(child_layer)
            	    # Append successor layer to current layer
            	    adjlist[i].append(child_layer_idx)
    
    # Increment all values by 1 to reserve 0 for virtual source
    adjlist_graph = {}
    for node in adjlist:
        adjlist_graph[node+1] = [j+1 for j in adjlist[node]]
    
    # Convert to graph
    DG = nx.DiGraph(adjlist_graph)
    
    # Write graph to file
    nx.write_adjlist(DG, mmmt_model + "/" + mmmt_model + ".adjlist")
    
    def get_parent_idx(adjlist, child_idx):
        parent_idx = -1
    
        for node in adjlist:
            if(child_idx in adjlist[node]):
                parent_idx = node
                break
    
        return parent_idx
    
    
    # Estimate and assign output dimensions for each layer
    layer_params = {}
    input_idx    = 0
    
    for i in range(num_of_layers):
        layer_params[i] = {}
        
        parent_idx = get_parent_idx(adjlist, i)
        #if(i == 0):
        #print(i, mmmt['Layer_Name'][i], parent_idx)
        if(parent_idx == -1):
            #print(mmmt['Layer_Name'][i])
            layer_params[i]['IC'] = input_depth[input_idx]
            layer_params[i]['IH'] = input_height[input_idx]
            layer_params[i]['IW'] = input_width[input_idx]
            input_idx+=1
        else:
            layer_params[i]['IC'] = layer_params[parent_idx]['OC']
            layer_params[i]['IH'] = layer_params[parent_idx]['OH']
            layer_params[i]['IW'] = layer_params[parent_idx]['OW']
        
        if(mmmt['Layer_Type'][i] in ['conv','maxpool']):
        
            IC = layer_params[i]['IC']
            IH = layer_params[i]['IH']
            IW = layer_params[i]['IW']
            
            K = mmmt['Kernel_Type'][i]
            S = mmmt['Stride'][i]
            P = mmmt['Padding'][i]
            
            # Compute output dimensions
            OH = math.floor((IH - K + 2*P)/S) + 1
            OW = math.floor((IW - K + 2*P)/S) + 1
            
            if(mmmt['Layer_Type'][i] == 'conv'):
            	OC = int(mmmt['Filter_Size'][i])
            else:
            	OC = IC
            
            layer_params[i]['OC'] = OC
            layer_params[i]['OH'] = OH
            layer_params[i]['OW'] = OW
        
        elif(mmmt['Layer_Type'][i] == 'fc'):
            layer_params[i]['OC'] = int(mmmt['Filter_Size'][i])
            layer_params[i]['OH'] = 1
            layer_params[i]['OW'] = 1
        
        elif(mmmt['Layer_Type'][i] in ['add','lstm']):
            layer_params[i]['OC'] = layer_params[i]['IC']
            layer_params[i]['OH'] = layer_params[i]['IH']
            layer_params[i]['OW'] = layer_params[i]['IW']
        
        elif(mmmt['Layer_Type'][i] == 'avgpool'):
            layer_params[i]['OC'] = layer_params[i]['IC']
            layer_params[i]['OH'] = 1
            layer_params[i]['OW'] = 1
        
        elif(mmmt['Layer_Type'][i] == 'intpl'):
            K = mmmt['Kernel_Type'][i]
            layer_params[i]['OC'] = layer_params[i]['IC']
            layer_params[i]['OH'] = K * layer_params[i]['IH'] 
            layer_params[i]['OW'] = K * layer_params[i]['IW']
        	
        elif(mmmt['Layer_Type'][i] == 'concat'):
            N = 0
            for node in adjlist:
                if(i in adjlist[node]):
                    N += 1
            
            layer_params[i]['OC'] = N * layer_params[i]['IC']
            layer_params[i]['OH'] = layer_params[i]['IH'] 
            layer_params[i]['OW'] = layer_params[i]['IW']
    
    # Compute cost of each layer
    layer_costs = {}
    total_cost  = 0
    
    ap_width    = 16 # bits
    BW          = 10**8 # bits per second
    		
    for i in range(num_of_layers):
        if(mmmt['Layer_Type'][i] == 'conv'):
            IC = layer_params[i]['IC']
            OC = layer_params[i]['OC']
            OH = layer_params[i]['OH']
            OW = layer_params[i]['OW']
            K  = mmmt['Kernel_Type'][i]
            
            layer_cost     = OC * OH * OW * IC * K * K
            comm_cost      = (OC * OH * OW * ap_width) / BW
            #layer_cost     = int(layer_cost/(10**6))
            layer_costs[i] = [int(layer_cost), mmmt['Layer_Type'][i] + str(int(K)), comm_cost]
        
        elif(mmmt['Layer_Type'][i] in ['add', 'maxpool', 'avgpool']):
            OC = layer_params[i]['OC']
            OH = layer_params[i]['OH']
            OW = layer_params[i]['OW']
            
            layer_cost     = (OC * OH * OW)/2 # Half MAC
            comm_cost      = (OC * OH * OW * ap_width) / BW
            #layer_cost     = int(layer_cost/(10**6))
            layer_costs[i] = [int(layer_cost), mmmt['Layer_Type'][i], comm_cost]
        	
        elif(mmmt['Layer_Type'][i] == 'fc'):
            IC = layer_params[i]['IC']
            OC = layer_params[i]['OC']
            OH = layer_params[i]['OH']
            OW = layer_params[i]['OW']
            
            layer_cost     = OC * IC
            comm_cost      = (OC * OH * OW * ap_width) / BW
            #layer_cost     = int(layer_cost/(10**6))
            layer_costs[i] = [int(layer_cost), mmmt['Layer_Type'][i], comm_cost]
        
        elif(mmmt['Layer_Type'][i] == 'lstm'):
            C_lstm = mmmt['Filter_Size'][i]
            T      = layer_params[i]['IH']
            F_d    = layer_params[i]['IW']
            OC = layer_params[i]['OC']
            OH = layer_params[i]['OH']
            OW = layer_params[i]['OW']
            
            layer_cost = ((F_d + C_lstm + 1) * 4 * C_lstm + C_lstm) * T
            comm_cost      = (OC * OH * OW * ap_width) / BW
            layer_costs[i] = [int(layer_cost), mmmt['Layer_Type'][i], comm_cost]
        	
        else:
            layer_cost     = 0
            comm_cost      = 0
            layer_costs[i] = [int(layer_cost), mmmt['Layer_Type'][i], comm_cost]
        
        layer_cost = int(layer_cost)
        #print(i, mmmt['Layer_Name'][i], f'{layer_cost:,}')
        total_cost += int(layer_cost)
    
    print("Total computation cost of", mmmt_model, ":", f'{total_cost:,}', "MACs")
    
    # Write to CSV file
    csv_filename = mmmt_model + "/config_" + mmmt_model + ".csv"

    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        for row in layer_costs:
            writer.writerow(layer_costs[row])
