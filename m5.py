import networkx as nx


#----------------------------------------------------------------
# Find valid DSP-delay configurations between two given nodes
#----------------------------------------------------------------
def find_valid_configs(TG, curr_node, curr_group_dsp, curr_group_delay, 
                       curr_node_idx, next_node, next_node_idx, 
                       DSP_total, II, acc_config_enable, offset):
    #breakpoint()
    valid_dsp_list   = []
    valid_delay_list = []
    
    next_node_dsp    = list(next_node['DSP-delay'].keys())
    next_node_delay  = list(next_node['DSP-delay'].values())

    if(not acc_config_enable):
        delay_func = ['sum'] # Assumes dependent nodes
        dsp_func   = ['sum'] # Assumes no resource sharing
    else:
        if(next_node_idx in TG.successors(curr_node_idx)): # Connected
            delay_func = ['sum']
            if(curr_node['type'] == next_node['type']): # Same layer type, so resources shared
                dsp_func = ['max']
            else: # Different type, separate resources
                dsp_func = ['sum']
        else: # Unconnected nodes
            if(curr_node['type'] == next_node['type']): # Same layer type
                delay_func = ['sum','max']
                dsp_func   = ['max','sum']
            else:
                delay_func = ['max']
                dsp_func   = ['sum']
    
    for m in range(len(curr_group_dsp)):
        for n in range(len(next_node_dsp)):
            for f in range(len(dsp_func)):
                if(dsp_func[f] == 'sum'):
                    total_util = curr_group_dsp[m] + next_node_dsp[n]
                else:
                    total_util = max(curr_group_dsp[m], next_node_dsp[n])
                
                if(delay_func[f] == 'sum'):
                    total_delay = curr_group_delay[m] + next_node_delay[n]
                else:
                    total_delay = max(curr_group_delay[m], next_node_delay[n])

                # Added Sep 3
                total_delay += offset
                
                if(total_util < DSP_total and total_delay < II):
                    valid_dsp_list.append(total_util)
                    valid_delay_list.append(total_delay)

    return valid_dsp_list, valid_delay_list

#----------------------------------------------------------------
# Estimate sum of latencies between given limits and latency list
#----------------------------------------------------------------
def latency_sum(latency_list, start_idx, end_idx):
    lsum = 0
    for l in range(start_idx, end_idx+1):
            lsum += latency_list[l]
    return lsum

#----------------------------------------------------------------
# Find maximum latency using table look up 
#----------------------------------------------------------------
def find_max_latency(L, latency_list, r, j, k):
    return max(L[k-1][r], latency_sum(latency_list, r+1, j))

#----------------------------------------------------------------
# DP Solver
# 
# Performs polynomial-time load-balancing of layers onto FPGAs
# using dynamic programming.
#
# Only the computation cost of a layer is considered for mapping.
# The purpose of this function to provide cost-balanced partitions
# for evaluation.
# 
# First, all layers are mapped onto a single FPGA. Then,
# the optimal cost-balanced mapping is for more number of FPGAs,
# iteratively, upto N FPGAs (atleast one layer per FPGA)
#----------------------------------------------------------------
def dp_solver(G, topo_order, DSP_total, II):
    N = len(topo_order)
    M = N

    # Initialize look up table for DP
    L = [[0]*N for i in range(M)]
    
    # Create list of node costs (computational complexity) in MMAC
    C = []
    for node_idx in topo_order:
        C.append(G.nodes[node_idx]['cost'])
   
    # Initialize mappings
    mappings = {}
   
    # Map all layers on single FPGA
    k = 0
    mappings[k+1] = {}
    for j in range(N):
        L[k][j] = latency_sum(C, 0, j)
    
    mappings[k+1][1] = [topo_order[x] for x in range(j+1)]
    #print("Critical Path Cost for mapping", N, "layers onto", k+1, "FPGAs is", L[k][N-1])

    # Map onto multiple FPGAs incrementally
    for k in range(1, M):
        #breakpoint() 
        for j in range(k, N):
            max_list = []
            
            if(j == k):
                L[k][j] = find_max_latency(L, C, k-1, j, k)
                continue
            
            for r in range(k-1,j):
                max_list.append(find_max_latency(L, C, r, j, k))
            
            L[k][j] = min(max_list)
    
        #print("Critical Path Cost for mapping", N, "layers onto", k+1, "FPGAs is", L[k][N-1])

        # Backtrace to find a valid mapping
        mappings[k+1] = {}
        for t in range(k):
            mappings[k+1][t+1] = []

        t = k
        j = N - 1
        while(1):
            if(t == 0):
                mappings[k+1][t+1] = [topo_order[x] for x in range(j+1)]
                break
            
            for r in range(j-1, t-2, -1):
                if(L[t][j] == find_max_latency(L, C, r, j, t)):
                    #if(L[t][j] == L[t-1][r] or L[t][j] == latency_sum(C,r+1,j)):
                    mappings[k+1][t+1] = [topo_order[x] for x in range(r+1,j+1)]
                    t -= 1
                    j = r
                    break

    return mappings

#----------------------------------------------------------------
# Find overall cost of each partition
#----------------------------------------------------------------
def cost_of_maps(DG, mappings, M):
    costs = [0]*M
    for part_num in mappings[M]:
        group_cost = 0 
        for node in mappings[M][part_num]:
            group_cost += DG.nodes[node]['cost']
        costs[part_num-1] = group_cost
    #print(costs)

    return costs

def get_comm_cost(DG, critical_part):
    offset = 0

    for j in range(len(critical_part)):
        curr_node_idx = critical_part[j]
        curr_node     = DG.nodes[curr_node_idx]

        for u, v, data in DG.in_edges(curr_node_idx, data=True):
            delay = data['comm_delay']
            if(delay > offset):
                offset = delay

    return offset

#----------------------------------------------------------------
# Use dynamic programming to obtain the minimal number of 
# partitions of given DAG evaluated using the given
# node ordering and the corresponding node mapping
#----------------------------------------------------------------
def find_optimal_partitions_dp(G, node_order, DSP_total, II, acc_config_enable, T):
    # Maximum number of partitions is same the number of nodes 
    # in the node ordering as each partition must contain 
    # at least one node
    max_num_of_partitions = len(node_order)
  
    # Upper bound on cost that can fit on given board
    cost_bound = (II * DSP_total * 10**6) / T

    # Use DP solver to find all mappings of given graph
    all_mappings = dp_solver(G, node_order, DSP_total, II)
    
    for M in range(max_num_of_partitions,0,-1):
        mapping_costs      = cost_of_maps(G, all_mappings, M)
        
        critical_part_cost = max(mapping_costs)
        critical_part_num  = mapping_costs.index(max(mapping_costs))
        critical_part      = all_mappings[M][critical_part_num+1]
        num_of_layers      = len(critical_part)
        
        # If critical part cost exceed the max cost that can fit on board,
        # no valid configuration exists. So skip.
        if(critical_part_cost > cost_bound):
            continue

        if(num_of_layers == 1):
            best_mapping = all_mappings[M]
            continue

        delay_offset = get_comm_cost(G, critical_part)
        
        curr_node_idx    = critical_part[0]
        curr_node        = G.nodes[curr_node_idx]
        curr_group_dsp   = list(curr_node['DSP-delay'].keys())
        curr_group_delay = list(curr_node['DSP-delay'].values())
        group_nodes_processed = False
            
        for j in range(1, len(critical_part)): 
            next_node_idx   = critical_part[j]
            next_node       = G.nodes[next_node_idx]

            valid_dsp_list, valid_delay_list = find_valid_configs(G,
                                                                  curr_node, 
                                                                  curr_group_dsp, 
                                                                  curr_group_delay,
                                                                  curr_node_idx,
                                                                  next_node,
                                                                  next_node_idx,
                                                                  DSP_total,
                                                                  II, 
                                                                  acc_config_enable,
                                                                  delay_offset)

            if(len(valid_dsp_list) == 0): # Adding next node exceeds II or DSPs
                break
            else:
                curr_group_dsp   = valid_dsp_list.copy()
                curr_group_delay = valid_delay_list.copy()
                
                group_nodes_processed = True if (j == len(critical_part)-1) else False

        if(group_nodes_processed):
            best_mapping = all_mappings[M]    

    return best_mapping

#----------------------------------------------------------------
# Cost of given path in the DAG
#----------------------------------------------------------------
def path_cost(DG, path):
    cost = 0
    for node in path:
        if(node > 0):
            cost += max(DG.nodes[node]['DSP-delay'].values())
    
    return cost

#----------------------------------------------------------------
# Critical path aware mapping (depth-wise)
#----------------------------------------------------------------
def add_virtual_source_and_sink_nodes(G):
    DG = G.copy()

    roots = [x for x in DG.nodes() if DG.in_degree(x) == 0]
    DG.add_node(0) # Virtual source node index
    DG.nodes[0]['cost'] = 0
    DG.nodes[0]['level'] = 0
    for root in roots:
        DG.add_edge(0,root)

    leaves = [x for x in DG.nodes() if DG.out_degree(x)==0]
    DG.add_node(-1)
    DG.nodes[-1]['cost'] = 0
    DG.nodes[-1]['level'] = 0
    for leaf in leaves:
        DG.add_edge(leaf,-1)    

    return DG

#----------------------------------------------------------------
# Critical path aware mapping (depth-wise)
#----------------------------------------------------------------
def critical_path_mapping(G, DSP_total, II, acc_config_enable, T):
    
    # Add a virtual source node and a virtual sink node
    DG = add_virtual_source_and_sink_nodes(G)

    # Find critical path (longest path)
    critical_path       = nx.dag_longest_path(DG)
    critical_path_delay = path_cost(DG, critical_path)

    DG.remove_node(0)
    DG.remove_node(-1)
    
    #print(critical_path)
    critical_path.remove(0)
    critical_path.remove(-1)

    # Get the subgraph with critical path to reuse mapping function
    CG = DG.subgraph(critical_path)

    # Find best critical path mapping
    mappings = []
    mapping  = find_optimal_partitions_dp(CG, critical_path, DSP_total, II, acc_config_enable, T)
    mappings.append(mapping)

    #print("\nBest Critical Path Mapping:")
    #print(mappings[0])

    # Map non-critical residual nodes
    residual_node_list = list(set(list(DG.nodes)) - set(critical_path))
    
    # Get subgraph with secondary critical path
    SG = DG.subgraph(residual_node_list)
    p  = 1

    while(1):
        critical_path       = nx.dag_longest_path(SG)
        critical_path_delay = path_cost(SG, critical_path)
    
        # Find best mapping
        mapping  = find_optimal_partitions_dp(SG, critical_path, DSP_total, II, acc_config_enable, T)
        mappings.append(mapping)

        #print("\nNon-Critical Path Mapping:")
        #print(mappings[p])
        p+=1

        # Move to next critical path in the subgraph
        residual_node_list = list(set(list(SG.nodes)) - set(critical_path))
        SG = SG.subgraph(residual_node_list)

        # Exit if all paths processed
        if(SG.number_of_nodes() == 0):
            break
    
    best_mapping = {}
    P = 0
    for p in range(len(mappings)):
        for map_idx in mappings[p]:
            best_mapping[P + map_idx] = mappings[p][map_idx]
        P += len(mappings[p])
    
    #total_num_of_partitions = 0 
    #for p in range(len(mappings)):
    #    total_num_of_partitions += len(mappings[p])
    #print("\nTotal number of partitions:", total_num_of_partitions) 

    return best_mapping

# Check if given order is a valid topological order
def check_order(DG, order):
    for node in order:
        for neighbor in DG.neighbors(node):
            if(neighbor < 0):
                continue
            elif(order.index(neighbor) < order.index(node)):
                return False

    return True

#----------------------------------------------------------------
# List schedule based mapping (breadth-wise)
#----------------------------------------------------------------
def list_schedule_mapping(G, DSP_total, II, acc_config_enable, schedule_type, T):

    # Add a virtual source node and a virtual sink node
    DG = add_virtual_source_and_sink_nodes(G)

    # Initialize levels of each node
    for node in DG.nodes:
        DG.nodes[node]['level'] = 0

    # Update levels based on ASAP/ALAP scheduling
    TG = DG.copy()
    
    if(schedule_type == 'ALAP'):
        while(1):
            leaves = [x for x in TG.nodes() if TG.out_degree(x)==0]
            if(leaves == []):
                break

            for leaf in leaves:
                for parent in DG.predecessors(leaf):
                    if(DG.nodes[leaf]['level'] + 1 > DG.nodes[parent]['level']):
                        DG.nodes[parent]['level'] = DG.nodes[leaf]['level'] + 1

                TG.remove_node(leaf)
        
        num_levels = DG.nodes[0]['level'] # virtual source has highest level

    else: #ASAP
        while(1):
             roots = [x for x in TG.nodes() if TG.in_degree(x)==0]
             if(roots == []):
                 break

             for root in roots:
                 for child in DG.successors(root):
                    if(DG.nodes[root]['level'] + 1 > DG.nodes[child]['level']):
                        DG.nodes[child]['level'] = DG.nodes[root]['level'] + 1
                
                 TG.remove_node(root)
        
        num_levels = DG.nodes[-1]['level'] # virtual sink has highest level

    # Create dictionary with levels as keys and nodes as values
    level_dict = {}
    
    if(schedule_type == 'ALAP'):
        for l in range(num_levels-1, 0, -1):
            level_dict[l] = []
    else:
        for l in range(1, num_levels):
            level_dict[l] = []
    
    for node in DG.nodes:
        if(node > 0):
            node_level = DG.nodes[node]['level']
            level_dict[node_level].append(node)

    # Remove virtual source and sink nodes
    DG.remove_node(0)
    DG.remove_node(-1)

    ##level_mappings = []

    ### Find best mapping at each level
    ##for level in level_dict:
    ##    LG       = DG.subgraph(level_dict[level])
    ##    mapping  = find_optimal_partitions_dp(LG, level_dict[level], DSP_total, II, acc_config_enable)
    ##    level_mappings.append(mapping)        

    ### Merge all partitions
    ##best_mapping = {}
    ##P = 0
    ##
    ##for p in range(len(level_mappings)):
    ##    for map_idx in level_mappings[p]:
    ##        best_mapping[P + map_idx] = level_mappings[p][map_idx]
    ##    P += len(level_mappings[p])

    level_order = []

    for nodes in level_dict.values():
        for node in nodes:
            level_order.append(node)

    if(not check_order(DG, level_order)):
        print("Topological order invalid. Exiting...")
        print(level_order)
        exit()

    #print(level_order)

    best_mapping = find_optimal_partitions_dp(DG, level_order, DSP_total, II, acc_config_enable, T) 
    
    return best_mapping

# Alternatively choose between max and min costs of eligible nodes
# while obtaining the topological sort
def get_cost_guided_topo_sort(DG):
	TG = DG.copy()

	pick_max_cost = False	

	topo_order = []

	while(1):
		if(TG.number_of_nodes() == 0):
			break

		candidate_nodes = [x for x in TG.nodes() if TG.in_degree(x) == 0]
		cost_list       = [TG.nodes[node]['cost'] for node in candidate_nodes]
		
		#if(pick_max_cost):
		#	pick_max_cost = False
		#	node_idx      = cost_list.index(max(cost_list))
		#else:
		#	pick_max_cost = True
		#	node_idx      = cost_list.index(min(cost_list))
		
		node_idx = cost_list.index(min(cost_list))
		node = candidate_nodes[node_idx]
		topo_order.append(node)
		TG.remove_node(node)

	# Check if topo sort is valid
	if(not check_order(DG, topo_order)):
		print("Topological order invalid. Exiting...")
		print(topo_order)
		exit()

	#print(topo_order)
	#print("Default:")
	#print(list(nx.topological_sort(DG)))
	#exit()
	return topo_order

def cost_guided_mapping(DG, DSP_total, II, T, acc_config_enable):
	TG = DG.copy()

	cost_balance = (DSP_total * II * 10**6)/T

	topo_order = []

	while(1):
		if(TG.number_of_nodes() == 0):
			break

		candidate_nodes = [x for x in TG.nodes() if TG.in_degree(x) == 0]
		cost_list       = [TG.nodes[node]['cost'] for node in candidate_nodes]
		
		cost_difference = [cost_balance - cost_list[x] for x in range(len(cost_list))]
		
		if(all(bal < 0 for bal in cost_difference)): # Reset balance
			cost_balance = (DSP_total * II * 10**6)/T
			cost_difference = [cost_balance - cost_list[x] for x in range(len(cost_list))]

		closest_cost    = min([i for i in cost_difference if i > 0])
		node_idx = cost_difference.index(closest_cost)
		cost_balance -= cost_list[node_idx]
		
		node = candidate_nodes[node_idx]
		topo_order.append(node)
		TG.remove_node(node)
	
	# Check if topo sort is valid
	if(not check_order(DG, topo_order)):
		print("Topological order invalid. Exiting...")
		print(topo_order)
		exit()
	
	mapping = {}

	mapping = find_optimal_partitions_dp(DG, topo_order, DSP_total, II, acc_config_enable, T)

	return mapping 
