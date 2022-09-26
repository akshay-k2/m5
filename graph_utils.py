import networkx as nx
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Union
import csv
import math

from networkx.readwrite.pajek import parse_pajek


def plot_graph(G: nx.DiGraph,
               plot_fp: Union[str, Path, None] = None) -> None:

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    pos = nx.kamada_kawai_layout(G, scale=0.5)
    labels = G.nodes.data()
    # pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
    nx.draw_networkx_nodes(G, pos, node_size=500, ax=ax, node_color='#add8e6',
                           alpha=0.5, edgecolors='#86bfd1', linewidths=2)
    nx.draw_networkx_edges(G, pos, connectionstyle="arc3,rad=0.0", ax=ax, node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax)

    if plot_fp:
        Path(plot_fp)
        plt.savefig(plot_fp, dpi=300)
    else:
        plt.show()


def load_graph(adj_list_fp: Union[str, Path],
               config_fp: Union[str, Path],
               DSP_total, II, T) -> nx.DiGraph:

    adj_list_fp = Path(adj_list_fp)
    config_fp = Path(config_fp)

    G = nx.read_adjlist(adj_list_fp, create_using=nx.DiGraph, nodetype=int)
    assert nx.is_directed_acyclic_graph(G)

    with open(config_fp, 'r') as f:
        f_reader = csv.reader(f)
        i = 1
        for row in f_reader:
            G.nodes[i]['cost'] = int(row[0])
            G.nodes[i]['type'] = row[1]
            i += 1

    for node_idx in G.nodes:
        node = G.nodes[node_idx]
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

    return G

def load_graph_v2(adj_list_fp: Union[str, Path],
               config_fp: Union[str, Path],
               DSP_total, II, T) -> nx.DiGraph:

    adj_list_fp = Path(adj_list_fp)
    config_fp = Path(config_fp)

    G = nx.read_adjlist(adj_list_fp, create_using=nx.DiGraph, nodetype=int)
    assert nx.is_directed_acyclic_graph(G)

    with open(config_fp,'r') as f:
        f_reader = csv.reader(f)
        i = 1
        for row in f_reader:
            G.nodes[i]['cost'] = int(row[0])
            G.nodes[i]['type'] = row[1]
            #DG.nodes[i]['level'] = 0
            for successor in G.successors(i):
                G[i][successor]['comm_delay'] = float(row[2])

            i+=1

    for node_idx in G.nodes:
        node = G.nodes[node_idx]
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

    return G
