import math
import os
import sys
from traceback import print_tb

sys.path.append("./src/")

import random
from functools import cache, partial
from pathlib import Path
from pprint import pprint as pp
from typing import List, Optional, Union

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import tqdm
from joblib import Parallel, delayed

from graph_utils import load_graph
from m5 import find_optimal_partitions_dp


layout_dot = partial(nx.nx_agraph.graphviz_layout, prog="dot", args="-Grankdir=LR")


def plot_tagged_graph(G, pos, fp):
    color_dict = {"center": "red", "top": "green", "bottom": "blue", "unvisited": "white"}
    node_colors = [color_dict[n["tag"]] for _, n in G.nodes(data=True)]
    # nx.draw(G, pos=pos, node_color=node_colors, node_size=50, linewidths=2)
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.drawing.draw_networkx_nodes(G, pos=pos, node_color=node_colors, node_size=100, linewidths=2, ax=ax)
    nx.drawing.draw_networkx_edges(G, pos=pos, alpha=0.5, ax=ax, node_size=100)
    nx.drawing.draw_networkx_labels(G, pos=pos, font_size=8, ax=ax)
    plt.savefig(fp, dpi=300)


def topo_sort_random(G, n=1, seed=None, as_ndarray=False, n_jobs=1):

    if seed is not None:
        random.seed(seed)

    sorts = []

    node_list = list(G.nodes())
    num_nodes = len(node_list)

    

    # for i in range(n):
    #     G_unique = G.copy()
    #     L = []

    #     while len(L) < num_nodes:
    #         start_nodes = [node for node in G_unique.nodes if G_unique.in_degree(node) == 0]
    #         # pick a random start node
    #         random_start_node = random.choice(start_nodes)
    #         L.append(random_start_node)

    #         # remove the start node from the graph
    #         G_unique.remove_node(random_start_node)

    #     if not check_sort(G, L):
    #         raise ValueError("Sort is not a valid topological sort")
    #     sorts.append(L)

    def topo_sort_random_single(G, seed=None, as_ndarray=False):
        G_unique = G.copy()
        L = []

        while len(L) < num_nodes:
            start_nodes = [node for node in G_unique.nodes if G_unique.in_degree(node) == 0]
            # pick a random start node
            random_start_node = random.choice(start_nodes)
            L.append(random_start_node)

            # remove the start node from the graph
            G_unique.remove_node(random_start_node)

        if not check_sort(G, L):
            raise ValueError("Sort is not a valid topological sort")
        return L
    
    sorts = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(topo_sort_random_single)(G, seed=None, as_ndarray=as_ndarray) for i in tqdm.tqdm(range(n))
    )

    if as_ndarray:
        return np.array(sorts)
    else:
        return sorts


    if as_ndarray:
        sorts_np = [np.array(s) for s in sorts]
        return sorts_np
    else:
        return sorts


def topo_sort_random_reverse(G, n=1, seed=None, as_ndarray=False, n_jobs=1):

    if seed is not None:
        random.seed(seed)

    sorts = []

    node_list = list(G.nodes())
    num_nodes = len(node_list)

    # for i in range(n):
    #     G_unique = G.copy()

    #     L = []

    #     while len(L) < num_nodes:
    #         start_nodes = [node for node in G_unique.nodes if G_unique.out_degree(node) == 0]
    #         # pick a random start node
    #         random_start_node = random.choice(start_nodes)
    #         L.append(random_start_node)

    #         # remove the start node from the graph
    #         G_unique.remove_node(random_start_node)

    #     L.reverse()
    #     if not check_sort(G, L):
    #         raise ValueError("Sort is not a valid topological sort")
    #     sorts.append(L)

    def topo_sort_random_reverse_single(G, seed=None, as_ndarray=False):
        G_unique = G.copy()

        L = []

        while len(L) < num_nodes:
            start_nodes = [node for node in G_unique.nodes if G_unique.out_degree(node) == 0]
            # pick a random start node
            random_start_node = random.choice(start_nodes)
            L.append(random_start_node)

            # remove the start node from the graph
            G_unique.remove_node(random_start_node)

        L.reverse()
        if not check_sort(G, L):
            raise ValueError("Sort is not a valid topological sort")
        return L
    
    sorts = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(topo_sort_random_reverse_single)(G, seed=None, as_ndarray=as_ndarray) for i in tqdm.tqdm(range(n))
    )

    if as_ndarray:
        sorts_np = [np.array(s) for s in sorts]
        return sorts_np
    else:
        return sorts


def topo_sort_starting_node(
    G: nx.DiGraph, strating_node: int, seed: Optional[int] = None, as_ndarray: bool = False
) -> Union[List[int], np.ndarray]:

    if seed is not None:
        random.seed(seed)

    G_unique = G.copy()
    L: list[int] = []

    # set all node tags to "unvisited"
    for node in G_unique.nodes:
        G_unique.nodes[node]["tag"] = "unvisited"

    # set starting node tag to "center"
    G_unique.nodes[strating_node]["tag"] = "center"

    def tag_connected_nodes(G, node):
        # compute intersection of successors and predecessors
        successors = set(G.successors(node))
        predecessors = set(G.predecessors(node))
        connected_nodes_intersection = successors.intersection(predecessors)
        if len(connected_nodes_intersection) > 0:
            raise ValueError("Found nodes that are both successors and predecessors, this is not allowed in a DAG")

        for child in G.successors(node):
            if G.nodes[child]["tag"] == "top":
                continue
            else:
                G.nodes[child]["tag"] = "bottom"
        for parent in G.predecessors(node):
            if G.nodes[parent]["tag"] == "bottom":
                continue
            else:
                G.nodes[parent]["tag"] = "top"

    def remove_and_add_to_list(G_unique, L, node):
        # get node data
        node_tag = G_unique.nodes[node]["tag"]

        if node_tag == "unvisited":
            raise ValueError("Node is not tagged as a node that can be removed")

        if node_tag == "center":
            # append to empty list
            if L != []:
                raise ValueError("Center node is not the first node in the list")
            L.append(node)
            # tag connected nodes
            tag_connected_nodes(G_unique, node)
            # remove node from graph
            G_unique.remove_node(node)

        elif node_tag == "top":
            # append to the top of the list
            L.insert(0, node)
            # tag connected nodes
            tag_connected_nodes(G_unique, node)
            # remove node
            G_unique.remove_node(node)

        elif node_tag == "bottom":
            # append to the bottom of the list
            L.append(node)
            # tag connected nodes
            tag_connected_nodes(G_unique, node)
            # remove node
            G_unique.remove_node(node)
        else:
            raise ValueError("Node is not tagged as a node that can be removed")

    def filter_top_nodes(G: nx.DiGraph, possible_nodes: list[int]) -> list[int]:
        top_nodes = []
        for node in possible_nodes:
            node_descendants = set(nx.descendants(G, node))
            node_descendants_tags = [G.nodes[n]["tag"] for n in node_descendants]
            if "top" in node_descendants_tags:
                continue
            else:
                top_nodes.append(node)
        return top_nodes

    def filter_bottom_nodes(G: nx.DiGraph, possible_nodes: list[int]) -> list[int]:
        bottom_nodes = []
        for node in possible_nodes:
            node_ancestors = set(nx.ancestors(G, node))
            node_ancestors_tags = [G.nodes[n]["tag"] for n in node_ancestors]
            if "bottom" in node_ancestors_tags:
                continue
            else:
                bottom_nodes.append(node)
        return bottom_nodes

    count = 0

    # inital_pos = layout_dot(G_unique)
    # os.makedirs("./figures/random_graph_gen/", exist_ok=True)
    # plot_tagged_graph(G_unique, inital_pos, f"./figures/random_graph_gen/{count}.png")

    remove_and_add_to_list(G_unique, L, strating_node)
    while len(L) < len(G.nodes):
        possible_top_nodes = [node for node in G_unique.nodes if G_unique.nodes[node]["tag"] == "top"]
        possible_bottom_nodes = [node for node in G_unique.nodes if G_unique.nodes[node]["tag"] == "bottom"]

        possible_top_nodes_filtered = filter_top_nodes(G_unique, possible_top_nodes)
        possible_bottom_nodes_filtered = filter_bottom_nodes(G_unique, possible_bottom_nodes)

        possible_node = possible_top_nodes_filtered + possible_bottom_nodes_filtered
        # randomly pick a node
        random_node = random.choice(possible_node)
        # remove node from graph and add to list
        remove_and_add_to_list(G_unique, L, random_node)
        count += 1
        # print(f"Progress: {round(count/len(G.nodes)*100, 2)}%")
        # plot_tagged_graph(G_unique, inital_pos, f"./figures/random_graph_gen/{count}.png")

    if not check_sort(G, L):
        raise ValueError("Sort is not a valid topological sort")

    if as_ndarray:
        L_np = np.array(L)
        return L_np
    else:
        return L


def topo_sort_middle(G, n: int = 1, seed: int = None, as_ndarray: bool = False, n_jobs: int = 1):

    if seed is not None:
        random.seed(seed)

    n_sample = 5
    foward_sorts = topo_sort_random(G, n=n_sample, seed=seed, as_ndarray=False)
    reverse_sorts = topo_sort_random_reverse(G, n=n_sample, seed=seed, as_ndarray=False)
    all_sampled_sorts = foward_sorts + reverse_sorts
    middle_index = int(len(all_sampled_sorts[0]) / 2)
    middle_nodes = []
    for sort in all_sampled_sorts:
        middle_nodes.append(sort[middle_index])
    middle_nodes_sorted = sorted(middle_nodes)

    # center_points = [random.choice(middle_nodes_sorted) for _ in range(n)]
    # sorts: list[list[int]] = []
    # for i in range(n):
    #     center_point = random.choice(middle_nodes_sorted)
    #     sort = topo_sort_starting_node(G, center_point, seed=seed, as_ndarray=as_ndarray)
    #     sorts.append(sort)

    center_points = random.choices(middle_nodes_sorted, k=n)
    sorts = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(topo_sort_starting_node)(G, center_point, seed=None, as_ndarray=as_ndarray)
        for center_point in tqdm.tqdm(center_points)
    )

    if as_ndarray:
        sorts_np = [np.array(s) for s in sorts]
        return sorts_np
    else:
        return sorts


def topo_sort_random_start_node(G, n: int = 1, seed: int = None, as_ndarray: bool = False, n_jobs: int = 1):

    if seed is not None:
        random.seed(seed)


    # sorts: list = []
    # for i in range(n):
    #     center_point = random.choice(list(G.nodes))
    #     sort = topo_sort_starting_node(G, center_point, seed=seed, as_ndarray=False)
    #     sorts.append(sort)

    starting_points = random.choices(list(G.nodes), k=n)
    sorts = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(topo_sort_starting_node)(G, starting_point, seed=None, as_ndarray=as_ndarray)
        for starting_point in tqdm.tqdm(starting_points)
    )

    if as_ndarray:
        sorts_np = [np.array(s) for s in sorts]
        return sorts_np
    else:
        return sorts


def check_sort(G, L):
    for i, n in enumerate(L):
        ancestors = list(nx.ancestors(G, n))
        for a in ancestors:
            if a not in L[:i]:
                return False
    return True


def change_pos(in_arr: np.ndarray, pick_idx: int, put_idx: int) -> None:
    range_arr = np.arange(in_arr.size)
    tmp = in_arr[pick_idx]
    in_arr[range_arr != put_idx] = in_arr[range_arr != pick_idx]
    in_arr[put_idx] = tmp


def topo_bubble_mutation(
    G: nx.DiGraph,
    topo_sort: Union[list, np.ndarray],
    seed: Optional[int] = None,
    mutation_factor: int = 4,
) -> list:

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    node_list = np.array(list(G.nodes))
    num_nodes = len(node_list)

    if isinstance(topo_sort, list):
        topo_sort = np.array(topo_sort)

    new_topo_sort = topo_sort.copy()

    nodes = np.random.choice(topo_sort, num_nodes * mutation_factor, replace=True)
    for node in nodes:
        # node=2
        # pp(node)
        node_idx = np.where(new_topo_sort == node)[0][0]
        # pp(node_idx)

        node_ancesters = np.array(list(nx.ancestors(G, node)))
        node_decendants = np.array(list(nx.descendants(G, node)))
        # pp(node_ancesters)
        # pp(node_decendants)
        if node_ancesters.size == 0 or node_decendants.size == 0:
            continue

        # ancester_indexes = np.where(node_ancesters.reshape(node_ancesters.size, 1) == new_topo)[1]
        ancester_indexes = np.nonzero(node_ancesters[:, None] == new_topo_sort)[1]

        # pp(ancester_indexes)
        decendant_indexes = np.where(node_decendants.reshape(node_decendants.size, 1) == new_topo_sort)[1]
        # pp(decendant_indexes)

        lb_idx = np.max(ancester_indexes) + 1
        ub_idx = np.min(decendant_indexes) - 1
        # pp(lb_idx)
        # pp(ub_idx)
        new_index = np.random.randint(lb_idx, ub_idx + 1)

        change_pos(new_topo_sort, node_idx, new_index)

    new_topo_sort_list = new_topo_sort.tolist()
    return new_topo_sort_list


if __name__ == "__main__":

    II = 33
    DSP = 720
    acc_config_enable = True
    T = 5

    mapper_func = find_optimal_partitions_dp
    mapper_func_partial = partial(find_optimal_partitions_dp, II=II, DSP=DSP, acc_config_enable=acc_config_enable)
    mapper_func_cache = cache(mapper_func)

    # resnet_50_adj_list_fp = Path('./src/ResNet-50/ResNet50.adjlist')
    # resnet_50_config_fp = Path('./src/ResNet-50/config_resnet50.csv')
    # G = load_graph(resnet_50_adj_list_fp, resnet_50_config_fp, DSP, II, T)

    vlocnet_adj_list_fp = Path("./src/VLocNet/VLocNet.adjlist")
    vlocnet_config_fp = Path("./src/VLocNet/config_VLocNet.csv")
    G = load_graph(vlocnet_adj_list_fp, vlocnet_config_fp, DSP, II, T)

    # qdtrack_adj_list_fp = Path('./src/QDTrack/QDTrack.adjlist')
    # qdtrack_config_fp = Path('./src/QDTrack/config_QDTrack.csv')
    # G = load_graph(qdtrack_adj_list_fp, qdtrack_config_fp, DSP, II, T)

    total_cost = np.sum(list(G.nodes[node]["cost"] for node in G.nodes))

    print("Generating random topological sorts...")
    sorts = topo_sort_random(G, n=100, seed=0)

    exit()

    print("Computing mappings for random topological sorts...")
    # mappings = [find_optimal_partitions_dp(G, sort, DSP, II, acc_config_enable) for sort in sorts]

    # mappings = []
    # for sort in tqdm.tqdm(sorts):
    #     mappings.append(mapper_func_cache(G, tuple(sort), DSP, II, acc_config_enable))

    # use joblib to parallelize the computation
    mappings = Parallel(n_jobs=8)(
        delayed(mapper_func)(G, sort, DSP, II, acc_config_enable) for sort in tqdm.tqdm(sorts)
    )

    mappings_sizes = [len(m) for m in mappings]

    argmin_mapping = mappings[mappings_sizes.index(min(mappings_sizes))]
    argmin_mapping_size = min(mappings_sizes)
    print(f"Best Mapping: {argmin_mapping}")
    print(f"Best Mapping, # FPGAs: {argmin_mapping_size}")

    sorts_mutated = [topo_bubble_mutation(G, sort, seed=0, mutation_factor=2) for sort in sorts]

    mappings_mutated = Parallel(n_jobs=8)(
        delayed(mapper_func)(G, sort, DSP, II, acc_config_enable) for sort in tqdm.tqdm(sorts_mutated)
    )

    mappings_mutated_sizes = [len(m) for m in mappings_mutated]
    argmin_mapping_mutated = mappings_mutated[mappings_mutated_sizes.index(min(mappings_mutated_sizes))]
    argmin_mapping_mutated_size = min(mappings_mutated_sizes)
    print(f"Best Mapping (mutated): {argmin_mapping_mutated}")
    print(f"Best Mapping (mutated), # FPGAs: {argmin_mapping_mutated_size}")

    # sorts = [tuple(s) for s in sorts]
    # valid_sorts = [check_sort(G, s) for s in sorts]
    # pp(valid_sorts)
    # unique_sorts = set(sorts)
    # print(len(unique_sorts))
