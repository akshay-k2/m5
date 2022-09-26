import numpy as np
import networkx as nx
from typing import Union


import tqdm
from joblib import Parallel, delayed


def add_virtual_source_and_sink_nodes(G: nx.DiGraph) -> nx.DiGraph:
    DG = G.copy()

    roots = [x for x in DG.nodes() if DG.in_degree(x) == 0]
    DG.add_node(0)  # Virtual source node index
    DG.nodes[0]["cost"] = 0
    DG.nodes[0]["level"] = 0
    for root in roots:
        DG.add_edge(0, root)

    leaves = [x for x in DG.nodes() if DG.out_degree(x) == 0]
    DG.add_node(-1)
    DG.nodes[-1]["cost"] = 0
    DG.nodes[-1]["level"] = 0
    for leaf in leaves:
        DG.add_edge(leaf, -1)

    return DG


def check_order(DG: nx.DiGraph, order: list) -> bool:
    for node in order:
        for neighbor in DG.neighbors(node):
            if neighbor < 0:
                continue
            elif order.index(neighbor) < order.index(node):
                return False

    return True


def topo_sort_single_list_schedule(G: nx.DiGraph, schedule_type="ALAP", as_ndarray=False):

    DG = add_virtual_source_and_sink_nodes(G)

    for node in DG.nodes:
        DG.nodes[node]["level"] = 0

    TG = DG.copy()

    if schedule_type == "ALAP":
        while 1:
            leaves = [x for x in TG.nodes() if TG.out_degree(x) == 0]
            if leaves == []:
                break

            for leaf in leaves:
                for parent in DG.predecessors(leaf):
                    if DG.nodes[leaf]["level"] + 1 > DG.nodes[parent]["level"]:
                        DG.nodes[parent]["level"] = DG.nodes[leaf]["level"] + 1

                TG.remove_node(leaf)

        num_levels = DG.nodes[0]["level"]  # virtual source has highest level

    elif schedule_type == "ASAP":  # ASAP
        while 1:
            roots = [x for x in TG.nodes() if TG.in_degree(x) == 0]
            if roots == []:
                break

            for root in roots:
                for child in DG.successors(root):
                    if DG.nodes[root]["level"] + 1 > DG.nodes[child]["level"]:
                        DG.nodes[child]["level"] = DG.nodes[root]["level"] + 1

                TG.remove_node(root)

        num_levels = DG.nodes[-1]["level"]  # virtual sink has highest level
    else:
        raise Exception("Unknown schedule type: " + schedule_type)

    level_dict: dict[int, list] = {}

    if schedule_type == "ALAP":
        for l in range(num_levels - 1, 0, -1):
            level_dict[l] = []
    else:
        for l in range(1, num_levels):
            level_dict[l] = []

    for node in DG.nodes:
        if node > 0:
            node_level = DG.nodes[node]["level"]
            level_dict[node_level].append(node)

    DG.remove_node(0)
    DG.remove_node(-1)

    level_order = []

    for nodes in level_dict.values():
        nodes_np = np.array(nodes)
        nodes_shuffled_np = np.random.permutation(nodes_np)
        nodes_shuffled = nodes_shuffled_np.tolist()
        for node in nodes_shuffled:
            level_order.append(node)

    if not check_order(DG, level_order):
        raise Exception("Topological order invalid. Exiting...")

    if as_ndarray:
        return np.array(level_order)
    else:
        return level_order


def topo_sort_list_schedule(G: nx.DiGraph, n=1, seed=None, schedule_type="ALAP", as_ndarray=False, n_jobs=1):
    np.random.seed(seed)

    # sorts = []
    # for i in range(n):
    #     sort = topo_sort_single_list_schedule(G, schedule_type=schedule_type, as_ndarray=as_ndarray)
    #     sorts.append(sort)

    # use joblib to parallelize, also use tqdm to show progress bar, use loky backend
    sorts = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(topo_sort_single_list_schedule)(G, schedule_type=schedule_type, as_ndarray=as_ndarray)
        for i in tqdm.tqdm(range(n))
    )

    # check sorts for duplicates
    # if len(sorts) > 1:
    #     for i in range(len(sorts)):
    #         for j in range(i + 1, len(sorts)):
    #             if np.array_equal(sorts[i], sorts[j]):
    #                 raise Exception("Duplicate sorts found. Exiting...")

    return sorts


def topo_sort_list_schedule_alap(G: nx.DiGraph, n=1, seed=None, as_ndarray=False, n_jobs=1):
    return topo_sort_list_schedule(G, n=n, seed=seed, schedule_type="ALAP", as_ndarray=as_ndarray, n_jobs=n_jobs)


def topo_sort_list_schedule_asap(G: nx.DiGraph, n=1, seed=None, as_ndarray=False, n_jobs=1):
    return topo_sort_list_schedule(G, n=n, seed=seed, schedule_type="ASAP", as_ndarray=as_ndarray, n_jobs=n_jobs)


def topo_sort_cost_guided(DG: nx.DiGraph, as_ndarray=False) -> list:
    TG = DG.copy()

    topo_order = []

    while 1:
        if TG.number_of_nodes() == 0:
            break

        candidate_nodes = [x for x in TG.nodes() if TG.in_degree(x) == 0]
        cost_list = [TG.nodes[node]["cost"] for node in candidate_nodes]

        node_idx = cost_list.index(min(cost_list))
        node = candidate_nodes[node_idx]
        topo_order.append(node)
        TG.remove_node(node)

    if not check_order(DG, topo_order):
        raise Exception("Topological order invalid. Exiting...")

    sorts: list[Union[list, np.ndarray]] = []
    if as_ndarray:
        sorts.append(np.array(topo_order))
    else:
        sorts.append(topo_order)

    return sorts
