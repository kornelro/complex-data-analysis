from typing import Dict, List

import networkx as nx
import pandas as pd
from sklearn_hierarchical_classification.constants import ROOT


def read_hierarchy(path: str) -> nx.DiGraph:
    return nx.read_edgelist(path, create_using=nx.DiGraph)


def read_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        delimiter=' ', header=None
    )
    df = df.rename(columns={0: 'label'})
    df = df.drop(columns=[81])
    df['path'] = df.apply(lambda row: row['label'].split(','), axis=1)
    df['label'] = df.apply(lambda row: row['label'].split(',')[-1], axis=1)
    for c in df.columns:
        if c not in ('path', 'label'):
            df[c] = df.apply(lambda row: float(row[c].split(':')[1]), axis=1)

    return df


def ensure_is_directed_tree(G: nx.Graph):

    if not isinstance(G, nx.DiGraph):
        raise TypeError('G must be a directed graph!')
    if not nx.is_tree(G):
        raise TypeError('G must be a tree!')


def get_root(G: nx.DiGraph) -> str:
    return [n for n, d in G.in_degree() if d == 0][0]


def get_leafs(G: nx.DiGraph) -> List[str]:
    return [n for n in G.nodes() if len(list(G.neighbors(n))) == 0]


def get_levels(G: nx.DiGraph) -> Dict[int, List[str]]:

    ensure_is_directed_tree(G)

    root = get_root(G)
    levels = {}
    for n in G.nodes:
        paths = list(nx.all_simple_edge_paths(G, root, n))
        level = 0
        if len(paths) > 0:
            level = len(paths[0])
        try:
            levels[level].append(n)
        except KeyError:
            levels[level] = []
            levels[level].append(n)

    return levels


def get_node_level(
    node: str,
    levels: Dict[int, List[str]]
) -> int:
    for k, v in levels.items():
        if node in v:
            return k


def hierarchy_to_dict(
    hierarchy: nx.DiGraph
) -> Dict[str, List[str]]:
    hierarchy_dict = {}
    root = get_root(hierarchy)
    for node in hierarchy.nodes():
        child_nodes = list(hierarchy.neighbors(node))
        if node == root:
            node = ROOT
        if len(child_nodes) > 0:
            hierarchy_dict[node] = child_nodes

    return hierarchy_dict


def split_data(
    data: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 123
) -> List[pd.DataFrame]:

    test_data = data.sample(
        int(len(data)*test_size),
        random_state=random_state
    )
    train_data = data.drop(test_data.index)

    return [train_data, test_data]
