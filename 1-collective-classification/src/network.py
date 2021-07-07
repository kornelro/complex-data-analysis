from copy import deepcopy
from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd


def build_graph(
    data: pd.DataFrame,  # columns: Sender, Recipient, EventDate
    min_emails: int = 0
) -> nx.Graph:

    # count emails
    data = data.groupby(['Sender', 'Recipient']).count().reset_index()
    data = data.rename(columns={'EventDate': 'Emails'})

    # sum emails sent both ways
    added_pairs = []
    workers_a = []
    workers_b = []
    emails_nums = []

    for row in data.itertuples():

        sender = row[1]
        recipient = row[2]
        emails = row[3]

        if not (sender, recipient) in added_pairs:

            sub_df = data[
                (data['Sender'] == recipient) & (data['Recipient'] == sender)
            ]
            if len(sub_df) > 0:
                emails = emails + sub_df.iloc[0]['Emails']

            added_pairs.append((sender, recipient))
            workers_a.append(sender)
            workers_b.append(recipient)
            emails_nums.append(emails)

    # build and filter dataframe by min emails number
    df = pd.DataFrame({'a': workers_a, 'b': workers_b, 'emails': emails_nums})
    df = df[df['emails'] > min_emails]

    # convert to graph
    G = nx.from_pandas_edgelist(df, 'a', 'b').to_undirected()

    return G


def get_network_metrics(G: nx.Graph) -> Dict:

    return {
        'density': round(nx.density(G), 2),
        'degree_std': round(np.array(
            list(dict(nx.degree(G)).values())
        ).std(), 2),
        'degree_dist': np.array(list(dict(nx.degree(G)).values())),
        'assortavity': round(nx.degree_assortativity_coefficient(G), 2),
        'avg_clust': round(nx.average_clustering(G), 2),
        'modularity': round(nx.community.quality.modularity(
            G, nx.community.greedy_modularity_communities(G)
        ), 2),
        'avg_shortest_path': round(nx.average_shortest_path_length(G), 2)
    }


def get_nodes_static_metrics(
    G: nx.Graph
) -> pd.DataFrame:
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G)

    # results = {}
    # for node in G.nodes:
    #     results[node] = {
    #         'degree_centrality': degree_centrality[node],
    #         'betweenness_centrality': betweenness_centrality[node],
    #         'closeness_centrality': closeness_centrality[node],
    #         'pagerank': pagerank[node]
    #     }
    return pd.DataFrame({
        'node': list(G.nodes()),
        'degree': list(dict(nx.degree(G)).values()),
        'degree_centrality': list(degree_centrality.values()),
        'betweenness_centrality': list(betweenness_centrality.values()),
        'closeness_centrality': list(closeness_centrality.values()),
        'pagerank': list(pagerank.values())
    })


def get_nodes_dynamic_metrics(
    G: nx.Graph,
    classes: Dict
) -> pd.DataFrame:

    nodes = []
    first_neigh_0_nums = []
    first_neigh_1_nums = []
    first_neigh_2_nums = []
    second_neigh_0_nums = []
    second_neigh_1_nums = []
    second_neigh_2_nums = []
    labels = []

    for node in G.nodes:
        nodes.append(node)
        labels.append(classes[node])

        metrics = get_single_node_dynamic_metrics(G, classes, node)

        first_neigh_0_nums.append(metrics['first_neigh_0'])
        first_neigh_1_nums.append(metrics['first_neigh_1'])
        first_neigh_2_nums.append(metrics['first_neigh_2'])
        second_neigh_0_nums.append(metrics['second_neigh_0'])
        second_neigh_1_nums.append(metrics['second_neigh_1'])
        second_neigh_2_nums.append(metrics['second_neigh_2'])

    return pd.DataFrame({
        'node': nodes,
        'first_neigh_0': first_neigh_0_nums,
        'first_neigh_1': first_neigh_1_nums,
        'first_neigh_2': first_neigh_2_nums,
        'second_neigh_0': second_neigh_0_nums,
        'second_neigh_1': second_neigh_1_nums,
        'second_neigh_2': second_neigh_2_nums,
        'label': labels
    })


def get_single_node_dynamic_metrics(
    G: nx.Graph,
    classes: Dict,
    node: int
) -> pd.DataFrame:

    first_neigh_0 = 0
    first_neigh_1 = 0
    first_neigh_2 = 0
    second_neigh_0 = 0
    second_neigh_1 = 0
    second_neigh_2 = 0

    for n1 in G.neighbors(node):
        if classes[n1] == 0:
            first_neigh_0 += 1
        elif classes[n1] == 1:
            first_neigh_1 += 1
        elif classes[n1] == 2:
            first_neigh_2 += 1

        for n2 in G.neighbors(n1):
            if classes[n2] == 0:
                second_neigh_0 += 1
            elif classes[n2] == 1:
                second_neigh_1 += 1
            elif classes[n2] == 2:
                second_neigh_2 += 1

    return {
        'node': node,
        'first_neigh_0': first_neigh_0,
        'first_neigh_1': first_neigh_1,
        'first_neigh_2': first_neigh_2,
        'second_neigh_0': second_neigh_0,
        'second_neigh_1': second_neigh_1,
        'second_neigh_2': second_neigh_2
    }


def remove_nodes_by_degree(
    G: nx.graph,
    min_degree: int
) -> nx.Graph:

    G = nx.Graph(deepcopy(G))

    to_remove = [
        node for node, degree in dict(G.degree()).items()
        if degree < min_degree
    ]

    for node in to_remove:
        G.remove_node(node)

    return G
