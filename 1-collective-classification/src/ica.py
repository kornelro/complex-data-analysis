from copy import deepcopy
from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .network import (get_nodes_dynamic_metrics, get_nodes_static_metrics,
                      get_single_node_dynamic_metrics)


def uncover_nodes(
    G: nx.Graph,
    real_classes: Dict,
    metric: str,
    p: float,
    sort_ascending: bool = False
) -> Dict:
    static_metrics = get_nodes_static_metrics(G)
    static_metrics = static_metrics.sort_values(
        metric,
        ascending=sort_ascending
    )
    static_metrics = static_metrics.reset_index()
    static_metrics = static_metrics.iloc[0:int(p * len(static_metrics))]
    uncover_nodes = list(static_metrics['node'])

    uncover_classes = deepcopy(real_classes)
    for key, value in uncover_classes.items():
        if key not in uncover_nodes:
            uncover_classes[key] = -1

    return uncover_classes


def ICA(
    G: nx.Graph,
    real_classes: Dict,
    metric: str,
    p: float,
    classifier,
    max_steps: int,
    verbose: bool = True
):
    # uncover nodes
    init_labels = uncover_nodes(G, real_classes, 'betweenness_centrality', 0.1)

    # get all nodes metrics
    # for dynamic metrics count only uncover nodes
    data = pd.concat([
        get_nodes_static_metrics(G),
        get_nodes_dynamic_metrics(G, init_labels).drop(columns='node')
    ], axis=1)

    # train classifier using uncovered nodes
    data_uncovered = data[data['label'] != -1]
    X = data_uncovered.drop(columns=['node', 'label'])
    y = data_uncovered['label']
    classifier.fit(X, y)

    # collective classification loop
    step = 1
    change = True

    data['init_label'] = data['label']
    data = data.rename(columns={'label': 'current_label'})
    data['predicted_proba'] = 0.
    data['predicted_label'] = data['current_label']
    data['real_label'] = data.apply(lambda x: real_classes[x['node']], axis=1)
    labels_register = data[[
        'node', 'init_label', 'current_label',
        'predicted_proba', 'predicted_label', 'real_label'
    ]]

    while (step <= max_steps) and change:

        if verbose:
            print('Iteration '+str(step))

        # sort nodes by predicted proba
        labels_register = labels_register.sort_values(
            'predicted_proba', ascending=False
        )

        # update labels register
        labels_register['current_label'] = labels_register['predicted_label']

        # iterate over sorted nodes
        for row_id in list(labels_register.index):
            row = deepcopy(labels_register.iloc[row_id])
            node = row['node']

            # get current classes assingment dict
            current_classes = dict(
                zip(labels_register['node'], labels_register['current_label'])
            )

            # read static node's metrics and current dynamic metrics
            static_metrics = data[data['node'] == node].iloc[0]
            dynamic_metrics = get_single_node_dynamic_metrics(
                G, current_classes, node
            )

            # predict label
            x = [
                static_metrics['degree'],
                static_metrics['degree_centrality'],
                static_metrics['betweenness_centrality'],
                static_metrics['closeness_centrality'],
                static_metrics['pagerank'],
                dynamic_metrics['first_neigh_0'],
                dynamic_metrics['first_neigh_1'],
                dynamic_metrics['first_neigh_2'],
                dynamic_metrics['second_neigh_0'],
                dynamic_metrics['second_neigh_1'],
                dynamic_metrics['second_neigh_2']
            ]
            x = np.array(x).reshape(1, -1)
            y = classifier.predict_proba(x)[0]

            # update values in labels register
            row['predicted_proba'] = np.max(y)
            row['predicted_label'] = np.argmax(y)
            labels_register.iloc[row_id] = row

        # print(labels_register.head())

        # udate stop conditions
        step += 1
        if len(labels_register[
            labels_register['current_label'] != labels_register['predicted_label']
        ]) > 0:
            change = True
        else:
            change = False

        if verbose:
            print('F1 Score: ', + round(f1_score(
                labels_register['real_label'],
                labels_register['predicted_label'],
                average='macro'
            ), 3))

    return labels_register
