import pandas as pd
from sklearn.metrics import f1_score
from sklearn_hierarchical_classification.metrics import (h_fbeta_score,
                                                         multi_labeled)


def get_comparision(
    predictions,
    y_test,
    lib_graph
):

    h_f1_column = 'h_f1'
    f1_micro_column = 'f1_micro'
    comparision = pd.DataFrame(
        columns=['model', h_f1_column, f1_micro_column]
    )

    for (model, y_pred) in predictions:

        with multi_labeled(
            y_test, y_pred, lib_graph
        ) as (y_test_, y_pred_, graph_):

            h_fbeta = h_fbeta_score(
                y_test_,
                y_pred_,
                graph_,
            )

        f1_micro = f1_score(y_test, y_pred, average='micro')

        comparision = comparision.append(
            {
                'model': model,
                h_f1_column: h_fbeta,
                f1_micro_column: f1_micro
            },
            ignore_index=True
        )

    return comparision
