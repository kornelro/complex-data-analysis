import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm

from src.data_processing import get_leafs, get_levels, get_root


class LCPN:

    def __init__(
        self,
        base_classifier,
        hierarchy
    ):
        self.base_classifier = base_classifier
        self.H = hierarchy
        self.root = get_root(self.H)
        self.leafs = get_leafs(self.H)
        self.levels = get_levels(self.H)
        self.clfs = {}

        tqdm.pandas()

    def fit(
        self,
        df: pd.DataFrame,
        label_col: str = 'label',
        path_col: str = 'path'
    ):
        df = df.copy()

        for level in tqdm(range(len(self.levels.keys())-1)):

            parent_nodes = self.levels[level]

            for parent_node in parent_nodes:

                clf = clone(self.base_classifier)

                if parent_node != self.root:
                    sub_df = df[df[path_col].map(set) & set([parent_node])]
                    sub_df = sub_df[sub_df[label_col] != parent_node]
                else:
                    sub_df = df

                child_nodes = list(self.H.neighbors(parent_node))
                sub_df['sub_label'] = sub_df.apply(
                    lambda row: list(
                        set(row[path_col]).intersection(set(child_nodes))
                    )[0],
                    axis=1
                )

                X = sub_df.drop(columns=[label_col, path_col, 'sub_label'])
                y = sub_df['sub_label']
                clf.fit(X, y)

                self.clfs[parent_node] = clf

    def predict(
        self,
        df: pd.DataFrame,
        label_col: str = 'label',
        path_col: str = 'path',
        predicted_label_col: str = 'predicted_label',
        predicted_path_col: str = 'predicted_path'
    ):

        X = df.drop(columns=[label_col, path_col])
        df = df.copy()
        df_copy = df.copy()

        for node, clf in tqdm(self.clfs.items()):
            df_copy[node] = clf.predict(X)

        df_copy = df_copy.progress_apply(
            lambda row: self._add_prediction(
                row,
                predicted_label_col,
                predicted_path_col
            ),
            axis=1
        )

        df[predicted_label_col] = df_copy[predicted_label_col]
        df[predicted_path_col] = df_copy[predicted_path_col]

        return df

    def _add_prediction(
        self,
        row,
        predicted_label_col='predicted_label',
        predicted_path_col='predicted_path'
    ):
        path = []
        next_pred = row[self.root]

        while next_pred not in self.leafs:
            next_pred = row[next_pred]
            path.append(next_pred)

        row[predicted_label_col] = next_pred
        row[predicted_path_col] = path

        return row

    def predict_deprecated(
        self,
        df: pd.DataFrame(),
        label_col: str = 'label',
        path_col: str = 'path',
        predicted_label_col: str = 'predicted_label',
        predicted_path_col: str = 'predicted_path'
    ):
        df = df.copy()
        df[predicted_path_col] = df.drop(
            columns=[label_col, path_col]
        ).progress_apply(
            lambda row: self._single_predict(row),
            axis=1
        )

        return df

    def _single_predict(self, row):
        predicted_path = []
        predicted_node = self.root
        for level in range(len(self.levels.keys())-1):
            clf = self.clfs[predicted_node]
            predicted_node = clf.predict(np.array(row).reshape(1, -1))[0]
            predicted_path.append(predicted_node)


class LCN(LCPN):
    def __init__(
        self,
        base_classifier,
        hierarchy
    ):

        base_classifier = OneVsRestClassifier(base_classifier)

        super().__init__(base_classifier, hierarchy)


class LCL(LCPN):

    def __init__(
        self,
        base_classifier,
        hierarchy
    ):
        super().__init__(base_classifier, hierarchy)

    def fit(
        self,
        df: pd.DataFrame,
        label_col: str = 'label',
        path_col: str = 'path'
    ):
        df = df.copy()

        for level in tqdm(range(len(self.levels.keys())-1)):

            clf = clone(self.base_classifier)

            sub_df = df

            child_nodes = self.levels[level + 1]
            sub_df['sub_label'] = sub_df.apply(
                lambda row: list(
                    set(row[path_col]).intersection(set(child_nodes))
                )[0],
                axis=1
            )

            X = sub_df.drop(columns=[label_col, path_col, 'sub_label'])
            y = sub_df['sub_label']
            clf.fit(X, y)

            self.clfs[level] = clf

    def predict(
        self,
        df: pd.DataFrame,
        label_col: str = 'label',
        path_col: str = 'path',
        predicted_label_col: str = 'predicted_label',
        predicted_path_col: str = 'predicted_path'
    ):

        df = df.copy()
        df = df.drop(
            columns=[label_col, path_col]
        ).progress_apply(
            lambda row: self._single_predict(row),
            axis=1
        )

        return df

    def _single_predict(
        self,
        row,
        predicted_label_col='predicted_label',
        predicted_path_col='predicted_path'
    ):
        prediction = self.root
        path = []

        for level, clf in self.clfs.items():

            last_prediction_child_nodes = list(self.H.neighbors(prediction))

            prediction_proba = clf.predict_proba(
                np.array(row).reshape(1, -1)
            )[0]
            proba_classes = clf.classes_
            prediction_proba = zip(proba_classes, prediction_proba)
            prediction_proba = sorted(
                prediction_proba,
                key=lambda tup: tup[1],
                reverse=True
            )

            prediction = None
            i = 0
            while (not prediction) and (i < len(prediction_proba)):
                c, p = prediction_proba[i]
                if c in last_prediction_child_nodes:
                    prediction = c
                i += 1

            path.append(prediction)

        row[predicted_path_col] = path
        row[predicted_label_col] = path[-1]

        return row
