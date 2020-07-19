import numpy as np
import pandas as pd

class EnsembleTreeExplainer:
    """
    Prediction explainer for ensemble trees in Scikit-Learn
    """

    def __init__(self, estimator):
        self.estimator = estimator
        self.feature_importance_ranks = np.argsort(estimator.feature_importances_)[::-1]
        self.feature_count = len(self.feature_importance_ranks)

    def _path_generator(self, X):
        nf = self.feature_count
        tree_paths = []
        for row in X:
            nzid = np.nonzero(row)[0]
            tree_paths.append([0] * nf)
            for id in nzid:
                tree_paths.append(tree_paths[-1][:id] + [row[id]] + tree_paths[-1][id + 1:])
        tree_paths = np.array(tree_paths)
        return tree_paths

    def predict(self, X):
        nf = self.feature_count
        tree_paths = self._path_generator(np.array(X))
        path_outcomes = self.estimator.predict(tree_paths)
        incremental_contributions = path_outcomes[1:] - path_outcomes[:-1]
        contributions = []
        current_contribs = [0] * nf
        for path_no, path in enumerate(tree_paths[1:]):
            if (path == 0).all():
                contributions.append(current_contribs)
                current_contribs = [0] * nf
            else:
                current_contribs[np.argmax(np.nonzero(path)[0])] = incremental_contributions[path_no]
        contributions.append(current_contribs)
        contributions = np.array(contributions)[:, np.argsort(self.feature_importance_ranks)]
        contrib_intercept = path_outcomes[0]
        return contributions, contrib_intercept


class EnsembleTreeExplainerTransformer(EnsembleTreeExplainer):
    """
    Prediction explainer transformer for ensemble trees in Scikit-Learn
    """

    def __init__(self, estimator):
        super().__init__(estimator)

    def fit(self, *args, **kwargs):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        contributions, contrib_intercept = self.predict(df)
        df['feature_contributions'] = [[f] for f in contributions]
        df['intercept_contribution'] = contrib_intercept
        return df
