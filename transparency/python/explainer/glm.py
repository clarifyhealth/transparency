import numpy as np
import pandas as pd


class GLMExplainerTransformer(object):
    """
    Prediction explainer transformer for generalized linear models in Scikit-Learn
    output_proba: if set true, uses probabilities as output (e.g., for logistic regression)
    """

    def __init__(self, estimator, output_proba=False):
        self.estimator = estimator
        self.output_proba = output_proba

    def fit(self, *args, **kwargs):
        return self

    def transform(self, df: pd.DataFrame or np.array) -> pd.DataFrame:

        df = pd.DataFrame(df)
        linear_contribs = pd.DataFrame(df * self.estimator.coef_)
        contribs_pos = linear_contribs[linear_contribs > 0].fillna(0)
        contribs_neg = linear_contribs[linear_contribs < 0].fillna(0)
        sigma_pos = np.array(contribs_pos.sum(axis=1)).reshape(-1, 1)
        sigma_neg = np.array(contribs_neg.sum(axis=1)).reshape(-1, 1)
        sigma_pos[sigma_pos == 0] = 1
        sigma_neg[sigma_neg == 0] = 1

        if not self.output_proba:
            pred = self.estimator.predict(df)
            pred_pos = self.estimator.predict(df[contribs_pos != 0].fillna(0))
            pred_neg = self.estimator.predict(df[contribs_neg != 0].fillna(0))
            intercept_contrib = self.estimator.predict(np.zeros((1, df.shape[1])))[0]
        else:
            pred = self.estimator.predict_proba(df)[:, 1]
            pred_pos = self.estimator.predict_proba(df[contribs_pos != 0].fillna(0))[:, 1]
            pred_neg = self.estimator.predict_proba(df[contribs_neg != 0].fillna(0))[:, 1]
            intercept_contrib = self.estimator.predict_proba(np.zeros((1, df.shape[1])))[0, 1]

        deficit = pred + intercept_contrib - (pred_pos + pred_neg)
        sum_contribs_pos = (pred_pos - intercept_contrib + deficit / 2).reshape(-1, 1)
        sum_contribs_neg = (pred_neg - intercept_contrib + deficit / 2).reshape(-1, 1)

        contribs_pos = contribs_pos * sum_contribs_pos / sigma_pos
        contribs_neg = contribs_neg * sum_contribs_neg / sigma_neg
        contribs = (contribs_pos + contribs_neg)

        prediction_col_name = 'prediction'
        while prediction_col_name in df.columns:
            prediction_col_name = prediction_col_name + '_'
        df[prediction_col_name] = pred
        df['feature_contributions'] = [[f] for f in np.array(contribs)]
        df['intercept_contribution'] = intercept_contrib

        return df
