import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as r2
from transparency.python.explainer.ensemble_tree import EnsembleTreeExplainer
from transparency.python.explainer.ensemble_tree import EnsembleTreeExplainerTransformer
from xgboost import XGBRegressor

# %%

# loading the diabetes dataset

columns = 'age sex bmi map tc ldl hdl tch ltg glu'.split()
diabetes = load_diabetes()
X = np.array(pd.DataFrame(diabetes.data, columns=columns))
y = diabetes.target

# train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# model training
rf_model = RandomForestRegressor().fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# regression evaluation: r2 score
r2_eval = r2(y_test, y_pred)
print(r2_eval)

# %%
# prediction explanation generation
expl = EnsembleTreeExplainer(rf_model)
contributions, contrib_intercept = expl.predict(X_test)
assert(((np.sum(contributions, axis=1) + contrib_intercept) - y_pred < .01).all())

# %%

average_contribs = zip(columns, np.mean(contributions, axis=0))
print(list(average_contribs))

# %% XGBOOST

# loading the diabetes dataset
columns = 'age sex bmi map tc ldl hdl tch ltg glu'.split()
diabetes = load_diabetes()
X = np.array(pd.DataFrame(diabetes.data, columns=columns))
y = diabetes.target

# train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# model training
xgb_model = XGBRegressor().fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# regression evaluation: r2 score
r2_eval = r2(y_test, y_pred)
print(r2_eval)

# %%
# prediction explanation generation
expl = EnsembleTreeExplainer(xgb_model)
contributions, contrib_intercept = expl.predict(X_test)
assert(((np.sum(contributions, axis=1) + contrib_intercept) - y_pred < .01).all())

# %%

average_contribs = zip(columns, np.mean(contributions, axis=0))
print(list(average_contribs))

# %% Transformer Test

columns = 'age sex bmi map tc ldl hdl tch ltg glu'.split()
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=columns)
y = diabetes.target

# train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# model training
rf_model = RandomForestRegressor().fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# regression evaluation: r2 score
r2_eval = r2(y_test, y_pred)
print(r2_eval)

X_test2 = X_test.copy()

expl = EnsembleTreeExplainerTransformer(rf_model)
expl.fit()
X_test2 = expl.transform(X_test2)

assert('feature_contributions' in X_test2.columns)
assert('intercept_contribution' in X_test2.columns)
assert((np.abs(np.array(X_test2['feature_contributions'].apply(lambda x: sum(x[0])) + X_test2['intercept_contribution'])\
               - np.array(y_pred)) < .01).all())
