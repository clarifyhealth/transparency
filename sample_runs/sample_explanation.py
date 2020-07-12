import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as r2
from transparency.python.explainer.tree import EnsembleTreeExplainTransformer

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
expl = EnsembleTreeExplainTransformer(rf_model)
contributions, contrib_intercept = expl.predict(X_test)
assert(((np.sum(contributions, axis=1) + contrib_intercept) - y_pred < .01).all())

# %%

average_contribs = zip(columns, np.mean(contributions, axis=0))
print(list(average_contribs))

