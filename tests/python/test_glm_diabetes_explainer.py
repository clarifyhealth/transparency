# glmExplainer Example

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, f1_score
from transparency.python.explainer.glm import GLMExplainerTransformer

#%%

def test_glm_diabetes_explanation():

    # loading the diabetes dataset

    columns = 'age sex bmi map tc ldl hdl tch ltg glu'.split()
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=columns)
    y = diabetes.target

    # train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # model training
    clf = linear_model.Ridge()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # regression evaluation: r2 score
    r2_eval = r2_score(y_test, y_pred)
    print('r2 score = ', r2_eval)

    clf.predict(np.zeros((1, X_train.shape[1])))[0]

    # prediction explanation generation
    expl = GLMExplainerTransformer(clf)
    df = expl.transform(X_test)

    assert((np.abs(df['feature_contributions'].apply(lambda x: sum(x[0])) + \
                   df['intercept_contribution'] - df['prediction']) < .01).all())
    return
