# glmExplainer Example

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, f1_score
from transparency.python.explainer.glm import GLMExplainerTransformer

#%%

def test_glm_iris_explanation():
    # loading the iris dataset

    columns = 'Sepal_Length Sepal_Width Petal_Length Petal_Width'.split()
    iris = load_iris()
    y = iris.target
    X = pd.DataFrame(iris.data, columns=columns).iloc[y < 2, :]
    y = y[y < 2]

    # train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # model training
    clf = linear_model.LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    # regression evaluation: r2 score
    f1_eval = f1_score(y_test, y_pred)
    print('f1 score = ', f1_eval)

    clf.predict(np.zeros((1, X_train.shape[1])))[0]

    # prediction explanation generation
    expl = GLMExplainerTransformer(clf)
    df = expl.transform(X_test)

    assert ((np.abs(df['feature_contributions'].apply(lambda x: sum(x[0])) + \
                    df['intercept_contribution'] - df['prediction']) < .01).all())


def test_glm_iris_proba_explanation():
    # loading the iris dataset

    columns = 'Sepal_Length Sepal_Width Petal_Length Petal_Width'.split()
    iris = load_iris()
    y = iris.target
    X = pd.DataFrame(iris.data, columns=columns).iloc[y < 2, :]
    y = y[y < 2]

    # train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # model training
    clf = linear_model.LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    # regression evaluation: r2 score
    f1_eval = f1_score(y_test, y_pred)
    print('f1 score = ', f1_eval)

    clf.predict(np.zeros((1, X_train.shape[1])))[0]

    # prediction explanation generation
    expl = GLMExplainerTransformer(clf, output_proba=True)
    df = expl.transform(X_test)

    assert ((np.abs(df['feature_contributions'].apply(lambda x: sum(x[0])) + \
                    df['intercept_contribution'] - df['prediction']) < .01).all())