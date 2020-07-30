## Overview
One of the main barriers to more wide-spread adoption of machine learning is the Blackbox problem: users cannot see why the model predicted a certain value.

While simple linear models are relatively easy to explain, more advanced models like Generalized Linear Models, Random Forests and Boosted Trees are hard to explain especially in ways that the average person can understand. 

At Clarify Health, we had to solve this problem to get clinicians and health leaders to accept the predictions of our models. As I talked to other data scientists, I realized that this is a common problem.

Hence we are open sourcing our toolkit to explain ML models.

The Transparency project enables data scientists to explain ensemble trees (e.g., XGB, GBM, RF, and decision tree) and GLMs:

1. The explanation (feature contribution) is in the units of the prediction (e.g., dollars, days, probability etc)

2. Feature contributions add up to the predicted value so it is easy to see why a certain value was predicted

3. Can explain the model for any arbitrary sub-population at run-time

4. Works with Scikit-Learn and Apache Spark models

5. Can explain models with 100M+ rows in just a few seconds

6. Enables a what-if analysis to see what the prediction would become if you changed feature inputs



Feel free to use it and contribute to it so we can increase the adoption of machine learning.



## The "Transparency" Library
Scalable and Fast, local (single level) and global (population level) prediction explanation of:
- Ensemble trees (e.g., XGB, GBM, RF, and Decision tree)
- Generalized linear models GLM (support for various families, link powers, and variance powers, e.g., logistic regression)

implemented for models in:
- Python (Scikit-Learn)
- Pyspark (Scala and Pyspark).

The Transparency algorithm runs in a fraction of the time required by #SHAP and #LIME and produces aggregable explanations for all predictions. We have successfully used this stable library over billions of EHR records in commercial applications.

## Installation:
- `pip install transparency`

additional step for Spark users:
- Add this jar to spark classpath : https://github.com/alvinhenrick/spark_model_explainer/releases/download/v.0.0.1/spark_model_explainer-assembly-0.0.1.jar
(Maven repository release will soon be supported : https://github.com/clarifyhealth/spark_model_explainer)


## Transformer Set
### - Scikit-Learn Ensemble Tree Explainer Transformer
 ```python
from transparency.python.explainer.ensemble_tree import EnsembleTreeExplainerTransformer
expl = EnsembleTreeExplainerTransformer(estimator)
X_test_df = expl.transform(X_test_df)
 ```
- estimator: the ensemble tree estimator that has been trained (e.g., random forest, gbm, or xgb)
- X_test: a Pandas dataframe with features as columns and samples as rows
The resulting X_test_df will have 3 added columns: 'prediction', 'feature_contributions' and 'intercept_contribution':
- 'feature_contributions': column of nested arrays with feature contributions (1 array per row)
- 'intercept_contribution': column of the same scaler value representing the contribution of the intercept
sum(contributions) + contrib_intercept for each row equals the prediction for that row.
### - Scikit-Learn Generalized Linear Model (e.g., Logistic regression) Explainer Transformer
 ```python
from transparency.python.explainer.glm import GLMExplainerTransformer
expl = GLMExplainerTransformer(estimator)
X_test_df = expl.transform(X_test_df, output_proba=False)
 ```
- estimator: the glm estimator that has been trained (e.g., logistic regression)
- X_test: a Pandas dataframe with features as columns and samples as rows
The resulting X_test_df will have 3 added columns: 'prediction', 'feature_contributions' and 'intercept_contribution':
- 'feature_contributions': column of nested arrays with feature contributions (1 array per row)
- 'intercept_contribution': column of the same scaler value representing the contribution of the intercept
sum(contributions) + contrib_intercept for each row equals the prediction for that row.
- if output_proba is set to True, for the case of logistic regression, the output prediction and its corresponding explanation will be proba instead of the classification result
### - Pyspark Ensemble Tree Explainer Transformer
 ```python 
  from transparency.spark.prediction.explainer.tree import EnsembleTreeExplainTransformer
  EnsembleTreeExplainTransformer(predictionView=predictions_view, 
                                 featureImportanceView=features_importance_view,
                                 modelPath=rf_model_path, 
                                 label=label_column,
                                 dropPathColumn=True, 
                                 isClassification=classification, 
                                 ensembleType=ensemble_type)

 ```
- Path to load model `modelPath`

- Supported `ensembleType`
    1. `dct`
    2. `gbt`
    3. `rf`
    4. `xgboost4j`

- The feature importance extracted from Apache Spark Model Meta Data.`featureImportanceView`
  Reference this python script : `testutil.common.get_feature_importance`
    1. `Feature_Index`
    2. `Feature`
    3. `Original_Feature`
    4. `Importance`

- The transformer append 3 main column to the prediction view 
    1. contrib_column ==> `f"{prediction_{label_column}_contrib` : *array of contributions*
    2. contrib_column_sum ==>  `f"{contrib_column}_sum"`
    3. contrib_column_intercept ==> `f"{contrib_column}_intercept"`

### - Pyspark Generalized Linear Model (GLM) Explainer Transformer
 ```python 
   from transparency.spark.prediction.explainer.tree import GLMExplainTransformer
   GLMExplainTransformer(predictionView=predictions_view, 
                         coefficientView=coefficients_view,
                         linkFunctionType=link_function_type, 
                         label=label_column, nested=True,
                         calculateSum=True, 
                         family=family, 
                         variancePower=variance_power, 
                         linkPower=link_power)

 ```
-  Supported `linkFunctionType`
    1. `logLink`
    2. `powerHalfLink`
    3. `identityLink`
    4. `logitLink`
    5. `inverseLink`
    6. `otherPowerLink`

- The feature coefficient extracted from Apache Spark Model Meta Data.`coefficientView`
  Reference this python script : `testutil.common.get_feature_coefficients`
    1. `Feature_Index`
    2. `Feature`
    3. `Original_Feature`
    4. `Coefficient`

- The transformer append 3 main column to the prediction view 
    1. contrib_column ==> `f"{prediction_{label_column}_contrib` : *array of contributions*
    2. contrib_column_sum ==>  `f"{contrib_column}_sum"`
    3. contrib_column_intercept ==> `f"{contrib_column}_intercept"`

## Example Notebooks
- Python (Scikit-Learn) Ensemble Tree Explain Example:
https://github.com/imanbio/transparency/blob/master/examples/notebooks/python/python_ensemble_tree_explainer_samples.ipynb
- Python (Scikit-Learn) Generalized Linear Model Explain Example:
https://github.com/imanbio/transparency/blob/master/examples/notebooks/python/python_glm_explainer_samples.ipynb
- PySpark GLM Explain Example:
https://github.com/imanbio/transparency/blob/master/examples/notebooks/spark/pyspark_glm_explain.ipynb
- PySpark Random Forest Explain Example:
https://github.com/imanbio/transparency/blob/master/examples/notebooks/spark/pyspark_random_forest_explain.ipynb

## Authors
* Iman Haji <https://www.linkedin.com/in/imanhaji>
* Imran Qureshi <https://www.linkedin.com/in/imranq2/>
* Alvin Henrick <https://www.linkedin.com/in/alvinhenrick/>

## License
Apache License Version 2.0
