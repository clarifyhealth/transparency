# transparency
Model explanation generator for:
- Python (Scikit-Learn)
- Pyspark (Scala and Pyspark)

# Installation:
- `pip install transparency`
Or:
- Add this jar to spark classpath : https://github.com/alvinhenrick/spark_model_explainer/releases/download/v.0.0.1/spark_model_explainer-assembly-0.0.1.jar
- Maven repository release soon will be supported : https://github.com/clarifyhealth/spark_model_explainer


# Usage : Ensemble Tree models
## - Scikit-Learn Ensemble Tree Explainer
 ```python
 from transparency.python.explainer.ensemble_tree import EnsembleTreeExplainer
 expl = EnsembleTreeExplainer(estimator)
 contributions, contrib_intercept = expl.predict(X_test)
 ```
- estimator: the ensemble tree estimator that has been trained (e.g., random forest, gbm, or xgb)
- X_test: a numpy array or a pandas dataframe with features as columns and samples as rows
- contributions: array of feature contributions generated for each row of X_test
- contrib_intercept: the contribution of intercept (the same for all rows)
The sum of contributions + contrib_intercept for each row equals the prediction for that row.
## - Scikit-Learn Ensemble Tree Explainer Transformer
 ```python
from transparency.python.explainer.ensemble_tree import EnsembleTreeExplainerTransformer
expl = EnsembleTreeExplainerTransformer(estimator)
X_test_df = expl.transform(X_test_df)
 ```
- estimator: the ensemble tree estimator that has been trained (e.g., random forest, gbm, or xgb)
- X_test: a Pandas dataframe with features as columns and samples as rows
The resulting X_test_df will have 2 added columns: 'feature_contributions' and 'intercept_contribution':
- 'feature_contributions': column of nested arrays with feature contributions (1 array per row)
- 'intercept_contribution': column of the same scaler value representing the contribution of the intercept
sum(contributions) + contrib_intercept for each row equals the prediction for that row.
## - Pyspark Ensemble Tree Explainer Transformer
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

## - Pyspark Generalized Linear Model (GLM) Explainer Transformer
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

## Pyspark Example Notebooks
- [PySpark GLM Explain Example](examples/notebooks/spark/pyspark_glm_explain.ipynb)
- [PySpark Random Forest Explain Example](examples/notebooks/spark/pyspark_random_forest_explain.ipynb)

## Authors
* Iman Haji <https://www.linkedin.com/in/imanhaji>
* Imran Qureshi <https://www.linkedin.com/in/imranq2/>
* Alvin Henrick <https://www.linkedin.com/in/alvinhenrick/>

## License
Apache License Version 2.0
