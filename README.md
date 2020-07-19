# transparency
Model explanation generator:
- Python (Scikit-Learn)
- Pyspark (Scala and Pyspark)

# Installation:
- Add jar to spark classpath : https://github.com/clarifyhealth/spark_model_explainer
- `pip install transparency`


# Usage : Ensemble Tree models
- Scikit-Learn Transformer
XXX XXX 
- Pyspark Transformer
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

# Usage : GLM models
- Pyspark Transformer
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
