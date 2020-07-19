# transparency
Model explanation generator:
- Python (Scikit-Learn)
- Pyspark (Scala and Pyspark)

# Installation:
- Add jar to spark classpath : https://github.com/clarifyhealth/spark_model_explainer
- `pip install transparency`


# Usage : Ensemble Tree models
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
- Path to to load model `modelPath`

- Supported `ensembleType`
    * `dct`
    * `gbt`
    * `rf`
    * `xgboost4j`

- The feature importance extracted from Apache Spark Model Meta Data.`featureImportanceView`
  Reference this python script : `testutil.common.get_feature_importance`
    * `Feature_Index`
    * `Feature`
    * `Original_Feature`
    * `Importance`

- The transformer append 3 main column to the prediction view 
    * contrib_column ==> prediction_`label_column`_contrib : array of contributions
    * contrib_column_sum ==>  `f"{contrib_column}_sum"` (intercept `already` added)
    * contrib_column_intercept ==> `f"{contrib_column}_intercept"`

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
    * `logLink`
    * `powerHalfLink`
    * `identityLink`
    * `logitLink`
    * `inverseLink`
    * `otherPowerLink`

- The feature coefficient extracted from Apache Spark Model Meta Data.`coefficientView`
  Reference this python script : `testutil.common.get_feature_coefficients`
    * `Feature_Index`
    * `Feature`
    * `Original_Feature`
    * `Coefficient`

- The transformer append 3 main column to the prediction view 
    * contrib_column ==> prediction_`label_column`_contrib : array of contributions
    * contrib_column_sum ==>  `f"{contrib_column}_sum"` (intercept `not` added)
    * contrib_column_intercept ==> `f"{contrib_column}_intercept"`