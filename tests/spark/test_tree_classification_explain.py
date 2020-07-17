from pathlib import Path
import uuid

import pytest
from pandas._testing import assert_frame_equal
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import SparkSession, DataFrame
from testutil.common import get_ensemble_pipeline_stages, get_feature_importance, get_ensemble_explain_stages


@pytest.mark.parametrize("ensemble_type", ["dct", "gbt", "rf"])
def test_explain_classification(spark_session: SparkSession, ensemble_type: str):
    data_dir = Path(__file__).parent.parent.joinpath('data')

    prima_indian_diabetes_csv: Path = data_dir / 'classification' / 'dataset_prima_indian_diabetes.csv'

    prima_indian_diabetes_df: DataFrame = spark_session.read.option("header", True).option("inferSchema", True).csv(
        prima_indian_diabetes_csv.as_uri())

    label_column = 'outcome'
    features_column = f"features_{label_column}"
    prediction_column = f"prediction_{label_column}"

    contrib_column = f"prediction_{label_column}_contrib"
    contrib_column_sum = f"{contrib_column}_sum"
    contrib_column_intercept = f"{contrib_column}_intercept"

    features_importance_view = f"features_importance_{label_column}_view"
    predictions_view = f"predictions_{label_column}_view"

    categorical_columns = []
    continuous_columns = [x for x in prima_indian_diabetes_df.columns if x not in ['id', label_column]]

    stages = get_ensemble_pipeline_stages(categorical_columns, continuous_columns, label_column, ensemble_type,
                                          classification=True)

    pipeline = Pipeline(stages=stages)

    pipeline_model: PipelineModel = pipeline.fit(prima_indian_diabetes_df)
    rf_model = pipeline_model.stages[-1]
    string_id = uuid.uuid4()
    rf_model_path: str = f"/tmp/{string_id}"
    rf_model.write().save(rf_model_path)

    prediction_df = pipeline_model.transform(prima_indian_diabetes_df)
    prediction_df.createOrReplaceTempView(predictions_view)

    prediction_df.show(truncate=False)

    features_importance_df = get_feature_importance(spark_session, rf_model, prediction_df, features_column)
    features_importance_df.createOrReplaceTempView(features_importance_view)

    features_importance_df.show(truncate=False)

    explain_stages = get_ensemble_explain_stages(predictions_view, features_importance_view, label_column, rf_model_path,
                                                 ensemble_type, classification=True)

    explain_pipeline = Pipeline(stages=explain_stages)

    explain_df = explain_pipeline.fit(prediction_df).transform(prediction_df)

    explain_df_cache = explain_df.limit(100).cache()

    explain_df_cache.show(truncate=False)

    predictions = explain_df_cache.selectExpr(f"bround({prediction_column},5) as test_col").orderBy("id").toPandas()
    contributions = explain_df_cache.selectExpr(
        f"bround({contrib_column_sum}+{contrib_column_intercept},5) as test_col").orderBy("id").toPandas()

    assert_frame_equal(predictions, contributions)
