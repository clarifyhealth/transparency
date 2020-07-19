import uuid
from pathlib import Path

import pytest
from pandas._testing import assert_frame_equal
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import SparkSession, DataFrame

from testutil.common import get_glm_pipeline_stages, get_feature_coefficients, get_glm_explain_stages


@pytest.mark.parametrize("family", ["tweedie"])
@pytest.mark.parametrize("variance_power", [0.0, 1.0, 1.1, 1.2, 2.0])
def test_explain_regressor(spark_session: SparkSession, family: str, variance_power: float):
    data_dir = Path(__file__).parent.parent.joinpath('data')

    link_power = 1 - variance_power

    boston_csv: Path = data_dir / 'regression' / 'dataset_boston.csv'

    boston_df: DataFrame = spark_session.read.option("header", True).option("inferSchema", True).csv(
        boston_csv.as_uri())

    label_column = 'medv'
    features_column = f"features_{label_column}"
    prediction_column = f"prediction_{label_column}"

    contrib_column = f"prediction_{label_column}_contrib"
    contrib_column_sum = f"{contrib_column}_sum"
    contrib_column_intercept = f"{contrib_column}_intercept"

    features_coefficient_view = f"features_coefficient_{label_column}_view"
    predictions_view = f"predictions_{label_column}_view"

    categorical_columns = []
    continuous_columns = [x for x in boston_df.columns if x not in ['id', label_column]]

    stages = get_glm_pipeline_stages(categorical_columns, continuous_columns, label_column, family=family,
                                     variance_power=variance_power, link_power=link_power)

    pipeline = Pipeline(stages=stages)

    pipeline_model: PipelineModel = pipeline.fit(boston_df)
    glm_model = pipeline_model.stages[-1]
    string_id = uuid.uuid4()
    glm_model_path: str = f"/tmp/{string_id}"
    glm_model.write().save(glm_model_path)

    # param_dict = extract_params(glm_model)

    prediction_df = pipeline_model.transform(boston_df)
    prediction_df.createOrReplaceTempView(predictions_view)

    prediction_df.show(truncate=False)

    features_coefficients_df = get_feature_coefficients(spark_session, glm_model, prediction_df, features_column)
    features_coefficients_df.createOrReplaceTempView(features_coefficient_view)

    features_coefficients_df.show(truncate=False)

    explain_stages = get_glm_explain_stages(predictions_view, features_coefficient_view, label_column,
                                            family=family, variance_power=variance_power, link_power=link_power)

    explain_pipeline = Pipeline(stages=explain_stages)

    explain_df = explain_pipeline.fit(prediction_df).transform(prediction_df)

    explain_df_cache = explain_df.limit(100).cache()

    explain_df_cache.show(truncate=False)

    predictions = explain_df_cache.selectExpr(f"bround({prediction_column},5) as test_col").orderBy("id").toPandas()
    contributions = explain_df_cache.selectExpr(
        f"bround({contrib_column_sum} + {contrib_column_intercept},5) as test_col").orderBy("id").toPandas()

    assert_frame_equal(predictions, contributions)
