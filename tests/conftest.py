import logging
from pathlib import Path

import findspark  # this needs to be the first import
import pytest
from pyspark.sql import SparkSession

findspark.init()


def quiet_py4j():
    """ turn down spark logging for the test context """
    logger = logging.getLogger('py4j')
    logger.setLevel(logging.WARN)


@pytest.fixture(scope="session")
def spark_session(request):
    lib_dir = Path(__file__).parent.joinpath('jars')

    session = SparkSession.builder.appName("pytest-pyspark") \
        .master("local[2]") \
        .config("spark.sql.execution.arrow.enabled", "true") \
        .config("spark.jars.packages", "ml.dmlc:xgboost4j_2.11:1.0.0,ml.dmlc:xgboost4j-spark_2.11:1.0.0") \
        .config("spark.jars",
                lib_dir.joinpath('spark_model_explainer-assembly-0.0.1.jar').as_uri()) \
        .enableHiveSupport().getOrCreate()

    request.addfinalizer(lambda: session.stop())

    quiet_py4j()
    return session
