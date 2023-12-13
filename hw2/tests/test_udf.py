import pytest
from pyspark.sql import SparkSession
from chispa import *

from video_analytics.functions import scoring_videos, mean, udfPandasSplit
from pyspark.sql.types import *
import pyspark.sql.functions as f



@pytest.fixture(scope='session')
def spark():
    return (
        SparkSession
        .builder
        .master("local")
        .appName("chispa")
        .getOrCreate()
    )
def test_split(spark):
    input_data = [
        ("logan paul vlog|logan paul|logan|paul|olympics|logan paul youtube|vlog|daily",),
        ("reality|emoji|animoji|Face ID",),
        ("|||",),
        ("|",),
        ("",),
        (None,)
    ]

    expected_data = [
        (["logan paul vlog", "logan paul", "logan", "paul", "olympics", "logan paul youtube", "vlog", "daily"],),
        (["reality", "emoji", "animoji", "Face ID"],),
        (["", "", "", ""],),
        (["", ""],),
        ([""],),
        (None,)
    ]

    schema = ['tags']

    df = spark \
        .createDataFrame(data=input_data, schema=schema) \
        .withColumn("tags", udfPandasSplit("tags"))

    expected_df = spark \
        .createDataFrame(data=expected_data, schema=schema)

    assert_df_equality(df, expected_df)


def test_median(spark):
    input_data = [
        ("Shows", 3.0,),
        ("Shows", 0.0,),
        ("Education", 1.0,),
        ("Comedy", None)
    ]

    expected_data = [
        ("Shows", 2.0,),
        ("Education", 1.0,),
        ("Gaming", 100.0),
        ("Comedy", None)
    ]

    schema = StructType([ \
        StructField("category", StringType(), True), \
        StructField("score", DoubleType(), True)
    ])

    df = spark \
        .createDataFrame(data=input_data, schema=schema) \
        .groupBy("category").agg(mean("score").alias("score"))

    expected_df = spark \
        .createDataFrame(data=expected_data, schema=schema)

    assert_df_equality(df, expected_df, ignore_row_order=True)


def test_sum_func(spark):
    data = [
        (1, 1, 1, 1, 1, 1),
        (0, 0, 0, 0, 0, None),
        (100, 100, 100, 100, 0, 100)
    ]

    schema = ["likes", "dislikes", "views", "comment_likes", "comment_replies", "expected_score"]

    df = spark.createDataFrame(data, schema) \
        .withColumn("score", scoring_videos("likes", "dislikes", "views", "comment_likes", "comment_replies"))

    assert_column_equality(df, "score", "expected_score")