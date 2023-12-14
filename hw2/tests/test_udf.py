import pytest
from pyspark.sql import SparkSession
from chispa import *
from pyspark.sql.types import *
from video_analytics.functions import udfScoringVideos, udfMean, udfPandasSplit
import pyspark.sql.functions as f


@pytest.fixture(scope='session')
def spark():
    return (
        SparkSession
        .builder
        .master("local")
        .config("spark.driver.host", "127.0.0.1")
        .appName("chispa")
        .getOrCreate()
    )

def test_scoring_videos(spark):
    data = [
        (1.0, 1.0, 1.0, 1.0, 1.0, 0.18000000000000002),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (100.0, 100.0, 100.0, 100.0, 0.0, 16.0)
    ]
    schema = ["likes", "dislikes", "views", "comment_likes", "comment_replies", "expected_score"]
    df = spark.createDataFrame(data, schema) \
        .withColumn("score", udfScoringVideos("likes", "dislikes", "views", "comment_likes", "comment_replies"))
    assert_column_equality(df, "score", "expected_score")

def test_split(spark):
    input_data = [
        ("logan paul vlog|logan paul|logan|paul|olympics|logan paul youtube|vlog|daily",),
        ("reality|emoji|animoji|Face ID",),
        ("|",),
        ("",),
        (None,)
    ]

    expected_data = [
        (["logan paul vlog", "logan paul", "logan", "paul", "olympics", "logan paul youtube", "vlog", "daily"],),
        (["reality", "emoji", "animoji", "Face ID"],),
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
        ("Shows", 1.5,),
        ("Education", 1.0,),
        ("Comedy", None)
    ]

    schema = StructType([ \
        StructField("category", StringType(), True), \
        StructField("score", DoubleType(), True)
    ])

    df = spark \
        .createDataFrame(data=input_data, schema=schema) \
        .groupBy("category").agg(udfMean("score").alias("score"))

    expected_df = spark \
        .createDataFrame(data=expected_data, schema=schema)

    assert_df_equality(df, expected_df, ignore_row_order=True)



