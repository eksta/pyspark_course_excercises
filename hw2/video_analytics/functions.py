import pandas as pd
from numpy import median
from pyspark.sql.functions import *
from pyspark.sql.types import *
def scoring_videos(views: pd.Series,
                   likes: pd.Series,
                   dislikes: pd.Series,
                   comment_likes: pd.Series,
                   comment_replies: pd.Series) -> pd.Series:
    return views * 0.03 \
        + likes * 0.07 \
        + dislikes * 0.05 \
        + comment_likes * 0.01 \
        + comment_replies * 0.02


@pandas_udf(DoubleType(),PandasUDFType.GROUPED_AGG)
def mean(score: pd.Series) -> float:
    return median(score)
@pandas_udf(ArrayType(StringType()), functionType=PandasUDFType.SCALAR)
def udfPandasSplit(tags: pd.Series) -> pd.Series:
    return tags.str.split('|')


