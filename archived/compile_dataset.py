import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType

from petastorm.codecs import ScalarCodec, NdarrayCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField

def define_schema(signal_len,spectrogram_len):
    # The schema defines how the dataset schema looks like
    schema = Unischema('NanoporeDataSchema', [
        UnischemaField('id', str, (), ScalarCodec(StringType()), False),
        UnischemaField('label', np.uint8, (), ScalarCodec(IntegerType()), False),
        UnischemaField('array_seq', np.uint16, (None,), NdarrayCodec(), False),
        UnischemaField('array_bq', np.uint8, (None,), NdarrayCodec(), False),
        UnischemaField('array_signal', np.float16, (None,signal_len), NdarrayCodec(), False),
        UnischemaField('array_spectrogram', np.float16, (None,spectrogram_len), NdarrayCodec(), False),
    ])
    return schema


def row_generator(pandas_row):
    """Returns a single entry in the generated dataset. Return a bunch of random values as an example."""
    petastorm_dict = pandas_row.to_dict()
    return petastorm_dict


def generate_petastorm_dataset(output_url, signal_len, spectrogram_len):
    rowgroup_size_mb = 256
    schema = define_schema(signal_len,spectrogram_len)

    spark = SparkSession.builder
    spark = spark.config("spark.driver.maxResultSize", "{YOUR-VALUE}")
    spark = spark.config("spark.driver.memory", "{YOUR-VALUE}")
    spark = spark.config("spark.sql.broadcastTimeout", "{YOUR-VALUE}")
    spark = spark.config("spark.sql.debug.maxToStringFields", "{YOUR-VALUE}")
    spark = spark.config("spark.network.timeout", "{YOUR-VALUE}")
    spark = spark.config("spark.executor.heartbeatInterval", "{YOUR-VALUE}")
    spark = spark.config("spark.executor.extraJavaOptions",
                          "-XX:+UseG1GC -XX:+UnlockDiagnosticVMOptions -XX:+G1SummarizeConcMark \
                          -XX:InitiatingHeapOccupancyPercent=35 -verbose:gc -XX:+PrintGCDetails \
                          -XX:+PrintGCDateStamps -XX:OnOutOfMemoryError='kill -9 %p'")
    spark = spark.master('local[2]').getOrCreate()
    sc = spark.sparkContext

    # Wrap dataset materialization portion. Will take care of setting up spark environment variables as
    # well as saving petastorm specific metadata
    rows_count = 10
    with materialize_dataset(spark, output_url, schema, rowgroup_size_mb):

        rows_rdd = sc.parallelize(range(rows_count)) \
            .map(row_generator) \
            .map(lambda x: dict_to_spark_row(schema, x))

        spark.createDataFrame(rows_rdd, schema.as_spark_schema()) \
            .coalesce(10) \
            .write \
            .mode('overwrite') \
            .parquet(output_url)

    return None


def main():
    generate_petastorm_dataset()
    return None


if __name__ == "__main__":
    main()