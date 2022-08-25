from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *
import logging

logger = logging.getLogger(__name__)


class EHRSparkPipeline:
    def __init__(self, spark: SparkSession = None):
        self.spark = spark or SparkSession.builder \
            .appName("EHRProcessing") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "400") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()

    def read_ehr_data(self, path: str, format: str = "parquet") -> DataFrame:
        return self.spark.read.format(format).load(path)

    def standardize_dates(self, df: DataFrame) -> DataFrame:
        date_cols = ["admission_date", "discharge_date", "dob", "procedure_date"]
        for col in date_cols:
            if col in df.columns:
                df = df.withColumn(col, F.to_timestamp(F.col(col)))
        return df

    def compute_los(self, df: DataFrame) -> DataFrame:
        if "admission_date" in df.columns and "discharge_date" in df.columns:
            df = df.withColumn("length_of_stay_days",
                F.datediff(F.col("discharge_date"), F.col("admission_date")))
        return df

    def aggregate_patient_history(self, df: DataFrame) -> DataFrame:
        return df.groupBy("patient_id").agg(
            F.count("*").alias("total_visits"),
            F.avg("length_of_stay_days").alias("avg_los"),
            F.max("admission_date").alias("last_admission"),
            F.countDistinct("diagnosis_code").alias("unique_diagnoses"),
            F.sum(F.when(F.col("icu_flag") == 1, 1).otherwise(0)).alias("icu_admissions")
        )

    def write_to_delta(self, df: DataFrame, path: str, mode: str = "overwrite"):
        df.write.format("delta").mode(mode).partitionBy("admission_year").save(path)
        logger.info(f"Written {df.count()} records to {path}")
