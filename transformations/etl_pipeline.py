# Databricks notebook source
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window
from functools import reduce

# COMMAND ----------

# =========================================
# 1. CONFIGURATION
# =========================================
storage_account = "tokyoolympicdatabk"
container_name = "tokyo-olympic-data"
container_name = "tokyo-olympic-data"
tenant_id = "<tenant_id>"
client_id = "<client_id>"
client_secret = "<client_secret>"

# For production, store this in Databricks secrets:
# client_secret = dbutils.secrets.get(scope="azure-kv-scope", key="sp-client-secret")


base_path = f"abfss://{container_name}@{storage_account}.dfs.core.windows.net/"
raw_path = base_path + "raw-data/"

bronze_path = base_path + "bronze/"
silver_path = base_path + "silver/"
gold_path = base_path + "gold/"
audit_path = base_path + "audit/"

spark.conf.set(f"fs.azure.account.auth.type.{storage_account}.dfs.core.windows.net", "OAuth")
spark.conf.set(
    f"fs.azure.account.oauth.provider.type.{storage_account}.dfs.core.windows.net",
    "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider"
)
spark.conf.set(
    f"fs.azure.account.oauth2.client.id.{storage_account}.dfs.core.windows.net",
    client_id
)
spark.conf.set(
    f"fs.azure.account.oauth2.client.secret.{storage_account}.dfs.core.windows.net",
    client_secret
)
spark.conf.set(
    f"fs.azure.account.oauth2.client.endpoint.{storage_account}.dfs.core.windows.net",
    f"https://login.microsoftonline.com/{tenant_id}/oauth2/token"
)

# COMMAND ----------

# =========================================
# 2. UTILITY FUNCTIONS
# =========================================
def log_message(message: str) -> None:
    print(f"[INFO] {message}")


def standardize_string_columns(df: DataFrame) -> DataFrame:
    string_cols = [field.name for field in df.schema.fields if str(field.dataType) == "StringType()"]
    for col_name in string_cols:
        df = df.withColumn(col_name, F.trim(F.regexp_replace(F.col(col_name), r"\s+", " ")))
    return df


def standardize_column_names(df: DataFrame) -> DataFrame:
    for old_col in df.columns:
        new_col = old_col.strip().replace(" ", "_").replace("-", "_")
        df = df.withColumnRenamed(old_col, new_col)
    return df


def add_audit_columns(df: DataFrame, source_file_name: str) -> DataFrame:
    return (
        df.withColumn("source_file_name", F.lit(source_file_name))
          .withColumn("ingestion_timestamp", F.current_timestamp())
    )


def validate_non_empty(df: DataFrame, df_name: str) -> None:
    row_count = df.count()
    if row_count == 0:
        raise ValueError(f"{df_name} is empty.")
    log_message(f"{df_name} row count: {row_count}")


def write_parquet(df: DataFrame, path: str, mode: str = "overwrite") -> None:
    df.write.mode(mode).parquet(path)
    log_message(f"Written parquet to: {path}")


def write_csv(df: DataFrame, path: str, mode: str = "overwrite") -> None:
    df.write.mode(mode).option("header", "true").csv(path)
    log_message(f"Written csv to: {path}")

# COMMAND ----------

# =========================================
# 3. READ RAW FILES
# =========================================
athletes_raw = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(raw_path + "athletes.csv")

coaches_raw = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(raw_path + "coaches.csv")

entriesgender_raw = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(raw_path + "entriesGender.csv")

medals_raw = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(raw_path + "medals.csv")

teams_raw = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(raw_path + "teams.csv")

log_message("Raw files loaded successfully.")

# COMMAND ----------

# =========================================
# 4. BRONZE LAYER
# Raw ingestion + audit columns
# =========================================
athletes_bronze = add_audit_columns(athletes_raw, "athletes.csv")
coaches_bronze = add_audit_columns(coaches_raw, "coaches.csv")
entriesgender_bronze = add_audit_columns(entriesgender_raw, "entriesGender.csv")
medals_bronze = add_audit_columns(medals_raw, "medals.csv")
teams_bronze = add_audit_columns(teams_raw, "teams.csv")

validate_non_empty(athletes_bronze, "athletes_bronze")
validate_non_empty(coaches_bronze, "coaches_bronze")
validate_non_empty(entriesgender_bronze, "entriesgender_bronze")
validate_non_empty(medals_bronze, "medals_bronze")
validate_non_empty(teams_bronze, "teams_bronze")

write_parquet(athletes_bronze, bronze_path + "athletes")
write_parquet(coaches_bronze, bronze_path + "coaches")
write_parquet(entriesgender_bronze, bronze_path + "entriesgender")
write_parquet(medals_bronze, bronze_path + "medals")
write_parquet(teams_bronze, bronze_path + "teams")

# COMMAND ----------

# =========================================
# 5. SILVER LAYER
# Cleaning, standardization, typing
# =========================================

# Athletes
athletes_silver = (
    athletes_bronze
    .transform(standardize_column_names)
    .transform(standardize_string_columns)
    .select(
        F.col("PersonName").alias("athlete_name"),
        F.col("Country").alias("country"),
        F.col("Discipline").alias("discipline"),
        "source_file_name",
        "ingestion_timestamp"
    )
    .dropDuplicates()
)

# Coaches
coaches_silver = (
    coaches_bronze
    .transform(standardize_column_names)
    .transform(standardize_string_columns)
    .select(
        F.col("Name").alias("coach_name"),
        F.col("Country").alias("country"),
        F.col("Discipline").alias("discipline"),
        F.col("Event").alias("event"),
        "source_file_name",
        "ingestion_timestamp"
    )
    .dropDuplicates()
)

# Entries Gender
entriesgender_silver = (
    entriesgender_bronze
    .transform(standardize_column_names)
    .transform(standardize_string_columns)
    .select(
        F.col("Discipline").alias("discipline"),
        F.col("Female").cast(IntegerType()).alias("female"),
        F.col("Male").cast(IntegerType()).alias("male"),
        F.col("Total").cast(IntegerType()).alias("total"),
        "source_file_name",
        "ingestion_timestamp"
    )
    .dropDuplicates()
    .withColumn("female_percentage", F.round((F.col("female") / F.col("total")) * 100, 2))
    .withColumn("male_percentage", F.round((F.col("male") / F.col("total")) * 100, 2))
)

# Medals
medals_silver = (
    medals_bronze
    .transform(standardize_column_names)
    .transform(standardize_string_columns)
    .select(
        F.col("Rank").cast(IntegerType()).alias("rank"),
        F.col("TeamCountry").alias("country"),
        F.col("Gold").cast(IntegerType()).alias("gold"),
        F.col("Silver").cast(IntegerType()).alias("silver"),
        F.col("Bronze").cast(IntegerType()).alias("bronze"),
        F.col("Total").cast(IntegerType()).alias("total"),
        F.col("Rank_by_Total").cast(IntegerType()).alias("rank_by_total"),
        "source_file_name",
        "ingestion_timestamp"
    )
    .dropDuplicates()
    .withColumn("calculated_total", F.col("gold") + F.col("silver") + F.col("bronze"))
    .withColumn(
        "total_validation_status",
        F.when(F.col("total") == F.col("calculated_total"), F.lit("VALID")).otherwise(F.lit("MISMATCH"))
    )
)

# Teams
teams_silver = (
    teams_bronze
    .transform(standardize_column_names)
    .transform(standardize_string_columns)
    .select(
        F.col("TeamName").alias("team_name"),
        F.col("Discipline").alias("discipline"),
        F.col("Country").alias("country"),
        F.col("Event").alias("event"),
        "source_file_name",
        "ingestion_timestamp"
    )
    .dropDuplicates()
)

# COMMAND ----------

# =========================================
# 6. DATA QUALITY CHECKS
# =========================================
validate_non_empty(athletes_silver, "athletes_silver")
validate_non_empty(coaches_silver, "coaches_silver")
validate_non_empty(entriesgender_silver, "entriesgender_silver")
validate_non_empty(medals_silver, "medals_silver")
validate_non_empty(teams_silver, "teams_silver")

medal_mismatches = medals_silver.filter(F.col("total_validation_status") == "MISMATCH")
log_message(f"Medal total mismatches: {medal_mismatches.count()}")

null_summary = medals_silver.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in medals_silver.columns
])
display(null_summary)

# COMMAND ----------

# =========================================
# 7. WRITE SILVER OUTPUTS
# =========================================
write_parquet(athletes_silver, silver_path + "athletes")
write_parquet(coaches_silver, silver_path + "coaches")
write_parquet(entriesgender_silver, silver_path + "entriesgender")
write_parquet(medals_silver, silver_path + "medals")
write_parquet(teams_silver, silver_path + "teams")

# COMMAND ----------

# =========================================
# 8. GOLD LAYER
# Business-ready datasets
# =========================================

# Country athlete count
country_athlete_counts = (
    athletes_silver
    .groupBy("country")
    .agg(F.count("*").alias("athlete_count"))
)

# Country medal summary
country_medal_summary = (
    medals_silver
    .join(country_athlete_counts, on="country", how="left")
    .select(
        "country",
        "gold",
        "silver",
        "bronze",
        "total",
        "rank",
        "rank_by_total",
        "calculated_total",
        "total_validation_status",
        "athlete_count"
    )
)

# Country + discipline athlete summary
country_discipline_summary = (
    athletes_silver
    .groupBy("country", "discipline")
    .agg(F.count("*").alias("athlete_count"))
    .join(
        entriesgender_silver.select("discipline", "female", "male", "total", "female_percentage", "male_percentage"),
        on="discipline",
        how="left"
    )
)

# Team event summary
team_event_summary = (
    teams_silver
    .join(
        entriesgender_silver.select("discipline", "female", "male", "total", "female_percentage", "male_percentage"),
        on="discipline",
        how="left"
    )
)

# Coach team summary
coach_team_summary = (
    coaches_silver.alias("c")
    .join(
        teams_silver.select("country", "discipline", "event", "team_name").dropDuplicates().alias("t"),
        on=["country", "discipline", "event"],
        how="left"
    )
    .select(
        F.col("c.coach_name"),
        F.col("c.country"),
        F.col("c.discipline"),
        F.col("c.event"),
        F.col("t.team_name")
    )
)

# Top disciplines by athlete count
top_disciplines_by_athletes = (
    athletes_silver
    .groupBy("discipline")
    .agg(F.count("*").alias("athlete_count"))
    .orderBy(F.col("athlete_count").desc())
)

# Top countries by athlete participation
top_countries_by_athletes = (
    athletes_silver
    .groupBy("country")
    .agg(F.count("*").alias("athlete_count"))
    .orderBy(F.col("athlete_count").desc())
)

# Coach count by country
coach_count_by_country = (
    coaches_silver
    .groupBy("country")
    .agg(F.count("*").alias("coach_count"))
    .orderBy(F.col("coach_count").desc())
)

# Team count by discipline
team_count_by_discipline = (
    teams_silver
    .groupBy("discipline")
    .agg(F.count("*").alias("team_count"))
    .orderBy(F.col("team_count").desc())
)

# COMMAND ----------

# =========================================
# 9. DERIVED RANKING
# =========================================
gold_rank_window = Window.orderBy(F.col("gold").desc(), F.col("silver").desc(), F.col("bronze").desc())

top_countries_by_medals = (
    medals_silver
    .withColumn("gold_rank_derived", F.dense_rank().over(gold_rank_window))
    .select(
        "country",
        "gold",
        "silver",
        "bronze",
        "total",
        "rank",
        "rank_by_total",
        "gold_rank_derived"
    )
    .orderBy(F.col("gold").desc(), F.col("silver").desc(), F.col("bronze").desc())
)

# COMMAND ----------

# =========================================
# 10. WRITE GOLD OUTPUTS
# =========================================
write_parquet(country_medal_summary, gold_path + "country_medal_summary")
write_parquet(country_discipline_summary, gold_path + "country_discipline_summary")
write_parquet(team_event_summary, gold_path + "team_event_summary")
write_parquet(coach_team_summary, gold_path + "coach_team_summary")
write_parquet(top_disciplines_by_athletes, gold_path + "top_disciplines_by_athletes")
write_parquet(top_countries_by_athletes, gold_path + "top_countries_by_athletes")
write_parquet(coach_count_by_country, gold_path + "coach_count_by_country")
write_parquet(team_count_by_discipline, gold_path + "team_count_by_discipline")
write_parquet(top_countries_by_medals, gold_path + "top_countries_by_medals")

# COMMAND ----------

# =========================================
# 11. AUDIT TABLE
# =========================================
audit_records = [
    ("athletes_bronze", athletes_bronze.count()),
    ("coaches_bronze", coaches_bronze.count()),
    ("entriesgender_bronze", entriesgender_bronze.count()),
    ("medals_bronze", medals_bronze.count()),
    ("teams_bronze", teams_bronze.count()),
    ("athletes_silver", athletes_silver.count()),
    ("coaches_silver", coaches_silver.count()),
    ("entriesgender_silver", entriesgender_silver.count()),
    ("medals_silver", medals_silver.count()),
    ("teams_silver", teams_silver.count()),
    ("country_medal_summary", country_medal_summary.count()),
    ("country_discipline_summary", country_discipline_summary.count()),
    ("team_event_summary", team_event_summary.count()),
    ("coach_team_summary", coach_team_summary.count()),
    ("top_disciplines_by_athletes", top_disciplines_by_athletes.count()),
    ("top_countries_by_athletes", top_countries_by_athletes.count()),
    ("coach_count_by_country", coach_count_by_country.count()),
    ("team_count_by_discipline", team_count_by_discipline.count()),
    ("top_countries_by_medals", top_countries_by_medals.count())
]

audit_df = spark.createDataFrame(audit_records, ["dataset_name", "row_count"]) \
    .withColumn("audit_timestamp", F.current_timestamp())

write_parquet(audit_df, audit_path + "pipeline_audit")
display(audit_df)

# COMMAND ----------

