from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
from pyspark.ml.feature import MinMaxScaler, VectorAssembler

# Créer la SparkSession
spark = SparkSession.builder.appName("PreparationData").getOrCreate()

# Charger les données CSV depuis HDFS
df = spark.read.csv("hdfs://namenode:9000/data/historic/*.csv", header=True, inferSchema=True)

# Parser la colonne Date correctement
df = df.withColumn("Date", to_date(col("Date"), "yyyy-MM-dd"))

# Sélectionner et caster la colonne Close
prices = df.select("Date", col("Close").cast("double").alias("Close")).na.drop()

# Normalisation des prix avec MinMaxScaler
assembler = VectorAssembler(inputCols=["Close"], outputCol="features")
prices_vector = assembler.transform(prices)

scaler = MinMaxScaler(inputCol="features", outputCol="scaled")
scaler_model = scaler.fit(prices_vector)
scaled_prices = scaler_model.transform(prices_vector).select("Date", "scaled")

# Trier par date pour conserver l’ordre chronologique
ordered_prices = scaled_prices.orderBy("Date")

# Calcul des tailles
total_rows = ordered_prices.count()
train_size = int(total_rows * 0.7)
val_size = int(total_rows * 0.15)

# Ajouter un index pour faciliter le split (temporaire)
from pyspark.sql.functions import monotonically_increasing_id
ordered_prices = ordered_prices.withColumn("index", monotonically_increasing_id())

# Split des données
train_df = ordered_prices.filter(col("index") < train_size)
val_df = ordered_prices.filter((col("index") >= train_size) & (col("index") < train_size + val_size))
test_df = ordered_prices.filter(col("index") >= train_size + val_size)

# Sauvegarde des datasets en HDFS (format Parquet)
train_df.select("scaled").write.mode("overwrite").parquet("hdfs://namenode:9000/data/processed/train")
val_df.select("scaled").write.mode("overwrite").parquet("hdfs://namenode:9000/data/processed/validation")
test_df.select("scaled").write.mode("overwrite").parquet("hdfs://namenode:9000/data/processed/test")

spark.stop()