# TP TRAIDER

## 🌍 Project Overview
Ce projet fournit un environnement Dockerisé prêt à l'emploi pour travailler avec :
- **Hadoop 3.3.6** (pseudo-distribué)
- **Spark 3.5.1** (mode autonome)
- **Kafka 3.7.2** (avec Zookeeper)
- **Python 3**
- **Superset**

## 📊 Architecture
- Hadoop HDFS pour le stockage distribué des fichiers (configuration mono-nœud)
- Spark pour le traitement des données par lots et en streaming
- Kafka pour l'ingestion en streaming

## 🔄 Quick Start

### 1. Build the Docker Image
```bash
docker-compose down
docker-compose up -d --build
docker logs -f bigdata-container
```

### 2. To launch preprocess_data.py
```bash
docker cp ./data/2008_Globla_Markets_Data.csv namenode:/tmp/market.csv
docker exec -it namenode bash
hdfs dfs -mkdir -p /data/historic
hdfs dfs -put /tmp/market.csv /data/historic/
hdfs dfs -ls /data/historic
exit
docker exec -it spark-client spark-submit --master spark://spark-master:7077 /app/preprocess_data.py
```

### 2. To launch lstm.py
```bash
docker cp .\app\lstm_model.h5 namenode:/tmp/lstm_model.h5   
docker exec -it namenode bash
hdfs dfs -mkdir -p /model
hdfs dfs -put -f tmp/lstm_model.h5 /model/
hdfs dfs -ls /model/
exit
docker exec -it spark-client spark-submit --master spark://spark-master:7077 /app/lstm.py
```