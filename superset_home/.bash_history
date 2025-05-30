cat /app/pythonpath/superset_config.py
docker exec -it superset superset dbs add --database-name profits --sqlalchemy-uri sqlite:////app/profits.db
docker exec -it superset superset dbs add --database-name profits --sqlalchemy-uri sqlite:////app/profits.db
exit
