SQLALCHEMY_DATABASE_URI = 'sqlite:////app/superset_home/superset.db'

FEATURE_FLAGS = {
    "ALLOW_SQLITE_DATABASES": True,
}

# Empêche Superset de refuser les URI SQLite
ALLOWED_DATABASES = ['sqlite']