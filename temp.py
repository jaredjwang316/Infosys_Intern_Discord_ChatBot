import os
import psycopg2
from psycopg2 import OperationalError

PG_CONFIG = {
    "host":     os.getenv("DB_HOST"),
    "port":     os.getenv("DB_PORT", 5432),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "dbname":   os.getenv("DB_NAME"),
}

query = """
DROP TABLE IF EXISTS channels;
"""

conn = psycopg2.connect(**PG_CONFIG)
conn.autocommit = True
cur = conn.cursor()

try:
    cur.execute(query)
    print("Successfully dropped channels table.")
except OperationalError as e:
    print(f"OperationalError: {e}")
except psycopg2.Error as e:
    print(f"Database error: {e}")
finally:
    cur.close()
    conn.close()