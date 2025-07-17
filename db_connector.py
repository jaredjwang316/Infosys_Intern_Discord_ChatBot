import psycopg2
from google.cloud import secretmanager
import os

def access_secret_version(project_id, secret_id, version_id="latest"):
    """Accesses a secret from Google Cloud Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("UTF-8")

def get_db_connection():
    """
    Retrieves database connection parameters from Google Cloud Secret Manager.
    """
    try:
        project_id = "discord-bot-466220"
        db_name = access_secret_version(project_id, "db-name")
        db_user = access_secret_version(project_id, "db-user")
        db_password = access_secret_version(project_id, "db-password")

        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host="127.0.0.1",
            port="5432"
        )

        return conn
    
    except Exception as e:
        print(f"Error accessing database secrets: {e}")
        raise