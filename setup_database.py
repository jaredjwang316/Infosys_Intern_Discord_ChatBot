#!/usr/bin/env python3
# db_setup.py

import os
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import errorcode

def main():
    # 1) Load credentials from .env
    load_dotenv()
    DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASS = os.getenv("DB_PASS", "")
    DB_NAME = os.getenv("DB_NAME", "chatops")

    # 2) Connect to MySQL (no database specified yet)
    try:
        tmp_conn = mysql.connector.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASS
        )
        tmp_cursor = tmp_conn.cursor()
    except mysql.connector.Error as err:
        print(f"❌ Could not connect to MySQL server: {err}")
        return

    # 3) Create the database if it doesn't exist
    try:
        tmp_cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` "
            "DEFAULT CHARACTER SET 'utf8mb4'"
        )
        print(f"✔️ Database `{DB_NAME}` ensured.")
    except mysql.connector.Error as err:
        print(f"❌ Failed creating database: {err}")
        return
    finally:
        tmp_cursor.close()
        tmp_conn.close()

    # 4) Connect to the new (or existing) database
    try:
        conn = mysql.connector.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME
        )
        cur = conn.cursor()
    except mysql.connector.Error as err:
        print(f"❌ Could not connect to database `{DB_NAME}`: {err}")
        return

    # ── 5) Define DDL strings ────────────────────────────────────────────────────
    ddl_employee = """
CREATE TABLE IF NOT EXISTS Employee (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(100)   NOT NULL,
    address     VARCHAR(200)   NOT NULL,
    start_date  DATE           NOT NULL
) ENGINE=InnoDB;"""

    ddl_department = """
CREATE TABLE IF NOT EXISTS Department (
    id    INT AUTO_INCREMENT PRIMARY KEY,
    name  VARCHAR(100) NOT NULL UNIQUE
) ENGINE=InnoDB;"""

    ddl_project = """
CREATE TABLE IF NOT EXISTS Project (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    name       VARCHAR(100) NOT NULL,
    start_date DATE           NOT NULL,
    end_date   DATE
) ENGINE=InnoDB;"""

    # ── 6) Create & seed Employee ──────────────────────────────────────────────
    cur.execute(ddl_employee)
    print("✔️ Table `Employee` ensured.")
    cur.execute("SELECT COUNT(*) FROM Employee")
    (emp_count,) = cur.fetchone()
    if emp_count == 0:
        employees = [
            ("Alice", "123 Main St",   "2025-06-02"),
            ("Bob",   "456 Oak Ave",    "2025-05-20"),
            ("Carol", "789 Pine Rd",    "2025-06-10"),
            ("Dave",  "321 Elm Blvd",   "2025-06-15"),
            ("Muh",   "123 Treehouse",  "2025-08-12"),
        ]
        cur.executemany(
            "INSERT INTO Employee (name, address, start_date) VALUES (%s, %s, %s)",
            employees
        )
        conn.commit()
        print("✔️ Seeded dummy data into `Employee`.")
    else:
        print(f"ℹ️ `Employee` has {emp_count} rows; skipping seeding.")

    # ── 7) Create & seed Department ────────────────────────────────────────────
    cur.execute(ddl_department)
    print("✔️ Table `Department` ensured.")
    cur.execute("SELECT COUNT(*) FROM Department")
    (dept_count,) = cur.fetchone()
    if dept_count == 0:
        departments = [
            ("Human Resources",),
            ("Engineering",),
            ("Marketing",),
            ("Sales",),
        ]
        cur.executemany(
            "INSERT INTO Department (name) VALUES (%s)",
            departments
        )
        conn.commit()
        print("✔️ Seeded dummy data into `Department`.")
    else:
        print(f"ℹ️ `Department` has {dept_count} rows; skipping seeding.")

    # ── 8) Create & seed Project ───────────────────────────────────────────────
    cur.execute(ddl_project)
    print("✔️ Table `Project` ensured.")
    cur.execute("SELECT COUNT(*) FROM Project")
    (proj_count,) = cur.fetchone()
    if proj_count == 0:
        projects = [
            ("Apollo",  "2025-01-01", "2025-06-30"),
            ("Zephyr",  "2025-03-15", None),
            ("Hermes",  "2025-05-01", "2025-07-31"),
        ]
        cur.executemany(
            "INSERT INTO Project (name, start_date, end_date) VALUES (%s, %s, %s)",
            projects
        )
        conn.commit()
        print("✔️ Seeded dummy data into `Project`.")
    else:
        print(f"ℹ️ `Project` has {proj_count} rows; skipping seeding.")

    # ── 9) Write DDL to schema.txt ──────────────────────────────────────────────
    with open("schema.txt", "w") as f:
        f.write(ddl_employee + "\n\n")
        f.write(ddl_department + "\n\n")
        f.write(ddl_project + "\n\n")
    print("✔️ schema.txt written with CREATE TABLE statements.")

    # ──10) Cleanup ──────────────────────────────────────────────────────────────
    cur.close()
    conn.close()
    print("✅ Database setup complete.")

if __name__ == "__main__":
    main()
