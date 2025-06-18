#!/usr/bin/env python3
import os
import random
from datetime import date, timedelta

from dotenv import load_dotenv
import mysql.connector
from faker import Faker

fake = Faker()

def main():
    # 1) Load credentials
    load_dotenv()
    DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASS = os.getenv("DB_PASS", "")
    DB_NAME = os.getenv("DB_NAME", "chatops")

    # 2) Ensure database exists
    admin = mysql.connector.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASS
    )
    admin_cursor = admin.cursor()
    admin_cursor.execute(
        f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` DEFAULT CHARACTER SET 'utf8mb4'"
    )
    admin_cursor.close()
    admin.close()
    print(f"✔️ Database `{DB_NAME}` ensured.")

    # 3) Connect to chatops DB
    conn = mysql.connector.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME
    )
    cur = conn.cursor()

    # 4) Define DDL
    ddl_employee = """
CREATE TABLE IF NOT EXISTS Employee (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(100)   NOT NULL,
    address     VARCHAR(200)   NOT NULL,
    start_date  DATE           NOT NULL
) ENGINE=InnoDB;
"""
    ddl_department = """
CREATE TABLE IF NOT EXISTS Department (
    id    INT AUTO_INCREMENT PRIMARY KEY,
    name  VARCHAR(100) NOT NULL UNIQUE
) ENGINE=InnoDB;
"""
    ddl_project = """
CREATE TABLE IF NOT EXISTS Project (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    name       VARCHAR(100) NOT NULL,
    start_date DATE           NOT NULL,
    end_date   DATE
) ENGINE=InnoDB;
"""

    # 5) Create tables
    for ddl in (ddl_employee, ddl_department, ddl_project):
        cur.execute(ddl)
    conn.commit()
    print("✔️ Tables ensured.")

    # 6) Truncate for a clean slate
    for tbl in ("Employee", "Department", "Project"):
        cur.execute(f"TRUNCATE TABLE {tbl}")
    conn.commit()
    print("🔄 Tables truncated.")

    # 7) Seed 20 Employees
    employees = []
    for _ in range(20):
        name     = fake.name()
        address  = fake.address().replace("\n", ", ")
        start_dt = fake.date_between_dates(
            date_start=date(2025,1,1),
            date_end=date(2025,12,31)
        ).isoformat()
        employees.append((name, address, start_dt))
    cur.executemany(
        "INSERT INTO Employee (name, address, start_date) VALUES (%s, %s, %s)",
        employees
    )
    conn.commit()
    print("✔️ Seeded 20 random Employees.")

    # 8) Seed 5 Departments (unique names)
    dept_pool = [
        "Human Resources", "Engineering", "Marketing", "Sales",
        "Finance", "IT", "Operations", "Customer Service",
        "Legal", "Research"
    ]
    departments = random.sample(dept_pool, 5)
    cur.executemany(
        "INSERT INTO Department (name) VALUES (%s)",
        [(d,) for d in departments]
    )
    conn.commit()
    print(f"✔️ Seeded Departments: {departments}")

    # 9) Seed 10 Projects
    projects = []
    for _ in range(10):
        proj_name = fake.catch_phrase()
        start_dt  = fake.date_between_dates(
            date_start=date(2025,1,1),
            date_end=date(2025,12,31)
        )
        end_dt    = start_dt + timedelta(days=random.randint(30,180))
        projects.append((proj_name, start_dt.isoformat(), end_dt.isoformat()))
    cur.executemany(
        "INSERT INTO Project (name, start_date, end_date) VALUES (%s, %s, %s)",
        projects
    )
    conn.commit()
    print("✔️ Seeded 10 random Projects.")

    # 10) Write schema.txt with raw DDL
    with open("./database/schema.txt", "w") as f:
        f.write(ddl_employee + "\n")
        f.write(ddl_department + "\n")
        f.write(ddl_project + "\n")
    print("✔️ schema.txt written.")

    # 11) Cleanup
    cur.close()
    conn.close()
    print("✅ Database setup complete.")

if __name__ == "__main__":
    main()