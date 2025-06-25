#!/usr/bin/env python3
import os
import re
import random
from datetime import date, timedelta
from dotenv import load_dotenv
import psycopg2
from psycopg2 import OperationalError
from faker import Faker
from itertools import product

def transform_mysql_to_postgres(stmt: str) -> str:
    # 1) INT AUTO_INCREMENT → SERIAL PRIMARY KEY
    stmt = re.sub(
        r'\bINT\s+AUTO_INCREMENT\s+PRIMARY\s+KEY\b',
        'SERIAL PRIMARY KEY',
        stmt,
        flags=re.IGNORECASE
    )
    # 2) Any leftover AUTO_INCREMENT → SERIAL
    stmt = re.sub(
        r'\bINT\s+AUTO_INCREMENT\b',
        'SERIAL',
        stmt,
        flags=re.IGNORECASE
    )
    # 3) Remove backticks
    stmt = stmt.replace('`', '')
    # 4) Strip ENGINE/CHARSET clauses
    stmt = re.sub(r'ENGINE=\w+\s*', '', stmt, flags=re.IGNORECASE)
    stmt = re.sub(r'DEFAULT CHARSET=\w+\s*', '', stmt, flags=re.IGNORECASE)
    return stmt

def main():
    load_dotenv()
    host = os.getenv("PG_DB_HOST")
    port = os.getenv("PG_DB_PORT", 5432)
    user = os.getenv("PG_DB_USER")
    pwd  = os.getenv("PG_DB_PASSWORD")
    db   = os.getenv("PG_DB_NAME")

    # Connect to RDS
    try:
        conn = psycopg2.connect(
            host=host, port=port, user=user, password=pwd, dbname=db
        )
        conn.autocommit = True
        cur = conn.cursor()
    except OperationalError as e:
        print("❌ Could not connect to Postgres:", e)
        return

    # 1) Load & apply schema
    with open("./database/Schema_test.sql", "r") as f:
        raw = f.read()

    for chunk in raw.split(";"):
        stmt = chunk.strip()
        if not stmt:
            continue
        pg = transform_mysql_to_postgres(stmt)
        try:
            cur.execute(pg + ";")
            print("✅", pg.splitlines()[0])
        except Exception as e:
            print("⚠️ Could not run:", pg.splitlines()[0], "→", e)

    # 2) Seed data
    fake = Faker()
    NUM = 10
    roles = ["Developer", "Tester", "Manager", "DevOps", "Analyst"]
    statuses = ["active", "completed", "on hold"]

    # 2a) Employees
    employees = [
        (
            fake.name(),
            fake.unique.email(),
            random.choice(roles),
            fake.date_between(start_date=date(2020,1,1), end_date=date.today()).isoformat()
        )
        for _ in range(NUM)
    ]
    cur.executemany(
        "INSERT INTO employees (name, email, role, joined_at) VALUES (%s,%s,%s,%s)",
        employees
    )

    # 2b) Clients
    clients = [
        (fake.company(), fake.bs().title(), fake.city())
        for _ in range(NUM)
    ]
    cur.executemany(
        "INSERT INTO clients (name, industry, location) VALUES (%s,%s,%s)",
        clients
    )

    # 2c) Projects
    cur.execute("SELECT id FROM clients")
    client_ids = [r[0] for r in cur.fetchall()]
    projects = []
    for _ in range(NUM):
        start_dt = fake.date_between(start_date=date(2021,1,1), end_date=date.today())
        end_dt   = start_dt + timedelta(days=random.randint(30,365))
        projects.append((
            fake.bs().title(),
            random.choice(client_ids),
            start_dt.isoformat(),
            end_dt.isoformat(),
            random.choice(statuses)
        ))
    cur.executemany(
        "INSERT INTO projects (name, client_id, start_date, end_date, status) VALUES (%s,%s,%s,%s,%s)",
        projects
    )

    # 2d) Assignments (one per loop, up to NUM)
    cur.execute("SELECT id FROM employees")
    emp_ids = [r[0] for r in cur.fetchall()]
    cur.execute("SELECT id FROM projects")
    proj_ids = [r[0] for r in cur.fetchall()]
    assignments = [
        (
            random.choice(emp_ids),
            random.choice(proj_ids),
            fake.date_between(start_date=date(2021,1,1), end_date=date.today()).isoformat(),
            random.choice(roles)
        )
        for _ in range(NUM)
    ]
    cur.executemany(
        "INSERT INTO employee_project_assignments (employee_id, project_id, assigned_on, role_on_project) VALUES (%s,%s,%s,%s)",
        assignments
    )

    # 2e) Skills
    skill_names = [fake.unique.word().title() for _ in range(NUM)]
    cur.executemany(
        "INSERT INTO skills (name) VALUES (%s)",
        [(s,) for s in skill_names]
    )

    # 2f) Employee‐Skills
    cur.execute("SELECT id FROM skills")
    skill_ids = [r[0] for r in cur.fetchall()]

    # Build every possible (employee, skill) pair
    all_pairs = [(emp, sk) for emp in emp_ids for sk in skill_ids]

    # Sample exactly NUM unique pairs
    emp_skills = random.sample(all_pairs, NUM)

    cur.executemany(
        "INSERT INTO employee_skills (employee_id, skill_id) VALUES (%s,%s)",
        emp_skills
    )

    # Done
    cur.close()
    conn.close()
    print(f"🎉 Seeded {NUM} rows into each table.")

if __name__ == "__main__":
    main()
