#!/usr/bin/env python3
import os
import random
from datetime import date, timedelta

from dotenv import load_dotenv
import mysql.connector
from faker import Faker

fake = Faker()

def main():
    # ── 1) Load credentials ───────────────────────────────────────────
    load_dotenv()
    DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASS = os.getenv("DB_PASS", "")
    DB_NAME = os.getenv("DB_NAME", "chatops")

    # ── 2) Ensure database exists ─────────────────────────────────────
    admin = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASS)
    admin_cursor = admin.cursor()
    admin_cursor.execute(
        f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` DEFAULT CHARACTER SET 'utf8mb4';"
    )
    admin_cursor.close()
    admin.close()

    # ── 3) Connect to the chatops DB ──────────────────────────────────
    conn = mysql.connector.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME
    )
    cur = conn.cursor()

    # ── 4) Load & execute your schema SQL ─────────────────────────────
    with open("./database/schema_test.sql", "r") as f:
        schema_sql = f.read()

    with open("./database/schema.txt", "w") as out:
        # ensure each CREATE TABLE ends up on its own block
        for stmt in schema_sql.split(';'):
            stmt = stmt.strip()
            if not stmt:
                continue
            out.write(stmt + ";\n\n")

    # --- now actually execute it against MySQL ---
    for stmt in schema_sql.split(';'):
        stmt = stmt.strip()
        if not stmt:
            continue
        cur.execute(stmt + ';')
    conn.commit()

    
    # split on semicolons and execute each statement individually
    for stmt in schema_sql.split(';'):
        stmt = stmt.strip()
        if not stmt:
            continue
        cur.execute(stmt)
    conn.commit()

    #

    # ── 5) Truncate tables in FK-safe order ───────────────────────────
        # ── 5) Truncate tables (disable FK checks to allow truncation) ────
    cur.execute("SET FOREIGN_KEY_CHECKS = 0;")
    for tbl in (
        "employee_skills",
        "employee_project_assignments",
        "projects",
        "clients",
        "skills",
        "employees",
    ):
        cur.execute(f"TRUNCATE TABLE {tbl};")
    cur.execute("SET FOREIGN_KEY_CHECKS = 1;")
    conn.commit()


    # ── 6) Seed 20 Employees ──────────────────────────────────────────
    roles = ["Developer", "Tester", "Manager", "DevOps", "Analyst"]
    employees = []
    for _ in range(20):
        employees.append((
            fake.name(),
            fake.unique.email(),
            random.choice(roles),
            fake.date_between(start_date=date(2020, 1, 1), end_date=date.today()).isoformat()
        ))
    cur.executemany(
        "INSERT INTO employees (name, email, role, joined_at) VALUES (%s, %s, %s, %s)",
        employees
    )
    conn.commit()

    # ── 7) Seed 5 Clients ─────────────────────────────────────────────
    industries = ["Finance", "Healthcare", "Retail", "Manufacturing", "Technology"]
    clients = []
    for _ in range(5):
        clients.append((
            fake.company(),
            random.choice(industries),
            fake.city()
        ))
    cur.executemany(
        "INSERT INTO clients (name, industry, location) VALUES (%s, %s, %s)",
        clients
    )
    conn.commit()

    # ── 8) Seed 10 Projects ───────────────────────────────────────────
    cur.execute("SELECT id FROM clients")
    client_ids = [row[0] for row in cur.fetchall()]

    statuses = ["active", "completed", "on hold"]
    projects = []
    for _ in range(10):
        start_dt = fake.date_between(start_date=date(2021, 1, 1), end_date=date.today())
        end_dt   = start_dt + timedelta(days=random.randint(30, 365))
        projects.append((
            fake.bs().title(),
            random.choice(client_ids),
            start_dt.isoformat(),
            end_dt.isoformat(),
            random.choice(statuses)
        ))
    cur.executemany(
        "INSERT INTO projects (name, client_id, start_date, end_date, status) VALUES (%s, %s, %s, %s, %s)",
        projects
    )
    conn.commit()

    # ── 9) Seed Assignments (1–3 projects per employee) ───────────────
    cur.execute("SELECT id FROM employees")
    emp_ids = [r[0] for r in cur.fetchall()]
    cur.execute("SELECT id FROM projects")
    proj_ids = [r[0] for r in cur.fetchall()]

    assignments = []
    for emp in emp_ids:
        for pj in random.sample(proj_ids, k=random.randint(1, 3)):
            assignments.append((
                emp,
                pj,
                fake.date_between(start_date=date(2021, 1, 1), end_date=date.today()).isoformat(),
                random.choice(roles)
            ))
    cur.executemany(
        "INSERT INTO employee_project_assignments (employee_id, project_id, assigned_on, role_on_project) VALUES (%s, %s, %s, %s)",
        assignments
    )
    conn.commit()

    # ── 10) Seed Skills & Link to Employees ───────────────────────────
    skill_list = ["Python", "Java", "SQL", "Project Management", "AWS", "Docker", "Kubernetes"]
    cur.executemany(
        "INSERT INTO skills (name) VALUES (%s)",
        [(s,) for s in skill_list]
    )
    conn.commit()

    cur.execute("SELECT id FROM skills")
    skill_ids = [r[0] for r in cur.fetchall()]
    emp_skills = []
    for emp in emp_ids:
        for sk in random.sample(skill_ids, k=random.randint(1, 4)):
            emp_skills.append((emp, sk))
    cur.executemany(
        "INSERT INTO employee_skills (employee_id, skill_id) VALUES (%s, %s)",
        emp_skills
    )
    conn.commit()

    # ── 11) Tear down ───────────────────────────────────────────────────
    cur.close()
    conn.close()
    print("Database seeded: 20 employees, 5 clients, 10 projects (plus assignments and skills).")

if __name__ == "__main__":
    main()
