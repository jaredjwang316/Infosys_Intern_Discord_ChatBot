-- 1. Employees – Infosys staff
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    role VARCHAR(50) NOT NULL,
    joined_at DATE NOT NULL
);

-- 2. Clients – Represents the external clients Infosys serves.
CREATE TABLE clients (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    industry VARCHAR(50) NOT NULL,
    location VARCHAR(100)
);

-- 3. Projects – client projects Infosys runs
CREATE TABLE projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    client_id INTEGER REFERENCES clients(id) ON DELETE CASCADE,
    start_date DATE NOT NULL,
    end_date DATE,
    status VARCHAR(20) NOT NULL -- e.g., 'active', 'completed', 'on hold'
);

-- 4. Employee Project Assignments – who is working on what
CREATE TABLE employee_project_assignments (
    id SERIAL PRIMARY KEY,
    employee_id INTEGER REFERENCES employees(id) ON DELETE CASCADE,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    assigned_on DATE NOT NULL,
    role_on_project VARCHAR(50) NOT NULL -- e.g., 'Developer', 'Tester', 'Manager'
);

-- 5. Skills – skill catalog
CREATE TABLE skills (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL
);

-- 6. Employee Skills – each employee’s skills
CREATE TABLE employee_skills (
    employee_id INTEGER REFERENCES employees(id) ON DELETE CASCADE,
    skill_id INTEGER REFERENCES skills(id) ON DELETE CASCADE,
    PRIMARY KEY (employee_id, skill_id)
);

INSERT INTO employees (name, email, role, joined_at) VALUES
('Alice Sharma', 'alice.sharma@infosys.com', 'Software Engineer', '2021-03-01'),
('Bob Mehta', 'bob.mehta@infosys.com', 'Tester', '2022-06-15'),
('Cathy Zhang', 'cathy.zhang@infosys.com', 'Project Manager', '2020-11-20');

INSERT INTO clients (name, industry, location) VALUES
('Acme Corp', 'Retail', 'New York'),
('Globex Inc', 'Finance', 'Chicago'),
('Initech', 'Technology', 'San Francisco');

INSERT INTO projects (name, client_id, start_date, end_date, status) VALUES
('E-Commerce Platform', 1, '2023-01-01', NULL, 'active'),
('Fraud Detection System', 2, '2022-08-10', '2023-05-30', 'completed'),
('Cloud Migration', 3, '2023-03-15', NULL, 'active');

INSERT INTO employee_project_assignments (employee_id, project_id, assigned_on, role_on_project) VALUES
(1, 1, '2023-01-02', 'Developer'),
(2, 1, '2023-01-10', 'Tester'),
(3, 1, '2023-01-05', 'Manager'),
(1, 3, '2023-03-16', 'Developer'),
(3, 3, '2023-03-17', 'Manager');

INSERT INTO skills (name) VALUES
('Java'), ('Python'), ('SQL'), ('Cloud Computing'), ('Testing');

INSERT INTO employee_skills (employee_id, skill_id) VALUES
(1, 1), -- Alice: Java
(1, 3), -- Alice: SQL
(1, 4), -- Alice: Cloud Computing
(2, 5), -- Bob: Testing
(3, 4); -- Cathy: Cloud Computing