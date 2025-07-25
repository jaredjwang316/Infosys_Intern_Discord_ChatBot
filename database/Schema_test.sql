-- 1. Employees – Infosys staff
CREATE TABLE IF NOT EXISTS employees (
    id        INT AUTO_INCREMENT PRIMARY KEY,
    name      VARCHAR(100) NOT NULL,
    email     VARCHAR(100) UNIQUE NOT NULL,
    role      VARCHAR(50)  NOT NULL,
    joined_at DATE          NOT NULL
);

-- 2. Clients – external clients Infosys serves
CREATE TABLE IF NOT EXISTS clients (
    id       INT AUTO_INCREMENT PRIMARY KEY,
    name     VARCHAR(100) NOT NULL,
    industry VARCHAR(50)  NOT NULL,
    location VARCHAR(100)
);

-- 3. Projects – client projects Infosys runs
CREATE TABLE IF NOT EXISTS projects (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    name       VARCHAR(100) NOT NULL,
    client_id  INT           NOT NULL,
    start_date DATE          NOT NULL,
    end_date   DATE,
    status     VARCHAR(20)   NOT NULL,
    FOREIGN KEY (client_id) REFERENCES clients(id) ON DELETE CASCADE
);

-- 4. Employee Project Assignments
CREATE TABLE IF NOT EXISTS employee_project_assignments (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    employee_id      INT           NOT NULL,
    project_id       INT           NOT NULL,
    assigned_on      DATE          NOT NULL,
    role_on_project  VARCHAR(50)   NOT NULL,
    FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE,
    FOREIGN KEY (project_id)  REFERENCES projects(id)  ON DELETE CASCADE
);

-- 5. Skills – skill catalog
CREATE TABLE IF NOT EXISTS skills (
    id   INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) NOT NULL
);

-- 6. Employee Skills
CREATE TABLE IF NOT EXISTS employee_skills (
    employee_id INT NOT NULL,
    skill_id    INT NOT NULL,
    PRIMARY KEY (employee_id, skill_id),
    FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE,
    FOREIGN KEY (skill_id)    REFERENCES skills(id)    ON DELETE CASCADE
);