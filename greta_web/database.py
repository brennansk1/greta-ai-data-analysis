import sqlite3
import os
from datetime import datetime
import bcrypt

DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'greta.db')

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Projects table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            owner_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (owner_id) REFERENCES users (id)
        )
    ''')

    # Project permissions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS project_permissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            permission TEXT NOT NULL CHECK (permission IN ('read', 'write', 'admin')),
            granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects (id),
            FOREIGN KEY (user_id) REFERENCES users (id),
            UNIQUE (project_id, user_id)
        )
    ''')

    conn.commit()
    conn.close()

# User functions
def create_user(username, email, password):
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                       (username, email, password_hash))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def get_user_by_username(username):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    return user

def get_user_by_id(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    return user

def verify_password(username, password):
    user = get_user_by_username(username)
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
        return user
    return None

# Project functions
def create_project(name, description, owner_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO projects (name, description, owner_id) VALUES (?, ?, ?)',
                   (name, description, owner_id))
    project_id = cursor.lastrowid
    # Add owner as admin
    cursor.execute('INSERT INTO project_permissions (project_id, user_id, permission) VALUES (?, ?, ?)',
                   (project_id, owner_id, 'admin'))
    conn.commit()
    conn.close()
    return project_id

def get_user_projects(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT p.*, pp.permission
        FROM projects p
        JOIN project_permissions pp ON p.id = pp.project_id
        WHERE pp.user_id = ?
        ORDER BY p.created_at DESC
    ''', (user_id,))
    projects = cursor.fetchall()
    conn.close()
    return projects

def get_project_by_id(project_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM projects WHERE id = ?', (project_id,))
    project = cursor.fetchone()
    conn.close()
    return project

def check_user_permission(user_id, project_id, required_permission):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT permission FROM project_permissions WHERE user_id = ? AND project_id = ?',
                   (user_id, project_id))
    result = cursor.fetchone()
    conn.close()
    if result:
        permissions = {'read': 1, 'write': 2, 'admin': 3}
        return permissions.get(result['permission'], 0) >= permissions.get(required_permission, 0)
    return False

def share_project(project_id, user_id, permission):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT OR REPLACE INTO project_permissions (project_id, user_id, permission) VALUES (?, ?, ?)',
                       (project_id, user_id, permission))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

def get_project_users(project_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT u.username, pp.permission
        FROM users u
        JOIN project_permissions pp ON u.id = pp.user_id
        WHERE pp.project_id = ?
    ''', (project_id,))
    users = cursor.fetchall()
    conn.close()
    return users

# Initialize database on import
init_db()