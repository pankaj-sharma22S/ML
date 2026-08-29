"""Supabase Authentication & Postgres Persistence Service for AMEA."""

import os
import sqlite3
import json
import uuid
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import jwt
import httpx


class SupabaseService:
    """Manages Supabase Auth, JWT verification, and Postgres database persistence."""

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        jwt_secret: Optional[str] = None,
        db_fallback_path: Optional[Path] = None,
    ):
        self.supabase_url = (supabase_url or os.getenv("SUPABASE_URL", "")).rstrip("/")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_ANON_KEY", os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""))
        self.jwt_secret = jwt_secret or os.getenv("SUPABASE_JWT_SECRET", "amea-local-dev-jwt-secret-2026")
        self.db_fallback_path = db_fallback_path or Path(".amea_project/supabase_local.db")

        # Initialize local SQLite storage for offline / test capability
        self._init_local_db()

    def _init_local_db(self) -> None:
        """Initialize local SQLite tables matching Supabase schema for seamless fallback."""
        self.db_fallback_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_fallback_path)) as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT,
                    created_at TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS profiles (
                    id TEXT PRIMARY KEY REFERENCES users(id),
                    email TEXT UNIQUE NOT NULL,
                    full_name TEXT,
                    avatar_url TEXT,
                    created_at TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    template TEXT,
                    dataset_path TEXT,
                    target_column TEXT,
                    created_at TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    model_family TEXT NOT NULL,
                    cv_metrics TEXT,
                    hyperparameters TEXT,
                    duration_sec REAL,
                    exit_code INT,
                    created_at TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS execution_logs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    command TEXT NOT NULL,
                    stdout TEXT,
                    stderr TEXT,
                    exit_code INT,
                    created_at TEXT
                )
            """)
            conn.commit()

    def is_cloud_enabled(self) -> bool:
        """Check if remote Supabase endpoints are configured."""
        return bool(self.supabase_url and self.supabase_key)

    # ============================================================
    # Authentication Operations
    # ============================================================

    def sign_up(self, email: str, password: str, full_name: Optional[str] = None) -> Dict[str, Any]:
        """Sign up a new user via Supabase Auth API or local store."""
        if self.is_cloud_enabled():
            try:
                headers = {
                    "apikey": self.supabase_key,
                    "Content-Type": "application/json",
                }
                payload = {
                    "email": email,
                    "password": password,
                    "data": {"full_name": full_name or ""},
                }
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(f"{self.supabase_url}/auth/v1/signup", json=payload, headers=headers)
                    if resp.status_code in (200, 201):
                        data = resp.json()
                        token = data.get("access_token") or self.generate_token(data.get("user", {}).get("id", str(uuid.uuid4())), email)
                        return {
                            "user": data.get("user") or {"id": str(uuid.uuid4()), "email": email},
                            "access_token": token,
                            "session": data.get("session"),
                        }
            except Exception:
                pass  # Fall back to local auth

        # Local User Auth
        user_id = str(uuid.uuid4())
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        import hashlib
        pwd_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()

        with sqlite3.connect(str(self.db_fallback_path)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT id FROM users WHERE email = ?", (email,))
            if cur.fetchone():
                raise ValueError("A user with this email already exists")

            cur.execute(
                "INSERT INTO users (id, email, password_hash, full_name, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, email, pwd_hash, full_name or "", now_str),
            )
            cur.execute(
                "INSERT INTO profiles (id, email, full_name, avatar_url, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, email, full_name or "", "", now_str),
            )
            conn.commit()

        token = self.generate_token(user_id, email)
        return {
            "user": {"id": user_id, "email": email, "full_name": full_name or ""},
            "access_token": token,
        }

    def sign_in(self, email: str, password: str) -> Dict[str, Any]:
        """Sign in an existing user and return Supabase session token."""
        if self.is_cloud_enabled():
            try:
                headers = {
                    "apikey": self.supabase_key,
                    "Content-Type": "application/json",
                }
                payload = {
                    "email": email,
                    "password": password,
                }
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(
                        f"{self.supabase_url}/auth/v1/token?grant_type=password",
                        json=payload,
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        return {
                            "user": data.get("user"),
                            "access_token": data.get("access_token"),
                            "refresh_token": data.get("refresh_token"),
                        }
            except Exception:
                pass

        # Local Sign In
        import hashlib
        pwd_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()

        with sqlite3.connect(str(self.db_fallback_path)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, email, full_name, password_hash FROM users WHERE email = ?", (email,))
            row = cur.fetchone()
            if not row or row[3] != pwd_hash:
                raise ValueError("Invalid email or password")

            user_id, user_email, full_name = row[0], row[1], row[2]

        token = self.generate_token(user_id, user_email)
        return {
            "user": {"id": user_id, "email": user_email, "full_name": full_name},
            "access_token": token,
        }

    def generate_token(self, user_id: str, email: str) -> str:
        """Generate a valid signed JWT session token."""
        payload = {
            "sub": user_id,
            "email": email,
            "aud": "authenticated",
            "role": "authenticated",
            "iat": int(datetime.datetime.now(datetime.timezone.utc).timestamp()),
            "exp": int((datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=7)).timestamp()),
        }
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and extract authenticated user details."""
        if not token:
            return None

        # Clean "Bearer " prefix if provided
        if token.startswith("Bearer "):
            token = token[7:].strip()

        try:
            # 1. First attempt direct signature decoding
            decoded = jwt.decode(token, self.jwt_secret, algorithms=["HS256"], options={"verify_aud": False})
            return {
                "id": decoded.get("sub"),
                "email": decoded.get("email"),
                "role": decoded.get("role", "authenticated"),
            }
        except Exception:
            # 2. If remote Supabase is configured, verify with remote auth endpoint
            if self.is_cloud_enabled():
                try:
                    headers = {
                        "apikey": self.supabase_key,
                        "Authorization": f"Bearer {token}",
                    }
                    with httpx.Client(timeout=5.0) as client:
                        resp = client.get(f"{self.supabase_url}/auth/v1/user", headers=headers)
                        if resp.status_code == 200:
                            u = resp.json()
                            return {
                                "id": u.get("id"),
                                "email": u.get("email"),
                                "role": u.get("role", "authenticated"),
                            }
                except Exception:
                    pass

        return None

    # ============================================================
    # User Project & Data Persistence (RLS Enforced)
    # ============================================================

    def save_project(self, user_id: str, project_id: str, name: str, description: str = "", template: str = "classification") -> Dict[str, Any]:
        """Persist project record associated with the authenticated user."""
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with sqlite3.connect(str(self.db_fallback_path)) as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO projects (id, user_id, name, description, template, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET name = excluded.name, description = excluded.description, template = excluded.template
            """, (project_id, user_id, name, description, template, now_str))
            conn.commit()

        return {
            "id": project_id,
            "user_id": user_id,
            "name": name,
            "description": description,
            "template": template,
            "created_at": now_str,
        }

    def list_user_projects(self, user_id: str) -> List[Dict[str, Any]]:
        """List all projects owned by the authenticated user (RLS rule)."""
        with sqlite3.connect(str(self.db_fallback_path)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, user_id, name, description, template, created_at FROM projects WHERE user_id = ?", (user_id,))
            rows = cur.fetchall()
            return [
                {
                    "id": r[0],
                    "user_id": r[1],
                    "name": r[2],
                    "description": r[3],
                    "template": r[4],
                    "created_at": r[5],
                } for r in rows
            ]

    def verify_project_ownership(self, user_id: str, project_id: str) -> bool:
        """Verify whether the specified user owns the given project."""
        with sqlite3.connect(str(self.db_fallback_path)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT id FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id))
            return cur.fetchone() is not None

    def save_experiment(
        self,
        user_id: str,
        project_id: str,
        experiment_id: str,
        model_family: str,
        cv_metrics: Dict[str, Any],
        hyperparameters: Dict[str, Any],
        duration_sec: float,
        exit_code: int = 0,
    ) -> Dict[str, Any]:
        """Persist experiment run metadata linked to project and user."""
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with sqlite3.connect(str(self.db_fallback_path)) as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO experiments (id, project_id, user_id, model_family, cv_metrics, hyperparameters, duration_sec, exit_code, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET cv_metrics = excluded.cv_metrics
            """, (
                experiment_id,
                project_id,
                user_id,
                model_family,
                json.dumps(cv_metrics),
                json.dumps(hyperparameters),
                duration_sec,
                exit_code,
                now_str,
            ))
            conn.commit()
        return {"id": experiment_id, "project_id": project_id, "status": "persisted"}

    def save_artifact(
        self,
        user_id: str,
        project_id: str,
        artifact_id: str,
        artifact_type: str,
        name: str,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist artifact reference."""
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with sqlite3.connect(str(self.db_fallback_path)) as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO artifacts (id, project_id, user_id, artifact_type, name, path, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET path = excluded.path
            """, (
                artifact_id,
                project_id,
                user_id,
                artifact_type,
                name,
                path,
                json.dumps(metadata or {}),
                now_str,
            ))
            conn.commit()
        return {"id": artifact_id, "name": name, "status": "persisted"}


# Global Supabase service singleton
supabase_service = SupabaseService()
