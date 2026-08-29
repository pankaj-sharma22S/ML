"""Tests for AMEA Unified Server, Project Management, ZIP export, Terminal, and Threads APIs."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from amea.server import app

client = TestClient(app)


def test_server_serves_frontend_html():
    """Verify that root endpoint serves index.html."""
    response = client.get("/")
    assert response.status_code == 200
    assert "AMEA" in response.text
    assert "Autonomous ML Engineering IDE" in response.text


def test_create_project_initializes_requirements_and_readme(tmp_path):
    """Verify that creating a project automatically generates requirements.txt and README.md."""
    req_payload = {
        "name": "test_churn_project",
        "location": str(tmp_path),
        "template": "classification",
    }
    response = client.post("/api/project/create", json=req_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

    proj_dir = tmp_path / "test_churn_project"
    assert (proj_dir / "requirements.txt").exists()
    assert (proj_dir / "README.md").exists()
    assert (proj_dir / "src" / "train.py").exists()

    req_text = (proj_dir / "requirements.txt").read_text(encoding="utf-8")
    assert "pandas" in req_text
    assert "scikit-learn" in req_text


def test_open_existing_project_scans_and_ensures_requirements(tmp_path):
    """Verify that opening an existing directory scans it and adds requirements.txt if missing."""
    folder = tmp_path / "legacy_project"
    folder.mkdir()
    (folder / "model.py").write_text("print('hello')", encoding="utf-8")

    response = client.post("/api/project/open", json={"path": str(folder)})
    assert response.status_code == 200
    assert (folder / "requirements.txt").exists()


def test_get_project_tree(tmp_path):
    """Verify project tree traversal."""
    proj = tmp_path / "tree_proj"
    proj.mkdir()
    (proj / "src").mkdir()
    (proj / "src" / "main.py").write_text("# main", encoding="utf-8")

    response = client.get(f"/api/project/tree?path={str(proj)}")
    assert response.status_code == 200
    tree_data = response.json()
    assert tree_data["root_name"] == "tree_proj"
    assert any(n["name"] == "src" for n in tree_data["tree"])


def test_file_read_and_write_with_traversal_protection(tmp_path):
    """Verify file read/write and path traversal security."""
    proj = tmp_path / "file_proj"
    proj.mkdir()

    # 1. Write file
    write_res = client.post("/api/project/file/write", json={
        "project_path": str(proj),
        "relative_path": "src/utils.py",
        "content": "def helper(): pass",
    })
    assert write_res.status_code == 200
    assert (proj / "src" / "utils.py").exists()

    # 2. Read file
    read_res = client.post("/api/project/file/read", json={
        "project_path": str(proj),
        "relative_path": "src/utils.py",
    })
    assert read_res.status_code == 200
    assert "def helper(): pass" in read_res.json()["content"]

    # 3. Path traversal blocked
    bad_res = client.post("/api/project/file/read", json={
        "project_path": str(proj),
        "relative_path": "../../etc/passwd",
    })
    assert bad_res.status_code == 403


def test_delete_file(tmp_path):
    """Verify safe file deletion."""
    proj = tmp_path / "del_proj"
    proj.mkdir()
    f = proj / "temp.txt"
    f.write_text("trash", encoding="utf-8")

    res = client.post("/api/project/file/delete", json={
        "project_path": str(proj),
        "relative_path": "temp.txt",
    })
    assert res.status_code == 200
    assert not f.exists()


def test_download_project_zip_scrubs_secrets(tmp_path):
    """Verify project ZIP export strips sensitive files like .env and credentials."""
    import zipfile
    import io

    proj = tmp_path / "export_proj"
    proj.mkdir()
    (proj / "main.py").write_text("print(1)", encoding="utf-8")
    (proj / ".env").write_text("SECRET_KEY=12345", encoding="utf-8")

    res = client.get(f"/api/project/download-zip?project_path={str(proj)}")
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/zip"

    # Inspect zip contents
    zf = zipfile.ZipFile(io.BytesIO(res.content))
    filenames = zf.namelist()
    assert "main.py" in filenames
    assert ".env" not in filenames


def test_terminal_exec(tmp_path):
    """Verify terminal command execution within project path."""
    proj = tmp_path / "term_proj"
    proj.mkdir()

    res = client.post("/api/terminal/exec", json={
        "project_path": str(proj),
        "command": "python -c \"print('terminal_success')\"",
    })
    assert res.status_code == 200
    data = res.json()
    assert "terminal_success" in data["stdout"]
    assert data["exit_code"] == 0


def test_thread_persistence_lifecycle():
    """Verify saving, retrieving, and deleting AI conversation threads."""
    thread_data = {
        "id": "t_test_123",
        "project_id": "proj_thread_test",
        "title": "Debug Random Forest",
        "messages": [
          {"sender": "user", "text": "Fix overfitting"},
          {"sender": "ai", "text": "Set max_depth=5"}
        ],
    }

    # Save thread
    save_res = client.post("/api/threads/save", json={
        "project_id": "proj_thread_test",
        "thread": thread_data,
    })
    assert save_res.status_code == 200

    # List threads
    list_res = client.get("/api/threads/list?project_id=proj_thread_test")
    assert list_res.status_code == 200
    threads = list_res.json()
    assert any(t["id"] == "t_test_123" for t in threads)

    # Delete thread
    del_res = client.post("/api/threads/delete?project_id=proj_thread_test&thread_id=t_test_123")
    assert del_res.status_code == 200


def test_orchestrator_run_endpoint(tmp_path):
    """Verify real multi-agent execution pipeline through backend REST API."""
    req_payload = {
        "project_id": "test_api_orch",
        "user_request": "Train a baseline classifier for customer churn prediction",
        "dataset_path": "data/sample_churn.csv",
        "target_column": "churn",
        "max_experiments": 2,
    }
    response = client.post("/api/orchestrator/run", json=req_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["terminal_phase"] == "TERMINATED"
    assert data["best_candidate"] is not None
    assert len(data["generated_files"]) > 0
    assert "train.py" in data["generated_files"]


def test_environment_info_endpoint():
    """Verify Python environment inspection endpoint."""
    response = client.get("/api/environment/info")
    assert response.status_code == 200
    data = response.json()
    assert "python_version" in data
    assert "executable" in data
    assert "scikit-learn" in data["packages"]

