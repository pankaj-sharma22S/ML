/**
 * AMEA Frontend API Client
 */

const API_BASE = "";

export const ProjectAPI = {
  create: async (data) => {
    const res = await fetch(`${API_BASE}/api/project/create`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },
  open: async (path) => {
    const res = await fetch(`${API_BASE}/api/project/open`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    return res.json();
  },
  getTree: async (path) => {
    const res = await fetch(`${API_BASE}/api/project/tree?path=${encodeURIComponent(path)}`);
    return res.json();
  },
  readFile: async (project_path, relative_path) => {
    const res = await fetch(`${API_BASE}/api/project/file/read`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_path, relative_path }),
    });
    return res.json();
  },
  writeFile: async (project_path, relative_path, content) => {
    const res = await fetch(`${API_BASE}/api/project/file/write`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_path, relative_path, content }),
    });
    return res.json();
  },
  createDir: async (project_path, relative_path) => {
    const res = await fetch(`${API_BASE}/api/project/file/create-dir`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_path, relative_path }),
    });
    return res.json();
  },
  deleteFile: async (project_path, relative_path) => {
    const res = await fetch(`${API_BASE}/api/project/file/delete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_path, relative_path }),
    });
    return res.json();
  },
  getDownloadZipUrl: (project_path) => {
    return `${API_BASE}/api/project/download-zip?project_path=${encodeURIComponent(project_path)}`;
  },
};

export const KernelAPI = {
  createSession: async (project_id) => {
    const res = await fetch(`${API_BASE}/api/kernel/session`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_id }),
    });
    return res.json();
  },
  getSession: async (session_id) => {
    const res = await fetch(`${API_BASE}/api/kernel/session/${session_id}`);
    return res.json();
  },
  executeCell: async (session_id, cell_id, code) => {
    const res = await fetch(`${API_BASE}/api/kernel/execute`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id, cell_id, code }),
    });
    return res.json();
  },
  interrupt: async (session_id) => {
    const res = await fetch(`${API_BASE}/api/kernel/interrupt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id }),
    });
    return res.json();
  },
  restart: async (session_id) => {
    const res = await fetch(`${API_BASE}/api/kernel/restart`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id }),
    });
    return res.json();
  },
  shutdown: async (session_id) => {
    const res = await fetch(`${API_BASE}/api/kernel/session/${session_id}`, {
      method: "DELETE",
    });
    return res.json();
  },
};

export const NotebookAPI = {
  save: async (notebook_path, cells, metadata) => {
    const res = await fetch(`${API_BASE}/api/kernel/notebook/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ notebook_path, cells, metadata }),
    });
    return res.json();
  },
  load: async (notebook_path) => {
    const res = await fetch(`${API_BASE}/api/kernel/notebook/load`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ notebook_path }),
    });
    return res.json();
  },
};

export const ThreadAPI = {
  list: async (project_id) => {
    const res = await fetch(`${API_BASE}/api/threads/list?project_id=${encodeURIComponent(project_id)}`);
    return res.json();
  },
  save: async (project_id, thread) => {
    const res = await fetch(`${API_BASE}/api/threads/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_id, thread }),
    });
    return res.json();
  },
  delete: async (project_id, thread_id) => {
    const res = await fetch(`${API_BASE}/api/threads/delete?project_id=${encodeURIComponent(project_id)}&thread_id=${encodeURIComponent(thread_id)}`, {
      method: "POST",
    });
    return res.json();
  },
};

export const AIAPI = {
  generateCell: async (prompt, active_variables) => {
    const res = await fetch(`${API_BASE}/api/kernel/ai/generate-cell`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, active_variables }),
    });
    return res.json();
  },
  interpretResult: async (result) => {
    const res = await fetch(`${API_BASE}/api/kernel/ai/interpret-result`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ result }),
    });
    return res.json();
  },
};

export const TerminalAPI = {
  exec: async (project_path, command) => {
    const res = await fetch(`${API_BASE}/api/terminal/exec`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_path, command }),
    });
    return res.json();
  },
};

export const OrchestratorAPI = {
  runTask: async (data) => {
    const res = await fetch(`${API_BASE}/api/orchestrator/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },
};

export const EnvironmentAPI = {
  getInfo: async () => {
    const res = await fetch(`${API_BASE}/api/environment/info`);
    return res.json();
  },
};
