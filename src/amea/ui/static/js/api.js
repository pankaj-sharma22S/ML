/**
 * AMEA Frontend API Client with Supabase Auth Support
 * Compatible with standard browser script tags and global window scope.
 */

const API_BASE = "";

// Auth token storage in localStorage
const TokenStorage = {
  get: () => localStorage.getItem("amea_auth_token"),
  set: (token) => localStorage.setItem("amea_auth_token", token),
  clear: () => localStorage.removeItem("amea_auth_token"),
};

const getHeaders = (customHeaders = {}) => {
  const headers = { "Content-Type": "application/json", ...customHeaders };
  const token = TokenStorage.get();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  return headers;
};

const AuthAPI = {
  signup: async (email, password, full_name = "") => {
    const res = await fetch(`${API_BASE}/api/auth/signup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, full_name }),
    });
    const data = await res.json();
    if (data.access_token) {
      TokenStorage.set(data.access_token);
    }
    return data;
  },
  login: async (email, password) => {
    const res = await fetch(`${API_BASE}/api/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (data.access_token) {
      TokenStorage.set(data.access_token);
    }
    return data;
  },
  me: async () => {
    const res = await fetch(`${API_BASE}/api/auth/me`, {
      headers: getHeaders(),
    });
    return res.json();
  },
  logout: () => {
    TokenStorage.clear();
  },
  publicChat: async (message) => {
    const res = await fetch(`${API_BASE}/api/public/chat`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({ message }),
    });
    return res.json();
  },
};

const ProjectAPI = {
  create: async (data) => {
    const res = await fetch(`${API_BASE}/api/project/create`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify(data),
    });
    return res.json();
  },
  open: async (path) => {
    const res = await fetch(`${API_BASE}/api/project/open`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({ path }),
    });
    return res.json();
  },
  getTree: async (path = "") => {
    const res = await fetch(`${API_BASE}/api/project/tree?path=${encodeURIComponent(path)}`, {
      headers: getHeaders(),
    });
    return res.json();
  },
  readFile: async (projectPath, relativePath) => {
    const res = await fetch(
      `${API_BASE}/api/project/file?project_path=${encodeURIComponent(projectPath)}&relative_path=${encodeURIComponent(relativePath)}`,
      { headers: getHeaders() }
    );
    return res.json();
  },
  writeFile: async (projectPath, relativePath, content) => {
    const res = await fetch(`${API_BASE}/api/project/file`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({
        project_path: projectPath,
        relative_path: relativePath,
        content: content,
      }),
    });
    return res.json();
  },
  deleteFile: async (projectPath, relativePath) => {
    const res = await fetch(
      `${API_BASE}/api/project/file?project_path=${encodeURIComponent(projectPath)}&relative_path=${encodeURIComponent(relativePath)}`,
      {
        method: "DELETE",
        headers: getHeaders(),
      }
    );
    return res.json();
  },
  downloadZip: (projectPath) => {
    window.location.href = `${API_BASE}/api/project/download-zip?project_path=${encodeURIComponent(projectPath)}`;
  },
};

const KernelAPI = {
  createSession: async (projectId) => {
    const res = await fetch(`${API_BASE}/api/kernel/session`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({ project_id: projectId }),
    });
    return res.json();
  },
  executeCell: async (sessionId, cellId, code) => {
    const res = await fetch(`${API_BASE}/api/kernel/execute-cell`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({
        session_id: sessionId,
        cell_id: cellId,
        code: code,
      }),
    });
    return res.json();
  },
  restart: async (sessionId) => {
    const res = await fetch(`${API_BASE}/api/kernel/session/${sessionId}/restart`, {
      method: "POST",
      headers: getHeaders(),
    });
    return res.json();
  },
  shutdown: async (sessionId) => {
    const res = await fetch(`${API_BASE}/api/kernel/session/${sessionId}`, {
      method: "DELETE",
      headers: getHeaders(),
    });
    return res.json();
  },
  getStatus: async (sessionId) => {
    const res = await fetch(`${API_BASE}/api/kernel/session/${sessionId}/status`, {
      headers: getHeaders(),
    });
    return res.json();
  },
};

const NotebookAPI = {
  save: async (notebookPath, cells) => {
    const res = await fetch(`${API_BASE}/api/notebook/save`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({
        notebook_path: notebookPath,
        cells: cells,
      }),
    });
    return res.json();
  },
  load: async (notebookPath) => {
    const res = await fetch(`${API_BASE}/api/notebook/load?path=${encodeURIComponent(notebookPath)}`, {
      headers: getHeaders(),
    });
    return res.json();
  },
};

const ThreadAPI = {
  create: async (title, initialMessage = null) => {
    const res = await fetch(`${API_BASE}/api/threads/create`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({
        title: title,
        initial_message: initialMessage,
      }),
    });
    return res.json();
  },
  list: async () => {
    const res = await fetch(`${API_BASE}/api/threads/list`, {
      headers: getHeaders(),
    });
    return res.json();
  },
  get: async (threadId) => {
    const res = await fetch(`${API_BASE}/api/threads/${threadId}`, {
      headers: getHeaders(),
    });
    return res.json();
  },
  addMessage: async (threadId, sender, text, structuredOutput = null) => {
    const res = await fetch(`${API_BASE}/api/threads/${threadId}/messages`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({
        sender: sender,
        text: text,
        structured_output: structuredOutput,
      }),
    });
    return res.json();
  },
};

const AIAPI = {
  generateCell: async (prompt, contextCode = "") => {
    const res = await fetch(`${API_BASE}/api/kernel/ai/generate-cell`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({
        prompt: prompt,
        context_code: contextCode,
      }),
    });
    return res.json();
  },
  interpretOutput: async (code, outputs) => {
    const res = await fetch(`${API_BASE}/api/kernel/ai/interpret`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({
        code: code,
        outputs: outputs,
      }),
    });
    return res.json();
  },
};

const TerminalAPI = {
  exec: async (projectPath, command) => {
    const res = await fetch(`${API_BASE}/api/terminal/exec`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({
        project_path: projectPath,
        command: command,
      }),
    });
    return res.json();
  },
};

const OrchestratorAPI = {
  runTask: async (data) => {
    const res = await fetch(`${API_BASE}/api/orchestrator/run`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify(data),
    });
    return res.json();
  },
};

const EnvironmentAPI = {
  getInfo: async () => {
    const res = await fetch(`${API_BASE}/api/environment/info`, {
      headers: getHeaders(),
    });
    return res.json();
  },
};

const DatasetAPI = {
  upload: async (file, project_path = "") => {
    const formData = new FormData();
    formData.append("file", file);
    if (project_path) {
      formData.append("project_path", project_path);
    }
    const token = TokenStorage.get();
    const headers = {};
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    const res = await fetch(`${API_BASE}/api/project/upload-dataset`, {
      method: "POST",
      headers: headers,
      body: formData,
    });
    return res.json();
  },
};

const LLMAPI = {
  getStatus: async () => {
    const res = await fetch(`${API_BASE}/api/llm/status`, {
      headers: getHeaders(),
    });
    return res.json();
  },
};

// Expose globally to window
if (typeof window !== "undefined") {
  window.TokenStorage = TokenStorage;
  window.AuthAPI = AuthAPI;
  window.ProjectAPI = ProjectAPI;
  window.KernelAPI = KernelAPI;
  window.NotebookAPI = NotebookAPI;
  window.ThreadAPI = ThreadAPI;
  window.AIAPI = AIAPI;
  window.TerminalAPI = TerminalAPI;
  window.OrchestratorAPI = OrchestratorAPI;
  window.EnvironmentAPI = EnvironmentAPI;
  window.DatasetAPI = DatasetAPI;
  window.LLMAPI = LLMAPI;
  window.AMEA_API = {
    TokenStorage,
    AuthAPI,
    ProjectAPI,
    KernelAPI,
    NotebookAPI,
    ThreadAPI,
    AIAPI,
    TerminalAPI,
    OrchestratorAPI,
    EnvironmentAPI,
    DatasetAPI,
    LLMAPI,
  };
}
