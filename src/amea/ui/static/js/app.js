import { ProjectAPI, KernelAPI, NotebookAPI, ThreadAPI, AIAPI, TerminalAPI } from "/static/js/api.js";

const { useState, useEffect, useRef } = React;

// ============================================================
// Root Application Component
// ============================================================

export default function App() {
  // Project State
  const [currentProject, setCurrentProject] = useState(null); // { name, path }
  const [fileTree, setFileTree] = useState([]);
  const [openFiles, setOpenFiles] = useState([]); // [{ path, name, content, isDirty }]
  const [activeFile, setActiveFile] = useState(null);
  
  // Workspace Mode (CODE | NOTEBOOK | GRAPH)
  const [workspaceMode, setWorkspaceMode] = useState("NOTEBOOK");

  // Notebook State
  const [notebookCells, setNotebookCells] = useState([
    {
      id: "c_1",
      type: "CODE",
      code: "import pandas as pd\nimport numpy as np\n\n# 1. Load sample dataset\ndf = pd.DataFrame({\n    'age': [21, 25, 30, 45, 52, 28],\n    'salary': [35000, 52000, 68000, 95000, 120000, 48000],\n    'department': ['IT', 'HR', 'Finance', 'Exec', 'Exec', 'IT'],\n    'churn': [0, 1, 0, 0, 1, 0]\n})\ndf.head()",
      status: "IDLE",
      output: null,
      execCount: null,
    },
    {
      id: "c_2",
      type: "CODE",
      code: "df.describe()",
      status: "IDLE",
      output: null,
      execCount: null,
    },
    {
      id: "c_3",
      type: "CODE",
      code: "import matplotlib.pyplot as plt\n\nplt.figure(figsize=(7, 3.5))\nplt.bar(df['department'], df['salary'], color='#38bdf8')\nplt.title('Salary by Department')\nplt.ylabel('Salary ($)')\nplt.show()",
      status: "IDLE",
      output: null,
      execCount: null,
    }
  ]);
  const [activeCellId, setActiveCellId] = useState("c_1");

  // Kernel State
  const [kernelSession, setKernelSession] = useState(null);
  const [kernelStats, setKernelStats] = useState({ status: "IDLE", memory_mb: 180, cpu_percent: 2.1, executions: 0 });

  // AI Threads State
  const [threads, setThreads] = useState([]);
  const [activeThread, setActiveThread] = useState(null);
  const [aiInput, setAiInput] = useState("");
  const [isAiStreaming, setIsAiStreaming] = useState(false);
  const [suggestedDiff, setSuggestedDiff] = useState(null);

  // Bottom Panel State (TERMINAL | OUTPUT | PROBLEMS | KERNEL | ARTIFACTS)
  const [bottomTab, setBottomTab] = useState("TERMINAL");
  const [terminalHistory, setTerminalHistory] = useState([
    { type: "stdout", text: "AMEA Interactive ML Environment initialized.\nPython 3.11.9 (main, Apr 2026)\nType 'help', 'copyright', 'credits' or 'license' for more info." }
  ]);
  const [terminalInput, setTerminalInput] = useState("");

  // Modals & Panels Visibility
  const [showNewProjectModal, setShowNewProjectModal] = useState(false);
  const [showDownloadModal, setShowDownloadModal] = useState(false);
  const [showDiffModal, setShowDiffModal] = useState(false);
  const [isExplorerOpen, setIsExplorerOpen] = useState(true);
  const [isThreadsOpen, setIsThreadsOpen] = useState(true);
  const [isBottomOpen, setIsBottomOpen] = useState(true);

  // Layout Dimensions
  const [explorerWidth, setExplorerWidth] = useState(240);
  const [threadsWidth, setThreadsWidth] = useState(360);
  const [bottomHeight, setBottomHeight] = useState(220);

  // Initialize Kernel on project start
  useEffect(() => {
    if (currentProject) {
      initProjectKernel();
      loadProjectTree();
      loadThreads();
    }
  }, [currentProject]);

  const initProjectKernel = async () => {
    try {
      const sess = await KernelAPI.createSession(currentProject.name);
      setKernelSession(sess);
      setKernelStats({
        status: sess.status || "IDLE",
        memory_mb: 210,
        cpu_percent: 1.5,
        executions: sess.execution_count || 0
      });
    } catch (e) {
      console.error("Kernel init failed", e);
    }
  };

  const loadProjectTree = async () => {
    try {
      const res = await ProjectAPI.getTree(currentProject.path);
      setFileTree(res.tree || []);
    } catch (e) {
      console.error("Tree load error", e);
    }
  };

  const loadThreads = async () => {
    try {
      const list = await ThreadAPI.list(currentProject.name);
      if (list && list.length > 0) {
        setThreads(list);
        setActiveThread(list[0]);
      } else {
        const defaultThread = {
          id: "thread_default",
          project_id: currentProject.name,
          title: "Data Exploration & EDA",
          messages: [
            {
              id: "msg_1",
              sender: "ai",
              text: `Welcome to **${currentProject.name}**!\n\nI am your Autonomous ML Engineering Partner. I can inspect datasets, draft validation strategies, write clean pipelines, debug models, and interpret outputs. What would you like to explore first?`
            }
          ]
        };
        setThreads([defaultThread]);
        setActiveThread(defaultThread);
        await ThreadAPI.save(currentProject.name, defaultThread);
      }
    } catch (e) {
      console.error("Threads load error", e);
    }
  };

  // Cell Execution Handling
  const runCell = async (cellId) => {
    const cell = notebookCells.find(c => c.id === cellId);
    if (!cell || cell.type !== "CODE") return;

    setNotebookCells(prev => prev.map(c => c.id === cellId ? { ...c, status: "RUNNING" } : c));

    try {
      const res = await KernelAPI.executeCell(kernelSession ? kernelSession.session_id : "default", cellId, cell.code);
      
      setNotebookCells(prev => prev.map(c => {
        if (c.id === cellId) {
          return {
            ...c,
            status: res.is_success ? "SUCCESS" : "ERROR",
            output: res.outputs || [],
            execCount: res.execution_count || (c.execCount ? c.execCount + 1 : 1),
          };
        }
        return c;
      }));

      setKernelStats(prev => ({
        ...prev,
        executions: prev.executions + 1,
        status: "IDLE"
      }));
    } catch (e) {
      setNotebookCells(prev => prev.map(c => {
        if (c.id === cellId) {
          return {
            ...c,
            status: "ERROR",
            output: [{ output_type: "ERROR", error_name: "ExecutionError", error_value: e.message }],
          };
        }
        return c;
      }));
    }
  };

  const runAllCells = async () => {
    for (const cell of notebookCells) {
      if (cell.type === "CODE") {
        await runCell(cell.id);
      }
    }
  };

  const runFromHere = async (startCellId) => {
    let start = false;
    for (const cell of notebookCells) {
      if (cell.id === startCellId) start = true;
      if (start && cell.type === "CODE") {
        await runCell(cell.id);
      }
    }
  };

  const addCell = (afterId, type = "CODE") => {
    const newCell = {
      id: "c_" + Date.now().toString(36),
      type: type,
      code: type === "CODE" ? "# Write Python code...\n" : "## New Markdown Section\n",
      status: "IDLE",
      output: null,
      execCount: null,
    };
    const index = notebookCells.findIndex(c => c.id === afterId);
    if (index === -1) {
      setNotebookCells(prev => [...prev, newCell]);
    } else {
      const updated = [...notebookCells];
      updated.splice(index + 1, 0, newCell);
      setNotebookCells(updated);
    }
    setActiveCellId(newCell.id);
  };

  const deleteCell = (cellId) => {
    if (notebookCells.length <= 1) return;
    setNotebookCells(prev => prev.filter(c => c.id !== cellId));
  };

  // AI Code Generation & Interpretation
  const handleAiSend = async () => {
    if (!aiInput.trim() || isAiStreaming) return;
    const userText = aiInput;
    setAiInput("");

    const newMsg = {
      id: "msg_" + Date.now(),
      sender: "user",
      text: userText,
      timestamp: new Date().toISOString()
    };

    const updatedMessages = [...(activeThread.messages || []), newMsg];
    const updatedThread = { ...activeThread, messages: updatedMessages };
    setActiveThread(updatedThread);
    setIsAiStreaming(true);

    try {
      const suggestion = await AIAPI.generateCell(userText, ["df"]);
      
      const aiResponseMsg = {
        id: "msg_ai_" + Date.now(),
        sender: "ai",
        text: `I've prepared the analysis for: **"${userText}"**.\n\n${suggestion.explanation}`,
        code_diff: {
          filename: "notebook_cell",
          code: suggestion.code,
          is_safe: suggestion.is_safe
        },
        timestamp: new Date().toISOString()
      };

      const finalThread = { ...updatedThread, messages: [...updatedMessages, aiResponseMsg] };
      setActiveThread(finalThread);
      await ThreadAPI.save(currentProject.name, finalThread);
    } catch (e) {
      console.error(e);
    } finally {
      setIsAiStreaming(false);
    }
  };

  const applyAiCodeToNotebook = (code) => {
    const newCell = {
      id: "c_ai_" + Date.now().toString(36),
      type: "CODE",
      code: code,
      status: "IDLE",
      output: null,
      execCount: null,
    };
    setNotebookCells(prev => [...prev, newCell]);
    setActiveCellId(newCell.id);
    setWorkspaceMode("NOTEBOOK");
  };

  // Terminal Execution
  const handleTerminalSubmit = async (e) => {
    e.preventDefault();
    if (!terminalInput.trim()) return;
    const cmd = terminalInput;
    setTerminalInput("");

    setTerminalHistory(prev => [...prev, { type: "stdin", text: `$ ${cmd}` }]);

    try {
      const res = await TerminalAPI.exec(currentProject.path, cmd);
      if (res.stdout) {
        setTerminalHistory(prev => [...prev, { type: "stdout", text: res.stdout }]);
      }
      if (res.stderr) {
        setTerminalHistory(prev => [...prev, { type: "stderr", text: res.stderr }]);
      }
    } catch (e) {
      setTerminalHistory(prev => [...prev, { type: "stderr", text: e.message }]);
    }
  };

  // -------------------------------------------------------------
  // Render Project Launcher if no project is active
  // -------------------------------------------------------------
  if (!currentProject) {
    return (
      <LauncherScreen
        onOpenNew={() => setShowNewProjectModal(true)}
        onOpenExisting={async (dirPath) => {
          const res = await ProjectAPI.open(dirPath);
          setCurrentProject({ name: res.project_name, path: res.project_path });
        }}
        showNewModal={showNewProjectModal}
        setShowNewModal={setShowNewProjectModal}
        onCreateProject={async (data) => {
          const res = await ProjectAPI.create(data);
          setCurrentProject({ name: res.project_name, path: res.project_path });
          setShowNewProjectModal(false);
        }}
      />
    );
  }

  // -------------------------------------------------------------
  // Render Main Engineering IDE Workspace
  // -------------------------------------------------------------
  return (
    <div className="flex flex-col h-screen w-screen bg-slate-950 text-slate-100 overflow-hidden font-sans select-none">
      
      {/* 1. TOP NAVIGATION BAR */}
      <header className="h-11 bg-slate-900 border-b border-slate-800 px-3 flex items-center justify-between text-xs shrink-0 z-20">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-1.5 font-bold text-sky-400">
            <span className="w-2.5 h-2.5 rounded-full bg-sky-400 animate-pulse"></span>
            <span className="tracking-wide text-sm font-mono">AMEA</span>
          </div>
          <span className="text-slate-600">/</span>
          <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-200 font-medium font-mono text-[11px] border border-slate-700">
            {currentProject.name}
          </span>
        </div>

        {/* Center Workspace Mode Tabs */}
        <div className="flex items-center bg-slate-950 p-0.5 rounded-lg border border-slate-800">
          {["CODE", "NOTEBOOK", "GRAPH"].map((mode) => (
            <button
              key={mode}
              onClick={() => setWorkspaceMode(mode)}
              className={`px-3 py-1 rounded text-[11px] font-semibold transition-all ${
                workspaceMode === mode
                  ? "bg-sky-500 text-slate-950 shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {mode}
            </button>
          ))}
        </div>

        {/* Right Controls */}
        <div className="flex items-center space-x-3">
          {/* Run All Button */}
          <button
            onClick={runAllCells}
            className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold transition-all"
            title="Run All Cells (Ctrl+Shift+Enter)"
          >
            <span>▶</span>
            <span>Run All</span>
          </button>

          {/* Kernel Status Badge */}
          <div className="flex items-center space-x-2 px-2 py-1 rounded bg-slate-800 border border-slate-700 text-[11px]">
            <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
            <span className="font-mono text-slate-300">Python 3.11</span>
            <span className="text-slate-500 font-mono">|</span>
            <span className="text-slate-400">{kernelStats.memory_mb} MB</span>
          </div>

          {/* Download Project ZIP */}
          <button
            onClick={() => setShowDownloadModal(true)}
            className="flex items-center space-x-1 px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-all"
          >
            <span>↓</span>
            <span>Download</span>
          </button>
        </div>
      </header>

      {/* 2. THREE-PANEL RESIZABLE WORKSPACE */}
      <div className="flex-1 flex overflow-hidden relative">
        
        {/* LEFT: File Explorer */}
        {isExplorerOpen && (
          <aside style={{ width: explorerWidth }} className="bg-slate-900 border-r border-slate-800 flex flex-col shrink-0">
            <div className="h-8 px-3 border-b border-slate-800 flex items-center justify-between text-[11px] font-bold text-slate-400 tracking-wider">
              <span>EXPLORER</span>
              <button onClick={loadProjectTree} className="hover:text-slate-200">↻</button>
            </div>
            <div className="flex-1 overflow-y-auto p-2 text-xs font-mono">
              <FileTreeNode
                tree={fileTree}
                onSelectFile={(path) => {
                  const name = path.split("/").pop();
                  if (!openFiles.some(f => f.path === path)) {
                    setOpenFiles(prev => [...prev, { path, name, content: "" }]);
                  }
                  setActiveFile(path);
                  setWorkspaceMode("CODE");
                }}
              />
            </div>
          </aside>
        )}

        {/* CENTER: Main Editor & Notebook Area */}
        <main className="flex-1 flex flex-col bg-slate-950 overflow-hidden relative">
          
          {workspaceMode === "NOTEBOOK" && (
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              <div className="max-w-4xl mx-auto space-y-4">
                {notebookCells.map((cell, idx) => (
                  <NotebookCellView
                    key={cell.id}
                    index={idx + 1}
                    cell={cell}
                    isActive={activeCellId === cell.id}
                    onFocus={() => setActiveCellId(cell.id)}
                    onChangeCode={(val) => {
                      setNotebookCells(prev => prev.map(c => c.id === cell.id ? { ...c, code: val } : c));
                    }}
                    onRun={() => runCell(cell.id)}
                    onRunFromHere={() => runFromHere(cell.id)}
                    onDelete={() => deleteCell(cell.id)}
                    onAddCell={(type) => addCell(cell.id, type)}
                  />
                ))}

                <div className="pt-4 flex justify-center space-x-3">
                  <button
                    onClick={() => addCell(null, "CODE")}
                    className="px-4 py-1.5 rounded-lg border border-slate-800 hover:border-sky-500 bg-slate-900 text-xs font-semibold text-sky-400 hover:text-sky-300 transition-all flex items-center space-x-1.5"
                  >
                    <span>+</span>
                    <span>Code Cell</span>
                  </button>
                  <button
                    onClick={() => addCell(null, "MARKDOWN")}
                    className="px-4 py-1.5 rounded-lg border border-slate-800 hover:border-slate-700 bg-slate-900 text-xs font-semibold text-slate-400 hover:text-slate-200 transition-all flex items-center space-x-1.5"
                  >
                    <span>+</span>
                    <span>Markdown</span>
                  </button>
                </div>
              </div>
            </div>
          )}

          {workspaceMode === "CODE" && (
            <div className="flex-1 flex flex-col">
              {/* Tab Header Bar */}
              <div className="h-9 bg-slate-900 border-b border-slate-800 flex items-center px-2 space-x-1 overflow-x-auto text-xs">
                {openFiles.map(f => (
                  <div
                    key={f.path}
                    onClick={() => setActiveFile(f.path)}
                    className={`flex items-center space-x-2 px-3 py-1.5 rounded-t border-t-2 text-xs font-mono cursor-pointer transition-all ${
                      activeFile === f.path
                        ? "bg-slate-950 border-sky-400 text-slate-100"
                        : "border-transparent text-slate-400 hover:bg-slate-800"
                    }`}
                  >
                    <span>{f.name}</span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setOpenFiles(prev => prev.filter(x => x.path !== f.path));
                      }}
                      className="text-slate-500 hover:text-slate-300"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>

              {/* Code Editor Body */}
              <div className="flex-1 bg-slate-950 p-4 font-mono text-sm overflow-auto">
                <textarea
                  className="w-full h-full bg-transparent resize-none outline-none text-slate-200 font-mono text-xs leading-relaxed"
                  value={openFiles.find(f => f.path === activeFile)?.content || "# Select a file from Explorer to edit"}
                  onChange={(e) => {
                    const val = e.target.value;
                    setOpenFiles(prev => prev.map(f => f.path === activeFile ? { ...f, content: val } : f));
                  }}
                />
              </div>
            </div>
          )}

          {workspaceMode === "GRAPH" && (
            <WorkflowGraphView onRunNode={(nodeId) => runCell(notebookCells[0]?.id)} />
          )}

          {/* 3. BOTTOM PANEL DOCK (Terminal, Kernel, Artifacts) */}
          {isBottomOpen && (
            <section style={{ height: bottomHeight }} className="bg-slate-900 border-t border-slate-800 flex flex-col shrink-0">
              <div className="h-8 border-b border-slate-800 flex items-center justify-between px-3 text-xs">
                <div className="flex space-x-4">
                  {["TERMINAL", "OUTPUT", "PROBLEMS", "KERNEL", "ARTIFACTS"].map((tab) => (
                    <button
                      key={tab}
                      onClick={() => setBottomTab(tab)}
                      className={`text-[11px] font-bold tracking-wide transition-all ${
                        bottomTab === tab ? "text-sky-400 border-b-2 border-sky-400 pb-1" : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      {tab}
                    </button>
                  ))}
                </div>
                <button onClick={() => setIsBottomOpen(false)} className="text-slate-500 hover:text-slate-300">_</button>
              </div>

              <div className="flex-1 overflow-hidden p-2 text-xs font-mono">
                {bottomTab === "TERMINAL" && (
                  <div className="flex flex-col h-full">
                    <div className="flex-1 overflow-y-auto space-y-1 text-slate-300">
                      {terminalHistory.map((line, i) => (
                        <div key={i} className={line.type === "stderr" ? "text-rose-400" : line.type === "stdin" ? "text-sky-300 font-bold" : "text-slate-300"}>
                          {line.text}
                        </div>
                      ))}
                    </div>
                    <form onSubmit={handleTerminalSubmit} className="pt-2 flex items-center space-x-2 border-t border-slate-800">
                      <span className="text-emerald-400 font-bold">$</span>
                      <input
                        type="text"
                        className="flex-1 bg-transparent outline-none text-slate-100 text-xs font-mono"
                        placeholder="python train.py"
                        value={terminalInput}
                        onChange={(e) => setTerminalInput(e.target.value)}
                      />
                    </form>
                  </div>
                )}

                {bottomTab === "KERNEL" && (
                  <div className="grid grid-cols-4 gap-4 p-3 text-slate-300">
                    <div className="p-3 bg-slate-950 rounded border border-slate-800">
                      <div className="text-[10px] text-slate-500">KERNEL STATUS</div>
                      <div className="text-sm font-bold text-emerald-400 flex items-center space-x-1.5 mt-1">
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                        <span>{kernelStats.status}</span>
                      </div>
                    </div>
                    <div className="p-3 bg-slate-950 rounded border border-slate-800">
                      <div className="text-[10px] text-slate-500">MEMORY CONSUMPTION</div>
                      <div className="text-sm font-bold text-sky-400 mt-1">{kernelStats.memory_mb} MB</div>
                    </div>
                    <div className="p-3 bg-slate-950 rounded border border-slate-800">
                      <div className="text-[10px] text-slate-500">CPU UTILIZATION</div>
                      <div className="text-sm font-bold text-amber-400 mt-1">{kernelStats.cpu_percent}%</div>
                    </div>
                    <div className="p-3 bg-slate-950 rounded border border-slate-800 flex items-center justify-between">
                      <button
                        onClick={() => KernelAPI.restart(kernelSession?.session_id)}
                        className="px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded border border-slate-700 font-semibold"
                      >
                        Restart Kernel
                      </button>
                    </div>
                  </div>
                )}

                {bottomTab === "ARTIFACTS" && (
                  <div className="p-3 space-y-2 text-slate-300">
                    <div className="text-[11px] text-slate-500 font-bold">GENERATED ARTIFACTS</div>
                    <div className="flex items-center space-x-3">
                      <div className="p-2 bg-slate-950 border border-slate-800 rounded flex items-center space-x-2">
                        <span>📊</span>
                        <span>salary_distribution.png</span>
                      </div>
                      <div className="p-2 bg-slate-950 border border-slate-800 rounded flex items-center space-x-2">
                        <span>📦</span>
                        <span>model.joblib</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </section>
          )}
        </main>

        {/* RIGHT: AI Engineering Threads */}
        {isThreadsOpen && (
          <aside style={{ width: threadsWidth }} className="bg-slate-900 border-l border-slate-800 flex flex-col shrink-0">
            <div className="h-8 px-3 border-b border-slate-800 flex items-center justify-between text-[11px] font-bold text-slate-400 tracking-wider">
              <span>AI THREADS</span>
              <button
                onClick={() => {
                  const newT = {
                    id: "thread_" + Date.now(),
                    project_id: currentProject.name,
                    title: "New Investigation",
                    messages: [{ id: "m1", sender: "ai", text: "Ready to assist with ML workflow." }]
                  };
                  setThreads(prev => [newT, ...prev]);
                  setActiveThread(newT);
                }}
                className="text-sky-400 hover:text-sky-300 font-bold"
              >
                + New Thread
              </button>
            </div>

            {/* Conversation Messages */}
            <div className="flex-1 overflow-y-auto p-3 space-y-3 text-xs">
              {activeThread?.messages?.map((msg) => (
                <div
                  key={msg.id}
                  className={`p-3 rounded-lg border text-xs leading-relaxed ${
                    msg.sender === "user"
                      ? "bg-slate-800 border-slate-700 text-slate-100 ml-4"
                      : "bg-slate-950 border-slate-800 text-slate-300 mr-2"
                  }`}
                >
                  <div className="text-[10px] font-bold text-slate-500 mb-1">
                    {msg.sender === "user" ? "USER" : "AMEA AGENT"}
                  </div>
                  <div className="whitespace-pre-wrap">{msg.text}</div>
                  
                  {msg.code_diff && (
                    <div className="mt-3 p-2 bg-slate-900 border border-slate-800 rounded font-mono text-[11px]">
                      <div className="text-sky-400 text-[10px] font-bold mb-1">PROPOSED CODE</div>
                      <pre className="text-slate-300 overflow-x-auto">{msg.code_diff.code}</pre>
                      <button
                        onClick={() => applyAiCodeToNotebook(msg.code_diff.code)}
                        className="mt-2 w-full py-1 bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold rounded text-[11px] transition-all"
                      >
                        + Add to Notebook
                      </button>
                    </div>
                  )}
                </div>
              ))}
              {isAiStreaming && (
                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg text-slate-400 text-xs">
                  <span>AMEA is analyzing and writing Python code</span>
                  <span className="stream-cursor ml-1"></span>
                </div>
              )}
            </div>

            {/* AI Prompt Input Bar */}
            <div className="p-3 border-t border-slate-800">
              <div className="relative">
                <textarea
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-100 outline-none focus:border-sky-500 resize-none"
                  rows={3}
                  placeholder="Ask AMEA to write an analysis cell, clean missing values, or train a model..."
                  value={aiInput}
                  onChange={(e) => setAiInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleAiSend();
                    }
                  }}
                />
                <button
                  onClick={handleAiSend}
                  className="absolute right-2 bottom-2.5 px-3 py-1 bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold rounded text-xs transition-all"
                >
                  Ask
                </button>
              </div>
            </div>
          </aside>
        )}
      </div>

      {/* Download ZIP Modal */}
      {showDownloadModal && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 max-w-md w-full shadow-2xl space-y-4">
            <h3 className="text-base font-bold text-slate-100">Download Project Bundle</h3>
            <p className="text-xs text-slate-400">
              Export <strong>{currentProject.name}</strong> as a clean, production-ready ZIP archive.
            </p>
            <div className="space-y-2 text-xs font-mono text-slate-300">
              <div className="flex items-center space-x-2"><span>☑</span><span>Source code (src/)</span></div>
              <div className="flex items-center space-x-2"><span>☑</span><span>Notebooks (.ipynb)</span></div>
              <div className="flex items-center space-x-2"><span>☑</span><span>requirements.txt</span></div>
              <div className="flex items-center space-x-2"><span>☑</span><span>Generated Artifacts</span></div>
            </div>
            <div className="pt-2 flex justify-end space-x-2">
              <button
                onClick={() => setShowDownloadModal(false)}
                className="px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs"
              >
                Cancel
              </button>
              <a
                href={ProjectAPI.getDownloadZipUrl(currentProject.path)}
                download
                onClick={() => setShowDownloadModal(false)}
                className="px-4 py-1.5 rounded bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold text-xs"
              >
                Download ZIP
              </a>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}


// ============================================================
// Subcomponents
// ============================================================

function LauncherScreen({ onOpenNew, onOpenExisting, showNewModal, setShowNewModal, onCreateProject }) {
  const [projName, setProjName] = useState("customer-churn-ai");
  const [template, setTemplate] = useState("classification");

  return (
    <div className="h-screen w-screen bg-slate-950 flex flex-col items-center justify-center p-6 text-slate-100 select-none">
      <div className="max-w-xl w-full text-center space-y-6">
        <div className="inline-block p-3 rounded-2xl bg-sky-500/10 border border-sky-500/20 text-sky-400 font-bold text-2xl font-mono">
          AMEA
        </div>
        <h1 className="text-3xl font-extrabold tracking-tight text-slate-100">
          Autonomous ML Engineering Workspace
        </h1>
        <p className="text-sm text-slate-400">
          Interactive Python Execution • AI Coding Agents • Visual Graph Workflows
        </p>

        <div className="grid grid-cols-2 gap-4 pt-4">
          <button
            onClick={onOpenNew}
            className="p-6 bg-slate-900 hover:bg-slate-800/80 border border-slate-800 hover:border-sky-500 rounded-xl flex flex-col items-center justify-center space-y-2 transition-all group"
          >
            <span className="text-3xl text-sky-400 group-hover:scale-110 transition-transform">+</span>
            <span className="text-sm font-bold text-slate-200">New Project</span>
            <span className="text-xs text-slate-500">Starter ML pipelines & requirements</span>
          </button>

          <button
            onClick={() => onOpenExisting("workspace/customer-churn-ai")}
            className="p-6 bg-slate-900 hover:bg-slate-800/80 border border-slate-800 hover:border-slate-700 rounded-xl flex flex-col items-center justify-center space-y-2 transition-all"
          >
            <span className="text-3xl text-slate-400">📁</span>
            <span className="text-sm font-bold text-slate-200">Open Folder</span>
            <span className="text-xs text-slate-500">Inspect existing dataset or workspace</span>
          </button>
        </div>

        <div className="pt-6 border-t border-slate-900 text-left">
          <div className="text-xs font-bold text-slate-500 tracking-wider mb-2">RECENT PROJECTS</div>
          <div className="space-y-1.5">
            {["customer-churn-ai", "house-price-prediction", "fraud-detection-model"].map(p => (
              <div
                key={p}
                onClick={() => onOpenExisting(`workspace/${p}`)}
                className="p-2.5 bg-slate-900/50 hover:bg-slate-900 border border-slate-800/50 rounded-lg text-xs font-mono text-slate-300 cursor-pointer flex justify-between items-center"
              >
                <span>{p}</span>
                <span className="text-[10px] text-slate-500">Last opened 2h ago</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {showNewModal && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 max-w-md w-full space-y-4">
            <h3 className="text-base font-bold text-slate-100">Create New Project</h3>
            <div>
              <label className="text-xs text-slate-400">Project Name</label>
              <input
                type="text"
                className="w-full mt-1 bg-slate-950 border border-slate-800 rounded p-2 text-xs text-slate-100 outline-none focus:border-sky-500 font-mono"
                value={projName}
                onChange={(e) => setProjName(e.target.value)}
              />
            </div>
            <div>
              <label className="text-xs text-slate-400">Template</label>
              <select
                className="w-full mt-1 bg-slate-950 border border-slate-800 rounded p-2 text-xs text-slate-100 outline-none focus:border-sky-500"
                value={template}
                onChange={(e) => setTemplate(e.target.value)}
              >
                <option value="classification">Classification Pipeline</option>
                <option value="regression">Regression Pipeline</option>
                <option value="data_analysis">Data Analysis & EDA</option>
                <option value="empty">Empty Project</option>
              </select>
            </div>
            <div className="flex justify-end space-x-2 pt-2">
              <button
                onClick={() => setShowNewModal(false)}
                className="px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-xs"
              >
                Cancel
              </button>
              <button
                onClick={() => onCreateProject({ name: projName, template })}
                className="px-4 py-1.5 rounded bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold text-xs"
              >
                Create Project
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function NotebookCellView({ index, cell, isActive, onFocus, onChangeCode, onRun, onRunFromHere, onDelete, onAddCell }) {
  return (
    <div
      onClick={onFocus}
      className={`notebook-cell bg-slate-900 border rounded-xl overflow-hidden transition-all ${
        isActive ? "border-sky-500/50 shadow-lg shadow-sky-500/5" : "border-slate-800"
      }`}
    >
      {/* Cell Header / Run Controls */}
      <div className="h-8 bg-slate-950/60 border-b border-slate-800/80 px-3 flex items-center justify-between text-xs">
        <div className="flex items-center space-x-2">
          <button
            onClick={onRun}
            className={`w-6 h-6 rounded flex items-center justify-center font-bold text-xs transition-all ${
              cell.status === "RUNNING"
                ? "bg-amber-400 text-slate-950 animate-spin"
                : cell.status === "SUCCESS"
                ? "bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30"
                : cell.status === "ERROR"
                ? "bg-rose-500/20 text-rose-400 hover:bg-rose-500/30"
                : "bg-sky-500/20 text-sky-400 hover:bg-sky-500/30"
            }`}
            title="Run Cell (Shift+Enter)"
          >
            {cell.status === "RUNNING" ? "◉" : cell.status === "SUCCESS" ? "✓" : cell.status === "ERROR" ? "!" : "▶"}
          </button>
          <span className="font-mono text-[11px] text-slate-500">
            [{cell.execCount ? cell.execCount : index}]
          </span>
        </div>

        <div className="flex items-center space-x-2 text-slate-400">
          <button onClick={onRunFromHere} className="hover:text-slate-200 text-[11px]">Run from here</button>
          <span>•</span>
          <button onClick={onDelete} className="hover:text-rose-400 text-[11px]">Delete</button>
        </div>
      </div>

      {/* Code Editor Area */}
      <div className="p-3 bg-slate-950 font-mono text-xs">
        <textarea
          rows={Math.max(3, cell.code.split("\n").length)}
          className="w-full bg-transparent resize-none outline-none text-slate-200 font-mono leading-relaxed"
          value={cell.code}
          onChange={(e) => onChangeCode(e.target.value)}
        />
      </div>

      {/* Rich Outputs View (DataFrame, Plot, Stream, Error) */}
      {cell.output && cell.output.length > 0 && (
        <div className="p-3 bg-slate-900/90 border-t border-slate-800 space-y-2">
          {cell.output.map((out, i) => (
            <div key={i}>
              {out.output_type === "STREAM" && (
                <pre className="text-slate-300 font-mono text-xs whitespace-pre-wrap">{out.text}</pre>
              )}
              {out.output_type === "SCALAR" && (
                <div className="p-2 bg-slate-950 rounded font-mono text-xs text-sky-400 font-bold">
                  {out.scalar_value}
                </div>
              )}
              {out.output_type === "DATAFRAME" && out.dataframe && (
                <div className="overflow-x-auto max-h-60 border border-slate-800 rounded">
                  <table className="w-full text-left text-[11px] font-mono dataframe-table">
                    <thead>
                      <tr className="border-b border-slate-700 bg-slate-800 text-slate-300">
                        {out.dataframe.columns.map(c => <th key={c} className="p-2">{c}</th>)}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800 text-slate-300">
                      {out.dataframe.data.map((row, rIdx) => (
                        <tr key={rIdx} className="hover:bg-slate-800/40">
                          {out.dataframe.columns.map(c => <td key={c} className="p-2">{row[c]}</td>)}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {out.output_type === "IMAGE" && out.image_base64 && (
                <div className="p-2 bg-slate-950 rounded border border-slate-800 flex justify-center">
                  <img src={`data:image/png;base64,${out.image_base64}`} alt="Plot" className="max-h-72 rounded" />
                </div>
              )}
              {out.output_type === "ERROR" && (
                <div className="p-3 bg-rose-950/40 border border-rose-900/50 rounded-lg text-rose-300 font-mono text-xs space-y-1">
                  <div className="font-bold">{out.error_name}: {out.error_value}</div>
                  {out.traceback && <pre className="text-[11px] text-rose-400 whitespace-pre-wrap">{out.traceback.join("\n")}</pre>}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function FileTreeNode({ tree, onSelectFile }) {
  return (
    <div className="space-y-1">
      {tree.map(node => (
        <div key={node.path}>
          <div
            onClick={() => !node.is_dir && onSelectFile(node.path)}
            className="flex items-center space-x-1.5 py-1 px-1.5 hover:bg-slate-800 rounded cursor-pointer text-slate-300 hover:text-slate-100"
          >
            <span>{node.is_dir ? "📁" : node.name.endsWith(".py") ? "🐍" : node.name.endsWith(".csv") ? "📊" : "📄"}</span>
            <span className="truncate">{node.name}</span>
          </div>
          {node.is_dir && node.children && (
            <div className="pl-4 border-l border-slate-800 ml-2">
              <FileTreeNode tree={node.children} onSelectFile={onSelectFile} />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function WorkflowGraphView({ onRunNode }) {
  const nodes = [
    { id: "n1", title: "Load Dataset", status: "COMPLETED" },
    { id: "n2", title: "Data Cleaning", status: "COMPLETED" },
    { id: "n3", title: "Train Baseline", status: "READY" },
    { id: "n4", title: "Evaluate Model", status: "PENDING" },
  ];

  return (
    <div className="flex-1 p-8 flex items-center justify-center space-x-6 overflow-auto">
      {nodes.map((node, i) => (
        <React.Fragment key={node.id}>
          <div className="w-52 p-4 bg-slate-900 border border-slate-800 rounded-xl space-y-3 shadow-xl">
            <div className="flex justify-between items-center">
              <span className="text-xs font-bold text-slate-200">{node.title}</span>
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                node.status === "COMPLETED" ? "bg-emerald-500/20 text-emerald-400" : "bg-sky-500/20 text-sky-400"
              }`}>
                {node.status}
              </span>
            </div>
            <button
              onClick={() => onRunNode(node.id)}
              className="w-full py-1.5 bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold rounded text-xs transition-all flex justify-center items-center space-x-1"
            >
              <span>▶</span>
              <span>Run Step</span>
            </button>
          </div>
          {i < nodes.length - 1 && <span className="text-slate-600 text-lg font-bold">→</span>}
        </React.Fragment>
      ))}
    </div>
  );
}

// Mount React Root
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
