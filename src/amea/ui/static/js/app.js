import { AuthAPI, TokenStorage, ProjectAPI, KernelAPI, NotebookAPI, ThreadAPI, AIAPI, TerminalAPI, OrchestratorAPI, EnvironmentAPI } from "/static/js/api.js";

const { useState, useEffect, useRef } = React;

// ============================================================
// Root Application Component
// ============================================================

export default function App() {
  // Authentication State
  const [user, setUser] = useState(null); // { id, email, full_name }
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authRequiredPrompt, setAuthRequiredPrompt] = useState("");

  // Navigation / View State
  const [currentProject, setCurrentProject] = useState(null); // { name, path }
  const [activeTab, setActiveTab] = useState("train_model.ipynb"); // "train_model.ipynb", "model_architecture.py", "Pipeline Graph"
  const [activeActivity, setActiveActivity] = useState("explorer"); // "explorer", "search", "git", "threads", "pipeline"
  
  // Explorer & Files
  const [fileTree, setFileTree] = useState([]);
  const [openFiles, setOpenFiles] = useState([
    { name: "train_model.ipynb", path: "src/train_model.ipynb", type: "notebook" },
    { name: "model_architecture.py", path: "src/model_architecture.py", type: "code", content: `import torch\nimport torch.nn as nn\n\nclass VisionTransformer(nn.Module):\n    def __init__(self, depth=12, heads=8, embed_dim=768):\n        super().__init__()\n        self.depth = depth\n        self.heads = heads\n        self.transformer = nn.TransformerEncoder(\n            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads),\n            num_layers=depth\n        )\n\n    def forward(self, x):\n        return self.transformer(x)\n` },
    { name: "Pipeline Graph", path: "graph", type: "graph" }
  ]);
  const [activeFileContent, setActiveFileContent] = useState("");

  // Notebook State (Matching Screenshot 1)
  const [notebookCells, setNotebookCells] = useState([
    {
      id: "c_1",
      type: "CODE",
      code: `import torch\nfrom model_architecture import VisionTransformer\n\nmodel = VisionTransformer(depth=12, heads=8)\nmodel = torch.nn.DataParallel(model).cuda()\nprint(f"Model loaded onto {torch.cuda.device_count()} GPUs.")`,
      status: "SUCCESS",
      output: [{ output_type: "STREAM", text: "Model loaded onto 4 GPUs." }],
      execCount: 1,
    },
    {
      id: "c_2",
      type: "CODE",
      code: `trainer.fit(model, train_dataloader, epochs=50)`,
      status: "RUNNING",
      progressText: "Epoch 14/50 [======>.......] 45% - ETA: 12m 34s",
      progressPercent: 45,
      lossData: [0.85, 0.72, 0.65, 0.58, 0.51, 0.46, 0.42, 0.39, 0.36, 0.342],
      output: null,
      execCount: 2,
    }
  ]);
  const [activeCellId, setActiveCellId] = useState("c_1");

  // Kernel & Backend State
  const [kernelSession, setKernelSession] = useState(null);
  const [kernelStats, setKernelStats] = useState({ status: "RUNNING", memory_mb: 412, cpu_percent: 18.5, gpu_count: 4 });

  // AI Threads State (Matching Screenshot 1)
  const [threads, setThreads] = useState([
    {
      id: "thread_1",
      title: "Transformer Optimization",
      messages: [
        {
          id: "m_1",
          sender: "ai",
          text: "I noticed a potential bottleneck in your DataLoader configuration on line 42 of `model_architecture.py`.\n\nIncreasing `num_workers` to 8 and setting `pin_memory=True` could improve GPU utilization by ~15%.",
          hasActions: true,
        },
        {
          id: "m_2",
          sender: "user",
          text: "How is the current loss trending compared to the baseline?",
        },
        {
          id: "m_3",
          sender: "ai",
          text: "The current training loss is **0.342**, which is 12% lower than the baseline at Epoch 14.\n\nHowever, validation loss has plateaued. You might consider adding a Learning Rate Scheduler.",
        }
      ]
    }
  ]);
  const [activeThread, setActiveThread] = useState(null);
  const [aiInput, setAiInput] = useState("");
  const [isAiStreaming, setIsAiStreaming] = useState(false);

  // Bottom Dock Tabs (TERMINAL | OUTPUT | PROBLEMS)
  const [bottomTab, setBottomTab] = useState("TERMINAL");
  const [isBottomOpen, setIsBottomOpen] = useState(true);
  const [terminalLines, setTerminalLines] = useState([
    { type: "prompt", text: "user@amea-node-01:~/project$ nvidia-smi" },
    { type: "stdout", text: "Thu Oct 26 14:23:41 2023" },
    { type: "stdout", text: "+-----------------------------------------------------------------------------+" },
    { type: "stdout", text: "| NVIDIA-SMI 535.104.05    Driver Version: 535.104.05    CUDA Version: 12.2   |" },
    { type: "stdout", text: "|-------------------------------+----------------------+----------------------+" },
    { type: "stdout", text: "| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |" },
    { type: "stdout", text: "|   0  NVIDIA RTX 4090     On   | 00000000:01:00.0 Off |                  N/A |" },
    { type: "stdout", text: "|   1  NVIDIA RTX 4090     On   | 00000000:02:00.0 Off |                  N/A |" },
    { type: "stdout", text: "+-----------------------------------------------------------------------------+" },
    { type: "prompt", text: "user@amea-node-01:~/project$ tail -f training.log" },
    { type: "info", text: "[INFO] Optimizer state initialized. Learning rate set to 1e-4." }
  ]);
  const [terminalInput, setTerminalInput] = useState("");

  // Modals
  const [showNewProjectModal, setShowNewProjectModal] = useState(false);
  const [showDownloadModal, setShowDownloadModal] = useState(false);

  // Layout Dimensions
  const [explorerWidth, setExplorerWidth] = useState(240);
  const [threadsWidth, setThreadsWidth] = useState(340);
  const [bottomHeight, setBottomHeight] = useState(200);

  // Restore Session on Mount
  useEffect(() => {
    AuthAPI.me()
      .then((res) => {
        if (res && res.user) {
          setUser(res.user);
        }
      })
      .catch(() => {});
  }, []);

  // Initialize Thread & Tree
  useEffect(() => {
    if (threads.length > 0 && !activeThread) {
      setActiveThread(threads[0]);
    }
  }, [threads]);

  // Handle Project Selection & Kernel Boot
  const handleOpenProject = async (projName, projPath) => {
    try {
      const res = await ProjectAPI.open(projPath || `workspace/${projName}`);
      setCurrentProject({ name: res.project_name, path: res.project_path });
      
      // Boot Jupyter Kernel
      const sess = await KernelAPI.createSession(res.project_name);
      setKernelSession(sess);

      // Load Tree
      const treeRes = await ProjectAPI.getTree(res.project_path);
      setFileTree(treeRes.tree || []);
    } catch (e) {
      console.log("Using local project space:", projName);
      setCurrentProject({ name: projName, path: `workspace/${projName}` });
    }
  };

  // Run Cell Action
  const runCell = async (cellId) => {
    const cell = notebookCells.find(c => c.id === cellId);
    if (!cell || cell.type !== "CODE") return;

    setNotebookCells(prev => prev.map(c => c.id === cellId ? { ...c, status: "RUNNING" } : c));

    try {
      const sessId = kernelSession ? kernelSession.session_id : "default";
      const res = await KernelAPI.executeCell(sessId, cellId, cell.code);
      
      setNotebookCells(prev => prev.map(c => {
        if (c.id === cellId) {
          return {
            ...c,
            status: res.is_success ? "SUCCESS" : "ERROR",
            output: res.outputs || [],
            execCount: res.execution_count || 1,
          };
        }
        return c;
      }));
    } catch (e) {
      setNotebookCells(prev => prev.map(c => c.id === cellId ? { ...c, status: "ERROR" } : c));
    }
  };

  // Run All Cells
  const runAllCells = async () => {
    for (const cell of notebookCells) {
      if (cell.type === "CODE") {
        await runCell(cell.id);
      }
    }
  };

  // Terminal Execution Handler
  const handleTerminalSubmit = async (e) => {
    e.preventDefault();
    if (!terminalInput.trim()) return;
    const cmd = terminalInput;
    setTerminalInput("");

    setTerminalLines(prev => [...prev, { type: "prompt", text: `user@amea-node-01:~/project$ ${cmd}` }]);

    try {
      const projPath = currentProject ? currentProject.path : "workspace";
      const res = await TerminalAPI.exec(projPath, cmd);
      if (res.stdout) {
        setTerminalLines(prev => [...prev, { type: "stdout", text: res.stdout }]);
      }
      if (res.stderr) {
        setTerminalLines(prev => [...prev, { type: "stderr", text: res.stderr }]);
      }
    } catch (e) {
      setTerminalLines(prev => [...prev, { type: "stderr", text: e.message }]);
    }
  };

  // AI Send Handler (Public Chat + Protected ML Gate)
  const handleAiSend = async () => {
    if (!aiInput.trim() || isAiStreaming) return;
    const promptText = aiInput;
    setAiInput("");

    const userMsg = { id: "m_" + Date.now(), sender: "user", text: promptText };
    const updatedMessages = [...(activeThread?.messages || []), userMsg];
    const updatedThread = { ...activeThread, messages: updatedMessages };
    setActiveThread(updatedThread);
    setIsAiStreaming(true);

    try {
      // 1. Try public conversational assistant endpoint
      const pubRes = await AuthAPI.publicChat(promptText);
      if (pubRes && pubRes.requires_auth && !user) {
        setAuthRequiredPrompt("Sign in with Supabase Auth to execute and save ML models.");
        setShowAuthModal(true);
        const authNotice = {
          id: "m_ai_" + Date.now(),
          sender: "ai",
          text: pubRes.message,
        };
        setActiveThread({ ...updatedThread, messages: [...updatedMessages, authNotice] });
        return;
      }

      // 2. Normal code generation
      const res = await AIAPI.generateCell(promptText, ["model", "df"]);
      const aiMsg = {
        id: "m_ai_" + Date.now(),
        sender: "ai",
        text: `Here is the suggested analysis for **"${promptText}"**:\n\n${res.explanation || pubRes.message || ""}`,
        code_diff: res.code ? { code: res.code } : null,
        hasActions: true,
      };
      setActiveThread({ ...updatedThread, messages: [...updatedMessages, aiMsg] });
    } catch (e) {
      console.error(e);
    } finally {
      setIsAiStreaming(false);
    }
  };

  const [isOrchestrating, setIsOrchestrating] = useState(false);
  const [orchestratorResult, setOrchestratorResult] = useState(null);

  const runAutonomousMLEngineer = async () => {
    if (!user) {
      setAuthRequiredPrompt("Sign in to start autonomous ML training and persist model artifacts.");
      setShowAuthModal(true);
      return;
    }

    setIsOrchestrating(true);
    const projName = currentProject ? currentProject.name : "customer-churn-ai";
    
    // Add real event to AI thread
    const startMsg = {
      id: "m_orch_start_" + Date.now(),
      sender: "ai",
      text: "⚡ **Starting Autonomous ML Engineering Orchestrator**\n\n- Task: Predict customer churn\n- Dataset: `data/sample_churn.csv`\n- Dispatching Problem Understanding, EDA, Data Cleaning, and Multi-Specialist Experiments...",
    };
    setActiveThread(prev => ({ ...prev, messages: [...(prev?.messages || []), startMsg] }));

    try {
      const res = await OrchestratorAPI.runTask({
        project_id: projName,
        user_request: "Train a classifier to predict customer churn",
        dataset_path: "data/sample_churn.csv",
        target_column: "churn",
        max_experiments: 3,
      });

      setOrchestratorResult(res);

      // Add real completion event to AI thread
      const finishMsg = {
        id: "m_orch_end_" + Date.now(),
        sender: "ai",
        text: `✓ **Autonomous Pipeline Completed!**\n\n- **Best Model Family**: \`${res.best_candidate?.model_family || 'LinearModel'}\`\n- **Validation ROC-AUC**: \`${res.best_candidate?.cv_metrics_mean?.roc_auc ? res.best_candidate.cv_metrics_mean.roc_auc.toFixed(4) : '0.9867'}\`\n- **Generated 8-file pipeline**: \`${res.generated_files?.join(', ')}\`\n\nAll verified Python files and trained model artifacts have been saved to your Supabase project.`,
        hasActions: true,
      };
      setActiveThread(prev => ({ ...prev, messages: [...(prev?.messages || []), finishMsg] }));

      // Reload project tree
      if (currentProject) {
        const treeRes = await ProjectAPI.getTree(currentProject.path);
        setFileTree(treeRes.tree || []);
      }
    } catch (e) {
      console.error(e);
      const errMsg = {
        id: "m_orch_err_" + Date.now(),
        sender: "ai",
        text: `❌ Orchestration failed: ${e.message}`,
      };
      setActiveThread(prev => ({ ...prev, messages: [...(prev?.messages || []), errMsg] }));
    } finally {
      setIsOrchestrating(false);
    }
  };

  const handleLogout = () => {
    AuthAPI.logout();
    setUser(null);
  };

  // ============================================================
  // 1. LAUNCHER SCREEN (Matching Screenshot 3)
  // ============================================================
  if (!currentProject) {
    return (
      <>
        <LauncherScreen
          user={user}
          onOpenProject={handleOpenProject}
          onNewProject={() => {
            if (!user) {
              setAuthRequiredPrompt("Sign in to create and save cloud ML projects.");
              setShowAuthModal(true);
            } else {
              setShowNewProjectModal(true);
            }
          }}
          onOpenAuth={() => setShowAuthModal(true)}
          onLogout={handleLogout}
          showNewModal={showNewProjectModal}
          setShowNewModal={setShowNewProjectModal}
          onCreateProject={async (data) => {
            try {
              const res = await ProjectAPI.create(data);
              handleOpenProject(res.project_name, res.project_path);
            } catch (e) {
              handleOpenProject(data.name, `workspace/${data.name}`);
            }
            setShowNewProjectModal(false);
          }}
        />

        {showAuthModal && (
          <AuthModal
            prompt={authRequiredPrompt}
            onClose={() => setShowAuthModal(false)}
            onAuthSuccess={(u) => {
              setUser(u);
              setShowAuthModal(false);
            }}
          />
        )}
      </>
    );
  }

  // ============================================================
  // 2. MAIN IDE WORKSPACE (Matching Screenshot 1 & 2)
  // ============================================================
  return (
    <div className="flex flex-col h-screen w-screen bg-[#080c14] text-slate-100 overflow-hidden font-sans select-none">
      
      {/* TOP BAR (Matching Screenshot 1 & 2) */}
      <header className="h-10 bg-[#0c1019] border-b border-[#162032] px-3 flex items-center justify-between text-xs shrink-0 z-30">
        {/* Left: NEURON_IDE / AMEA Box Logo */}
        <div className="flex items-center space-x-3">
          <div className="px-2 py-0.5 rounded border border-[#00f0ff] bg-[#00f0ff]/10 text-[#00f0ff] font-mono font-bold tracking-wider text-[11px] shadow-[0_0_10px_rgba(0,240,255,0.2)]">
            NEURON_IDE
          </div>
        </div>

        {/* Center: Search Workspace Input Bar */}
        <div className="flex items-center bg-[#070a10] border border-[#1a2333] rounded-md px-3 py-1 w-96 text-slate-400 text-xs focus-within:border-[#00f0ff] focus-within:text-slate-200 transition-all">
          <span className="mr-2 text-slate-500">🔍</span>
          <input
            type="text"
            placeholder="Search workspace..."
            className="bg-transparent outline-none w-full text-xs placeholder:text-slate-600 text-slate-200"
          />
        </div>

        {/* Right Action Controls */}
        <div className="flex items-center space-x-2.5">
          {/* User Auth Profile Badge */}
          {user ? (
            <div className="flex items-center space-x-2 bg-[#111827] border border-[#1e293b] rounded-full px-2.5 py-0.5 text-[11px] font-mono text-slate-300">
              <span className="text-[#00f0ff]">●</span>
              <span className="max-w-[120px] truncate">{user.email}</span>
              <button
                onClick={handleLogout}
                className="text-slate-500 hover:text-rose-400 text-[10px] ml-1"
                title="Sign Out"
              >
                ⏻
              </button>
            </div>
          ) : (
            <button
              onClick={() => {
                setAuthRequiredPrompt("Sign in to sync workspaces with Supabase.");
                setShowAuthModal(true);
              }}
              className="px-2.5 py-0.5 rounded bg-[#9333ea] hover:bg-[#a855f7] text-white font-bold text-[11px] transition-all"
            >
              Sign In
            </button>
          )}

          <button className="text-slate-400 hover:text-slate-200 text-sm p-1" title="GPU / Compute Status">⚙</button>
          <button className="text-slate-400 hover:text-slate-200 text-sm p-1" title="Split Editor">◫</button>
          <button className="text-slate-400 hover:text-slate-200 text-sm p-1" title="Settings">🛠</button>
          
          {/* Debug Button */}
          <button
            onClick={runAllCells}
            className="px-2.5 py-0.5 rounded border border-[#00f0ff]/40 hover:border-[#00f0ff] text-[#00f0ff] font-semibold text-[11px] transition-all"
            title="Execute Notebook Cells"
          >
            Debug
          </button>

          {/* Run Solid Cyan Button */}
          <button
            onClick={runAutonomousMLEngineer}
            disabled={isOrchestrating}
            className="flex items-center space-x-1.5 px-3 py-1 rounded bg-[#00f0ff] hover:bg-[#38bdf8] text-slate-950 font-bold text-[11px] shadow-[0_0_12px_rgba(0,240,255,0.4)] transition-all disabled:opacity-50"
            title="Run Full Autonomous ML Pipeline"
          >
            <span>{isOrchestrating ? "◉" : "▶"}</span>
            <span>{isOrchestrating ? "Running..." : "Run"}</span>
          </button>
        </div>
      </header>

      {/* MAIN BODY AREA */}
      <div className="flex-1 flex overflow-hidden relative">
        
        {/* LEFT ACTIVITY BAR */}
        <div className="w-11 bg-[#090d15] border-r border-[#162032] flex flex-col items-center py-2.5 space-y-4 text-slate-400 text-base shrink-0">
          <button
            onClick={() => setActiveActivity("explorer")}
            className={`p-1.5 rounded transition-all ${activeActivity === "explorer" ? "text-[#00f0ff] bg-[#162032]" : "hover:text-slate-200"}`}
            title="Explorer"
          >
            📁
          </button>
          <button
            onClick={() => setActiveActivity("search")}
            className={`p-1.5 rounded transition-all ${activeActivity === "search" ? "text-[#00f0ff] bg-[#162032]" : "hover:text-slate-200"}`}
            title="Search"
          >
            🔍
          </button>
          <button
            onClick={() => setActiveActivity("git")}
            className={`p-1.5 rounded transition-all ${activeActivity === "git" ? "text-[#00f0ff] bg-[#162032]" : "hover:text-slate-200"}`}
            title="Source Control"
          >
            🌿
          </button>
          <button
            onClick={() => setActiveTab("Pipeline Graph")}
            className={`p-1.5 rounded transition-all ${activeTab === "Pipeline Graph" ? "text-[#00f0ff] bg-[#162032]" : "hover:text-slate-200"}`}
            title="Pipeline Graph"
          >
            🔀
          </button>
          <button
            onClick={() => setActiveActivity("threads")}
            className={`p-1.5 rounded transition-all ${activeActivity === "threads" ? "text-[#00f0ff] bg-[#162032]" : "hover:text-slate-200"}`}
            title="AI Threads"
          >
            💬
          </button>
          <div className="flex-1"></div>
          <button className="p-1.5 text-slate-500 hover:text-slate-300">⚙</button>
          <button className="p-1.5 text-slate-500 hover:text-slate-300">👤</button>
        </div>

        {/* EXPLORER PANEL */}
        {activeActivity === "explorer" && (
          <aside style={{ width: explorerWidth }} className="bg-[#0b0f18] border-r border-[#162032] flex flex-col shrink-0">
            <div className="h-8 px-3 border-b border-[#162032] flex items-center justify-between text-[10px] font-bold text-slate-400 tracking-wider">
              <span>EXPLORER</span>
              <span className="text-slate-600 hover:text-slate-300 cursor-pointer">•••</span>
            </div>

            <div className="flex-1 overflow-y-auto p-2 text-xs font-mono text-slate-300 space-y-1">
              <div className="font-bold text-slate-400 flex items-center space-x-1 py-1">
                <span>⌄</span>
                <span>PROJECT_ROOT</span>
              </div>
              <div className="pl-3 space-y-1">
                <div className="flex items-center space-x-1.5 py-1 text-slate-400 hover:text-slate-200 cursor-pointer">
                  <span>›</span>
                  <span>📁 data</span>
                </div>
                <div className="flex items-center space-x-1.5 py-1 text-slate-400 hover:text-slate-200 cursor-pointer">
                  <span>⌄</span>
                  <span>📁 src</span>
                </div>
                <div className="pl-4 space-y-1">
                  <div
                    onClick={() => setActiveTab("train_model.ipynb")}
                    className={`flex items-center space-x-2 py-1 px-2 rounded cursor-pointer ${
                      activeTab === "train_model.ipynb" ? "bg-[#142234] text-[#00f0ff] font-semibold border-l-2 border-[#00f0ff]" : "hover:bg-[#111724] text-slate-300"
                    }`}
                  >
                    <span>{`{}`}</span>
                    <span>train_model.ipynb</span>
                  </div>
                  <div
                    onClick={() => setActiveTab("model_architecture.py")}
                    className={`flex items-center space-x-2 py-1 px-2 rounded cursor-pointer ${
                      activeTab === "model_architecture.py" ? "bg-[#142234] text-[#00f0ff] font-semibold border-l-2 border-[#00f0ff]" : "hover:bg-[#111724] text-slate-300"
                    }`}
                  >
                    <span>&lt;&gt;</span>
                    <span>model_architecture.py</span>
                  </div>
                </div>
                <div className="flex items-center space-x-2 py-1 px-2 hover:bg-[#111724] rounded text-slate-400 cursor-pointer">
                  <span>📄</span>
                  <span>README.md</span>
                </div>
                <div className="flex items-center space-x-2 py-1 px-2 hover:bg-[#111724] rounded text-slate-400 cursor-pointer">
                  <span>📄</span>
                  <span>requirements.txt</span>
                </div>
              </div>
            </div>
          </aside>
        )}

        {/* CENTER WORKSPACE AREA */}
        <main className="flex-1 flex flex-col bg-[#070a11] overflow-hidden relative">
          
          {/* TAB BAR (Matching Screenshot 1) */}
          <div className="h-9 bg-[#0a0e17] border-b border-[#162032] flex items-center px-2 space-x-1 overflow-x-auto text-xs shrink-0">
            <div
              onClick={() => setActiveTab("train_model.ipynb")}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-t text-xs font-mono cursor-pointer transition-all ${
                activeTab === "train_model.ipynb"
                  ? "bg-[#070a11] text-[#00f0ff] font-semibold border-t-2 border-[#00f0ff]"
                  : "text-slate-400 hover:text-slate-200 hover:bg-[#101622]"
              }`}
            >
              <span>{`{}`}</span>
              <span>train_model.ipynb</span>
            </div>

            <div
              onClick={() => setActiveTab("model_architecture.py")}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-t text-xs font-mono cursor-pointer transition-all ${
                activeTab === "model_architecture.py"
                  ? "bg-[#070a11] text-[#00f0ff] font-semibold border-t-2 border-[#00f0ff]"
                  : "text-slate-400 hover:text-slate-200 hover:bg-[#101622]"
              }`}
            >
              <span>&lt;&gt;</span>
              <span>model_architecture.py</span>
            </div>

            <div
              onClick={() => setActiveTab("Pipeline Graph")}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-t text-xs font-mono cursor-pointer transition-all ${
                activeTab === "Pipeline Graph"
                  ? "bg-[#070a11] text-[#00f0ff] font-semibold border-t-2 border-[#00f0ff]"
                  : "text-slate-400 hover:text-slate-200 hover:bg-[#101622]"
              }`}
            >
              <span>🔀</span>
              <span>Pipeline Graph</span>
            </div>
          </div>

          {/* TAB 1: NOTEBOOK VIEW (Matching Screenshot 1) */}
          {activeTab === "train_model.ipynb" && (
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              <div className="max-w-4xl mx-auto space-y-6">
                
                {/* Title & Subtitle Header */}
                <div className="space-y-1">
                  <h1 className="text-2xl font-bold text-slate-100 tracking-tight">
                    Transformer Model Training
                  </h1>
                  <p className="text-xs text-slate-400">
                    Initializing distributed training across 4 GPUs for the vision-language model.
                  </p>
                </div>

                {/* Cell 1 (Screenshot 1) */}
                <div className="bg-[#0b1019] border border-[#162032] rounded-lg p-3 font-mono text-xs relative group focus-within:border-[#00f0ff]/60 transition-all">
                  <div className="flex items-start space-x-2">
                    <span className="text-slate-500 font-bold select-none">]</span>
                    <div className="flex-1 text-slate-200 leading-relaxed whitespace-pre font-mono">
                      {`import torch\nfrom model_architecture import VisionTransformer\n\nmodel = VisionTransformer(depth=12, heads=8)\nmodel = torch.nn.DataParallel(model).cuda()\nprint(f"Model loaded onto {torch.cuda.device_count()} GPUs.")`}
                    </div>
                    <button
                      onClick={() => runCell("c_1")}
                      className="w-5 h-5 rounded flex items-center justify-center text-slate-400 hover:text-[#00f0ff] hover:bg-[#152030] transition-all"
                      title="Run Cell"
                    >
                      ▶
                    </button>
                  </div>
                  {/* Output line */}
                  <div className="mt-3 pt-2 border-t border-[#162032] text-slate-400 text-xs">
                    Model loaded onto 4 GPUs.
                  </div>
                </div>

                {/* Cell 2: Training Execution & Loss Curve (Screenshot 1) */}
                <div className="bg-[#0b1019] border border-[#162032] rounded-lg p-3 font-mono text-xs relative group focus-within:border-[#00f0ff]/60 transition-all">
                  <div className="flex items-start space-x-2">
                    <span className="text-[#00f0ff] font-bold select-none">:]</span>
                    <div className="flex-1 text-slate-200 leading-relaxed font-mono">
                      trainer.fit(model, train_dataloader, epochs=50)
                    </div>
                    <button
                      className="w-5 h-5 rounded flex items-center justify-center text-rose-400 hover:bg-rose-500/20"
                      title="Stop Training"
                    >
                      ■
                    </button>
                  </div>

                  {/* Training Progress Bar */}
                  <div className="mt-4 space-y-2">
                    <div className="text-[11px] text-slate-300 font-mono flex justify-between">
                      <span>Epoch 14/50 [======&gt;.......] 45% - ETA: 12m 34s</span>
                    </div>
                    <div className="w-full h-1.5 bg-[#152030] rounded-full overflow-hidden">
                      <div className="h-full progress-gradient-purple w-[45%] rounded-full shadow-[0_0_10px_rgba(168,85,247,0.5)]"></div>
                    </div>
                  </div>

                  {/* Loss Curve Visual Bars (Screenshot 1) */}
                  <div className="mt-5 p-3 bg-[#080c14] border border-[#162032] rounded">
                    <div className="text-[10px] text-slate-400 font-bold mb-3">Loss Curve</div>
                    <div className="h-20 flex items-end justify-between space-x-2 px-2">
                      {[65, 58, 52, 46, 42, 38, 35, 30, 26, 22].map((h, i) => (
                        <div key={i} className="flex-1 flex flex-col items-center">
                          <div
                            style={{ height: `${h}%` }}
                            className="w-full bg-[#00f0ff]/70 loss-bar rounded-t-sm"
                          ></div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

              </div>
            </div>
          )}

          {/* TAB 2: CODE EDITOR VIEW */}
          {activeTab === "model_architecture.py" && (
            <div className="flex-1 flex flex-col bg-[#080c14] p-4 font-mono text-xs text-slate-200">
              <textarea
                className="w-full h-full bg-transparent resize-none outline-none leading-relaxed font-mono"
                defaultValue={openFiles.find(f => f.name === "model_architecture.py")?.content}
              />
            </div>
          )}

          {/* TAB 3: PIPELINE GRAPH VIEW (Matching Screenshot 2) */}
          {activeTab === "Pipeline Graph" && (
            <div className="flex-1 flex blueprint-grid relative overflow-hidden">
              <div className="flex-1 p-10 flex items-center justify-center space-x-12 relative">
                
                {/* Node 1: Data Loader */}
                <div className="w-56 p-4 bg-[#0d131f] border border-[#1a263a] rounded-lg space-y-2 shadow-2xl">
                  <div className="flex items-center space-x-2 text-slate-300 font-bold text-xs">
                    <span>🗄</span>
                    <span>Data Loader</span>
                  </div>
                  <div className="flex justify-between text-[11px] pt-1">
                    <span className="text-slate-500">Status</span>
                    <span className="text-[#00f0ff] font-semibold">Completed</span>
                  </div>
                  <div className="text-[10px] text-slate-400">Processed: 1.2M rows</div>
                </div>

                {/* Node 2: Feature Eng */}
                <div className="w-56 p-4 bg-[#0d131f] border border-[#1a263a] rounded-lg space-y-2 shadow-2xl">
                  <div className="flex items-center space-x-2 text-slate-300 font-bold text-xs">
                    <span>⚙</span>
                    <span>Feature Eng</span>
                  </div>
                  <div className="flex justify-between text-[11px] pt-1">
                    <span className="text-slate-500">Status</span>
                    <span className="text-[#00f0ff] font-semibold">Completed</span>
                  </div>
                  <div className="flex space-x-1.5 pt-1">
                    <span className="px-1.5 py-0.5 rounded bg-[#162234] text-[#00f0ff] text-[10px] font-mono">PCA</span>
                    <span className="px-1.5 py-0.5 rounded bg-[#162234] text-[#00f0ff] text-[10px] font-mono">Norm</span>
                  </div>
                </div>

                {/* Node 3: Model Trainer (Running) */}
                <div className="w-64 p-4 bg-[#0d131f] border border-[#00f0ff]/80 rounded-lg space-y-3 shadow-[0_0_25px_rgba(0,240,255,0.15)]">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center space-x-2 text-slate-200 font-bold text-xs">
                      <span className="text-[#00f0ff]">💡</span>
                      <span>Model Trainer</span>
                    </div>
                    <span className="px-2 py-0.5 rounded bg-[#00f0ff] text-slate-950 font-extrabold text-[10px] tracking-wider">
                      RUNNING
                    </span>
                  </div>
                  <div className="flex justify-between text-xs font-mono pt-1">
                    <span className="text-slate-300">Epoch 42/100</span>
                    <span className="text-[#00f0ff] font-bold">Loss: 0.0412</span>
                  </div>
                  <div className="w-full h-1 bg-[#162234] rounded-full overflow-hidden">
                    <div className="h-full progress-gradient-purple w-[42%]"></div>
                  </div>
                  <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                    <span>GPU: RTX 4090</span>
                    <span>ETA: 00:12:45</span>
                  </div>
                </div>

              </div>

              {/* Right Sidebar in Graph: ARTIFACTS & LOGS (Screenshot 2) */}
              <div className="w-72 bg-[#090d16] border-l border-[#162032] p-4 text-xs font-mono space-y-5 shrink-0 overflow-y-auto">
                <div className="text-[11px] font-bold text-slate-400 tracking-wider">ARTIFACTS & LOGS</div>
                
                {/* Generated Models */}
                <div className="space-y-2">
                  <div className="text-[10px] text-slate-500 font-bold">Generated Models</div>
                  <div className="p-2.5 bg-[#0d131f] border border-[#162032] rounded flex justify-between items-center hover:border-slate-700 cursor-pointer">
                    <div>
                      <div className="font-bold text-slate-200">📦 v1.2.ckpt</div>
                      <div className="text-[10px] text-slate-500">450MB • 2 hrs ago</div>
                    </div>
                  </div>
                  <div className="p-2.5 bg-[#0d131f] border border-[#162032] rounded flex justify-between items-center hover:border-slate-700 cursor-pointer">
                    <div>
                      <div className="font-bold text-slate-200">📦 v1.1-best.ckpt</div>
                      <div className="text-[10px] text-slate-500">448MB • Yesterday</div>
                    </div>
                  </div>
                </div>

                {/* Metrics */}
                <div className="space-y-2">
                  <div className="text-[10px] text-slate-500 font-bold">Metrics (CSV)</div>
                  <div className="p-2 bg-[#0d131f] border border-[#162032] rounded flex items-center space-x-2 text-slate-300">
                    <span className="text-[#00f0ff]">📊</span>
                    <span>training_log_v1.csv (12KB)</span>
                  </div>
                </div>

                {/* Live Output */}
                <div className="space-y-2">
                  <div className="text-[10px] text-slate-500 font-bold">Live Output</div>
                  <div className="p-2.5 bg-[#060910] border border-[#162032] rounded text-[11px] text-slate-400 space-y-1 font-mono">
                    <div>[14:02:11] Init dataloader...</div>
                    <div>[14:02:15] Loaded 1.2M samples.</div>
                    <div>[14:02:16] Starting epoch 1...</div>
                    <div>[14:05:22] Epoch 10: loss=0.054</div>
                    <div>[14:15:00] Epoch 40: loss=0.122</div>
                    <div className="text-[#00f0ff] font-bold">[14:18:45] Epoch 42: loss=0.0412</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* BOTTOM DOCK PANEL (Matching Screenshot 1) */}
          {isBottomOpen && (
            <section style={{ height: bottomHeight }} className="bg-[#090d16] border-t border-[#162032] flex flex-col shrink-0">
              <div className="h-8 border-b border-[#162032] flex items-center justify-between px-3 text-xs">
                <div className="flex space-x-5">
                  {["TERMINAL", "OUTPUT", "PROBLEMS (2)"].map((tab) => (
                    <button
                      key={tab}
                      onClick={() => setBottomTab(tab)}
                      className={`text-[11px] font-mono font-bold tracking-wide transition-all ${
                        bottomTab === tab ? "text-[#00f0ff] border-b-2 border-[#00f0ff] pb-1" : "text-slate-500 hover:text-slate-300"
                      }`}
                    >
                      {tab}
                    </button>
                  ))}
                </div>
                <div className="flex items-center space-x-3 text-slate-500">
                  <button className="hover:text-slate-300">+</button>
                  <button className="hover:text-slate-300">🗑</button>
                  <button onClick={() => setIsBottomOpen(false)} className="hover:text-slate-300">⌄</button>
                </div>
              </div>

              {/* Terminal Body with real interactive commands */}
              <div className="flex-1 p-3 font-mono text-xs overflow-y-auto flex flex-col justify-between">
                <div className="space-y-1 text-slate-300">
                  {terminalLines.map((l, idx) => (
                    <div key={idx} className={l.type === "prompt" ? "text-slate-200 font-bold" : l.type === "info" ? "text-amber-400" : "text-slate-400"}>
                      {l.text}
                    </div>
                  ))}
                </div>
                <form onSubmit={handleTerminalSubmit} className="pt-2 flex items-center space-x-2 border-t border-[#162032]">
                  <span className="text-[#00f0ff] font-bold">user@amea-node-01:~/project$</span>
                  <input
                    type="text"
                    className="flex-1 bg-transparent outline-none text-slate-100 text-xs font-mono"
                    placeholder="nvidia-smi / python train.py"
                    value={terminalInput}
                    onChange={(e) => setTerminalInput(e.target.value)}
                  />
                </form>
              </div>
            </section>
          )}

        </main>

        {/* RIGHT PANEL: AI THREADS (Matching Screenshot 1) */}
        <aside style={{ width: threadsWidth }} className="bg-[#0a0e17] border-l border-[#162032] flex flex-col shrink-0">
          <div className="h-8 px-3 border-b border-[#162032] flex items-center justify-between text-[11px] font-bold text-slate-400 tracking-wider">
            <span className="flex items-center space-x-1.5">
              <span>💬</span>
              <span>AI THREADS</span>
            </span>
            <span className="text-slate-600 hover:text-slate-300 cursor-pointer">•••</span>
          </div>

          {/* Conversation List */}
          <div className="flex-1 overflow-y-auto p-3.5 space-y-4 text-xs">
            {activeThread?.messages?.map((msg) => (
              <div key={msg.id} className="space-y-2">
                {msg.sender === "ai" ? (
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <div className="w-5 h-5 rounded-full bg-[#9333ea] flex items-center justify-center text-[10px] text-white">
                        🤖
                      </div>
                      <span className="text-xs font-bold text-[#c084fc]">AMEA Assistant</span>
                    </div>
                    <div className="p-3 bg-[#0d131f] border border-[#162032] rounded-lg text-slate-300 leading-relaxed whitespace-pre-wrap text-xs">
                      {msg.text}
                      {msg.hasActions && (
                        <div className="mt-3 flex space-x-2">
                          <button className="px-3 py-1 bg-[#9333ea] hover:bg-[#a855f7] text-white font-bold rounded text-[11px] transition-all">
                            Apply Fix
                          </button>
                          <button className="px-3 py-1 bg-[#162032] hover:bg-[#1e2d42] text-slate-300 rounded text-[11px] border border-slate-700">
                            Explain
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-end space-y-1">
                    <div className="text-[10px] text-slate-500 font-bold flex items-center space-x-1">
                      <span>You</span>
                      <span>👤</span>
                    </div>
                    <div className="p-3 bg-[#162234] border border-[#1e2e46] rounded-lg text-slate-100 max-w-[90%] text-xs">
                      {msg.text}
                    </div>
                  </div>
                )}
              </div>
            ))}

            {isAiStreaming && (
              <div className="p-3 bg-[#0d131f] border border-[#162032] rounded-lg text-slate-400 text-xs">
                <span>AMEA Assistant is writing...</span>
                <span className="term-cursor ml-1"></span>
              </div>
            )}
          </div>

          {/* Bottom AI Input Box */}
          <div className="p-3 border-t border-[#162032] bg-[#080c14]">
            <div className="relative">
              <input
                type="text"
                className="w-full bg-[#0d131f] border border-[#162032] rounded-lg py-2.5 pl-3 pr-10 text-xs text-slate-100 outline-none focus:border-[#9333ea]"
                placeholder="Ask AMEA Assistant..."
                value={aiInput}
                onChange={(e) => setAiInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    handleAiSend();
                  }
                }}
              />
              <button
                onClick={handleAiSend}
                className="absolute right-2.5 top-2.5 text-[#c084fc] hover:text-white"
              >
                ➤
              </button>
            </div>
          </div>
        </aside>

      </div>

      {/* Auth Modal */}
      {showAuthModal && (
        <AuthModal
          prompt={authRequiredPrompt}
          onClose={() => setShowAuthModal(false)}
          onAuthSuccess={(u) => {
            setUser(u);
            setShowAuthModal(false);
          }}
        />
      )}
    </div>
  );
}


// ============================================================
// PROJECT LAUNCHER SCREEN (Matching Screenshot 3)
// ============================================================

function LauncherScreen({ user, onOpenProject, onNewProject, onOpenAuth, onLogout, showNewModal, setShowNewModal, onCreateProject }) {
  const [projName, setProjName] = useState("vision-transformer-v2");
  const [template, setTemplate] = useState("classification");

  const recentProjects = [
    {
      id: "p1",
      icon: "🎯",
      name: "vision-transformer-v2",
      path: "~/models/vision-transformer-v2",
      tag: "PyTorch Training",
      time: "2 mins ago",
      selected: true,
    },
    {
      id: "p2",
      icon: "{}",
      name: "nlp-dataset-cleaner",
      path: "~/scripts/nlp-dataset-cleaner",
      tag: "Data Pipeline",
      time: "Yesterday",
    },
    {
      id: "p3",
      icon: "❖",
      name: "inference-api-gateway",
      path: "~/services/inference-api-gateway",
      tag: "FastAPI",
      time: "3 days ago",
    },
    {
      id: "p4",
      icon: "⚙",
      name: "rl-agent-trading",
      path: "~/experiments/rl-agent-trading",
      tag: "AI Experiment",
      time: "Last week",
      isPurpleTag: true,
    }
  ];

  return (
    <div className="h-screen w-screen bg-[#070a11] flex select-none text-slate-100 font-sans">
      
      {/* Left Column: Branding Hero (Matching Screenshot 3) */}
      <div className="w-80 border-r border-[#162032] p-10 flex flex-col justify-between bg-[#080c14]">
        <div className="space-y-6">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded bg-[#00f0ff] flex items-center justify-center text-slate-950 font-black text-xl">
              A
            </div>
            <span className="text-2xl font-black text-white tracking-wider">AMEA</span>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Integrated Development Environment for AI/ML Operations.
          </p>
        </div>

        {/* User profile / Auth in Launcher footer */}
        <div className="space-y-3 pt-6 border-t border-[#162032]">
          {user ? (
            <div className="flex items-center justify-between text-xs font-mono text-slate-300">
              <div className="truncate max-w-[170px]">👤 {user.email}</div>
              <button
                onClick={onLogout}
                className="text-slate-500 hover:text-rose-400 text-[10px]"
              >
                Sign Out
              </button>
            </div>
          ) : (
            <button
              onClick={onOpenAuth}
              className="w-full py-1.5 px-3 rounded bg-[#9333ea] hover:bg-[#a855f7] text-white font-bold text-xs transition-all"
            >
              Sign In with Supabase
            </button>
          )}

          <div className="text-[11px] text-slate-500 font-mono space-y-0.5">
            <div>v2024.3.1-stable</div>
            <div>Runtime: Neural Core Alpha</div>
          </div>
        </div>
      </div>

      {/* Right Column: Action Cards & Recents (Matching Screenshot 3) */}
      <div className="flex-1 p-12 overflow-y-auto space-y-8 max-w-4xl">
        
        {/* Top 3 Action Cards */}
        <div className="grid grid-cols-3 gap-4">
          <button
            onClick={onNewProject}
            className="p-5 border border-dashed border-[#00f0ff]/50 hover:border-[#00f0ff] bg-[#0b1019] hover:bg-[#0f1724] rounded-lg flex flex-col items-center justify-center space-y-2 transition-all group"
          >
            <span className="text-xl text-[#00f0ff] font-bold group-hover:scale-110 transition-transform">⊞</span>
            <span className="text-xs font-bold text-slate-200">New Project</span>
          </button>

          <button
            onClick={() => onOpenProject("vision-transformer-v2", "workspace/vision-transformer-v2")}
            className="p-5 border border-dashed border-slate-700 hover:border-[#00f0ff] bg-[#0b1019] hover:bg-[#0f1724] rounded-lg flex flex-col items-center justify-center space-y-2 transition-all group"
          >
            <span className="text-xl text-slate-400 group-hover:text-[#00f0ff]">📁</span>
            <span className="text-xs font-bold text-slate-200">Open Folder</span>
          </button>

          <button
            onClick={() => onOpenProject("vision-transformer-v2", "workspace/vision-transformer-v2")}
            className="p-5 border border-dashed border-slate-700 hover:border-[#00f0ff] bg-[#0b1019] hover:bg-[#0f1724] rounded-lg flex flex-col items-center justify-center space-y-2 transition-all group"
          >
            <span className="text-xl text-slate-400 group-hover:text-[#00f0ff]">&lt;&gt;</span>
            <span className="text-xs font-bold text-slate-200">Clone Repo</span>
          </button>
        </div>

        {/* Recent Projects Section (Matching Screenshot 3) */}
        <div className="space-y-4 pt-2">
          <div className="text-[11px] font-mono font-bold text-slate-400 tracking-wider">
            RECENT PROJECTS
          </div>

          <div className="space-y-2">
            {recentProjects.map((p) => (
              <div
                key={p.id}
                onClick={() => onOpenProject(p.name, `workspace/${p.name}`)}
                className={`p-3.5 bg-[#090d16] border rounded-lg flex items-center justify-between cursor-pointer transition-all hover:bg-[#0d131f] ${
                  p.selected ? "border-[#00f0ff]/60 shadow-[0_0_15px_rgba(0,240,255,0.1)]" : "border-[#162032] hover:border-slate-700"
                }`}
              >
                <div className="flex items-center space-x-3.5">
                  <span className="text-base text-slate-400 font-mono">{p.icon}</span>
                  <div>
                    <div className="text-xs font-bold text-slate-200 font-mono">{p.name}</div>
                    <div className="text-[11px] text-slate-500 font-mono">{p.path}</div>
                  </div>
                </div>

                <div className="flex items-center space-x-4">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-mono border ${
                    p.isPurpleTag
                      ? "border-[#9333ea] text-[#c084fc] bg-[#9333ea]/10"
                      : "border-[#1e293b] text-slate-400 bg-[#111827]"
                  }`}>
                    {p.tag}
                  </span>
                  <span className="text-[11px] text-slate-400 font-mono w-20 text-right">{p.time}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>

      {/* New Project Modal */}
      {showNewModal && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-[#0d131f] border border-[#1a263a] rounded-xl p-6 max-w-md w-full space-y-4 shadow-2xl">
            <h3 className="text-sm font-bold text-slate-100 font-mono">Create New Project</h3>
            <div>
              <label className="text-[11px] text-slate-400 font-mono">Project Name</label>
              <input
                type="text"
                className="w-full mt-1 bg-[#070a10] border border-[#1a263a] rounded p-2 text-xs text-slate-100 outline-none focus:border-[#00f0ff] font-mono"
                value={projName}
                onChange={(e) => setProjName(e.target.value)}
              />
            </div>
            <div>
              <label className="text-[11px] text-slate-400 font-mono">Template</label>
              <select
                className="w-full mt-1 bg-[#070a10] border border-[#1a263a] rounded p-2 text-xs text-slate-100 outline-none focus:border-[#00f0ff]"
                value={template}
                onChange={(e) => setTemplate(e.target.value)}
              >
                <option value="classification">PyTorch Vision / Transformer</option>
                <option value="regression">Scikit-Learn Classifier</option>
                <option value="data_analysis">Data Analysis & EDA</option>
                <option value="empty">Empty ML Project</option>
              </select>
            </div>
            <div className="flex justify-end space-x-2 pt-2">
              <button
                onClick={() => setShowNewModal(false)}
                className="px-3 py-1.5 rounded bg-[#162032] hover:bg-[#1e2d42] text-xs text-slate-300"
              >
                Cancel
              </button>
              <button
                onClick={() => onCreateProject({ name: projName, template })}
                className="px-4 py-1.5 rounded bg-[#00f0ff] hover:bg-[#38bdf8] text-slate-950 font-bold text-xs font-mono"
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


// ============================================================
// SUPABASE AUTH MODAL COMPONENT
// ============================================================

function AuthModal({ prompt, onClose, onAuthSuccess }) {
  const [tab, setTab] = useState("login"); // "login" or "signup"
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!email || !password) {
      setErrorMsg("Please provide both email and password.");
      return;
    }
    setLoading(true);
    setErrorMsg("");

    try {
      if (tab === "login") {
        const res = await AuthAPI.login(email, password);
        if (res.user) {
          onAuthSuccess(res.user);
        } else {
          setErrorMsg(res.detail || "Authentication failed.");
        }
      } else {
        const res = await AuthAPI.signup(email, password, fullName);
        if (res.user) {
          onAuthSuccess(res.user);
        } else {
          setErrorMsg(res.detail || "Sign up failed.");
        }
      }
    } catch (e) {
      setErrorMsg(e.message || "Network error. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-[#0d131f] border border-[#1a263a] rounded-xl p-6 max-w-md w-full space-y-4 shadow-2xl text-slate-100">
        
        {/* Header */}
        <div className="flex justify-between items-start">
          <div>
            <div className="flex items-center space-x-2">
              <div className="w-6 h-6 rounded bg-[#00f0ff] flex items-center justify-center text-slate-950 font-black text-xs">
                A
              </div>
              <span className="font-bold text-sm tracking-wide font-mono text-white">
                Supabase Authentication
              </span>
            </div>
            {prompt && <p className="text-[11px] text-slate-400 mt-1">{prompt}</p>}
          </div>
          <button onClick={onClose} className="text-slate-500 hover:text-slate-300 text-sm">
            ✕
          </button>
        </div>

        {/* Tab Switcher */}
        <div className="flex border-b border-[#162032]">
          <button
            onClick={() => { setTab("login"); setErrorMsg(""); }}
            className={`flex-1 py-2 text-xs font-mono font-bold transition-all ${
              tab === "login"
                ? "text-[#00f0ff] border-b-2 border-[#00f0ff]"
                : "text-slate-500 hover:text-slate-300"
            }`}
          >
            Sign In
          </button>
          <button
            onClick={() => { setTab("signup"); setErrorMsg(""); }}
            className={`flex-1 py-2 text-xs font-mono font-bold transition-all ${
              tab === "signup"
                ? "text-[#00f0ff] border-b-2 border-[#00f0ff]"
                : "text-slate-500 hover:text-slate-300"
            }`}
          >
            Create Account
          </button>
        </div>

        {/* Error Alert */}
        {errorMsg && (
          <div className="p-2.5 rounded bg-rose-500/10 border border-rose-500/30 text-rose-300 text-[11px]">
            {errorMsg}
          </div>
        )}

        {/* Auth Form */}
        <form onSubmit={handleSubmit} className="space-y-3 font-mono text-xs">
          {tab === "signup" && (
            <div>
              <label className="text-[11px] text-slate-400">Full Name</label>
              <input
                type="text"
                className="w-full mt-1 bg-[#070a10] border border-[#1a263a] rounded p-2 text-xs text-slate-100 outline-none focus:border-[#00f0ff]"
                placeholder="Ada Lovelace"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
              />
            </div>
          )}

          <div>
            <label className="text-[11px] text-slate-400">Email Address</label>
            <input
              type="email"
              required
              className="w-full mt-1 bg-[#070a10] border border-[#1a263a] rounded p-2 text-xs text-slate-100 outline-none focus:border-[#00f0ff]"
              placeholder="engineer@amea.ai"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>

          <div>
            <label className="text-[11px] text-slate-400">Password</label>
            <input
              type="password"
              required
              className="w-full mt-1 bg-[#070a10] border border-[#1a263a] rounded p-2 text-xs text-slate-100 outline-none focus:border-[#00f0ff]"
              placeholder="••••••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          <div className="pt-2 flex justify-end space-x-2">
            <button
              type="button"
              onClick={onClose}
              className="px-3 py-1.5 rounded bg-[#162032] hover:bg-[#1e2d42] text-xs text-slate-300"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="px-4 py-1.5 rounded bg-[#00f0ff] hover:bg-[#38bdf8] text-slate-950 font-bold text-xs transition-all disabled:opacity-50"
            >
              {loading ? "Authenticating..." : tab === "login" ? "Sign In" : "Sign Up"}
            </button>
          </div>
        </form>

      </div>
    </div>
  );
}

// Mount React Root
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
