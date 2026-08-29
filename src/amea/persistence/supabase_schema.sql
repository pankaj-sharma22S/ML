-- ====================================================================
-- AMEA (Autonomous ML Engineer) Supabase Schema & Row Level Security
-- ====================================================================

-- 1. Profiles Table (extends auth.users)
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT,
    avatar_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Projects Table
CREATE TABLE IF NOT EXISTS public.projects (
    id TEXT PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    template TEXT DEFAULT 'classification',
    dataset_path TEXT,
    target_column TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Tasks Table
CREATE TABLE IF NOT EXISTS public.tasks (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES public.projects(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    user_request TEXT NOT NULL,
    task_spec JSONB,
    status TEXT DEFAULT 'PENDING',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Experiments Table
CREATE TABLE IF NOT EXISTS public.experiments (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES public.projects(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    model_family TEXT NOT NULL,
    model_class_name TEXT,
    hyperparameters JSONB DEFAULT '{}'::jsonb,
    cv_metrics JSONB DEFAULT '{}'::jsonb,
    train_metrics JSONB DEFAULT '{}'::jsonb,
    duration_sec NUMERIC DEFAULT 0,
    peak_memory_mb NUMERIC DEFAULT 0,
    exit_code INT DEFAULT 0,
    status TEXT DEFAULT 'COMPLETED',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 5. Agent Runs & Telemetry Table
CREATE TABLE IF NOT EXISTS public.agent_runs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES public.projects(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    phase TEXT NOT NULL,
    author_component TEXT NOT NULL,
    events JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 6. Artifacts Metadata Table
CREATE TABLE IF NOT EXISTS public.artifacts (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES public.projects(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    artifact_type TEXT NOT NULL, -- 'model', 'code', 'metrics', 'dataset', 'report'
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    size_bytes BIGINT DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 7. Execution Logs Table
CREATE TABLE IF NOT EXISTS public.execution_logs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES public.projects(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    command TEXT NOT NULL,
    stdout TEXT,
    stderr TEXT,
    exit_code INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ====================================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- ====================================================================

-- Enable RLS on all tables
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.experiments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.artifacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.execution_logs ENABLE ROW LEVEL SECURITY;

-- 1. Profiles Policies
CREATE POLICY "Users can view own profile" ON public.profiles
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON public.profiles
    FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can insert own profile" ON public.profiles
    FOR INSERT WITH CHECK (auth.uid() = id);

-- 2. Projects Policies
CREATE POLICY "Users can view own projects" ON public.projects
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own projects" ON public.projects
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own projects" ON public.projects
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own projects" ON public.projects
    FOR DELETE USING (auth.uid() = user_id);

-- 3. Tasks Policies
CREATE POLICY "Users can view own tasks" ON public.tasks
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own tasks" ON public.tasks
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own tasks" ON public.tasks
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own tasks" ON public.tasks
    FOR DELETE USING (auth.uid() = user_id);

-- 4. Experiments Policies
CREATE POLICY "Users can view own experiments" ON public.experiments
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own experiments" ON public.experiments
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own experiments" ON public.experiments
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own experiments" ON public.experiments
    FOR DELETE USING (auth.uid() = user_id);

-- 5. Agent Runs Policies
CREATE POLICY "Users can view own agent runs" ON public.agent_runs
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own agent runs" ON public.agent_runs
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- 6. Artifacts Policies
CREATE POLICY "Users can view own artifacts" ON public.artifacts
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own artifacts" ON public.artifacts
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own artifacts" ON public.artifacts
    FOR DELETE USING (auth.uid() = user_id);

-- 7. Execution Logs Policies
CREATE POLICY "Users can view own execution logs" ON public.execution_logs
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own execution logs" ON public.execution_logs
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- ====================================================================
-- Trigger to automatically create profile on Auth Signup
-- ====================================================================
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, email, full_name, avatar_url)
    VALUES (new.id, new.email, new.raw_user_meta_data->>'full_name', new.raw_user_meta_data->>'avatar_url')
    ON CONFLICT (id) DO NOTHING;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE PROCEDURE public.handle_new_user();
