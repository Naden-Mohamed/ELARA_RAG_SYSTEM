import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useState, useEffect, type CSSProperties, type ReactNode } from 'react';
import { QueryClient, QueryClientProvider, useQueryClient } from '@tanstack/react-query';
import {
  Activity,
  BookOpen,
  Check,
  ChevronRight,
  CircleAlert,
  Database,
  ExternalLink,
  Gauge,
  HeartPulse,
  Info,
  Library,
  Menu,
  MessageCircle,
  Network,
  RefreshCw,
  Send,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  UserCheck,
  X,
} from 'lucide-react';
import {
  getGetRagEvaluationQueryKey,
  getGetRagStatsQueryKey,
  getHealthCheckQueryKey,
  getListRagSourcesQueryKey,
  useGetRagEvaluation,
  useGetRagStats,
  useHealthCheck,
  useListRagSources,
  useRunRagEvaluation,
} from '@workspace/api-client-react';
import type {
  Citation,
  EvaluationReport,
  EvaluationSummary,
  RagAnswer,
  RagStats,
} from '@workspace/api-client-react';
import { ErrorBoundary } from '@/components/error-boundary';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import NotFound from '@/pages/not-found';
import { Link, Route, Router as WouterRouter, Switch, useLocation } from 'wouter';

const queryClient = new QueryClient();

const DOCTOR_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI2YTg2Y2EwZjk5OGMyNjk5ZjhhZjIyMTQiLCJwZXJzb25hIjoiZG9jdG9yIiwiZXhwIjoxNzg3ODIzMjQ3fQ.9oaalNy3MDbN6LD3SKZwyTSd709Q00DkHCrAbfzBrZ0";
const MOTHER_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI2YTg2ZDE4NzU5MzgzNTRmMTcwOWZlMzQiLCJwZXJzb25hIjoibW90aGVyIiwiZXhwIjoxNzg3ODI1MTU5fQ.HOdYzasjgON08FxjusMGFhedpYIS7wM6Zu_Nmjmtbo4";

const suggestedQuestions = {
  mother: [
    { text: 'ما علامات الخطر التي يجب الانتباه لها بعد الولادة؟', label: 'رعاية ما بعد الولادة' },
    { text: 'كم أحتاج من الحديد أثناء الحمل؟', label: 'التغذية' },
    { text: 'هل يمكنني الرضاعة الطبيعية وأنا مصابة بالزكام؟', label: 'الرضاعة الطبيعية' },
  ],
  doctor: [
    { text: 'What are the clinical guidelines for postpartum hemorrhage?', label: 'Clinical Guidelines' },
    { text: 'Iron supplementation dosage for second trimester?', label: 'Nutrition' },
    { text: 'Safe antibiotics for lactating mothers with URI?', label: 'Pharmacology' },
  ]
};

type ChatItem = { id: number; question: string; answer?: RagAnswer };

function formatDate(value?: string) {
  if (!value) return 'Not available';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

function formatPercent(value?: number) {
  if (value === undefined || value === null) return '—';
  return `${Math.round(value * 100)}%`;
}

function AppMark({ small = false }: { small?: boolean }) {
  return (
    <div className={`brand-mark ${small ? 'small-mark' : ''}`} aria-hidden="true">
      <HeartPulse size={small ? 16 : 19} strokeWidth={2.3} color="#ffffff" />
    </div>
  );
}

function Sidebar({ mobileOpen, onClose, persona, onTogglePersona }: { mobileOpen: boolean; onClose: () => void; persona: 'mother' | 'doctor', onTogglePersona: (p: 'mother' | 'doctor') => void }) {
  const [location] = useLocation();
  const [userName, setUserName] = useState<string>('Loading...');

  useEffect(() => {
    const fetchUserProfile = async () => {
      try {
        const token = persona === 'doctor' ? DOCTOR_TOKEN : MOTHER_TOKEN;
        const res = await fetch('http://127.0.0.1:8000/api/v1/auth/me', {
          headers: { 'Authorization': `Bearer ${token}` }
        });

        if (res.ok) {
          const result = await res.json();
          setUserName(result.data?.full_name || 'User');
        } else {
          setUserName(persona === 'doctor' ? 'Dr. Sameh Ahmed' : 'Nada Hassan');
        }
      } catch (err) {
        setUserName(persona === 'doctor' ? 'Dr. Sameh Ahmed' : 'Nada Hassan');
      }
    };
    fetchUserProfile();
  }, [persona]);

  const { data: health } = useHealthCheck({
    query: { queryKey: getHealthCheckQueryKey(), refetchInterval: 30000 },
  });
  const healthLabel = health?.status === 'ok' || health?.status === 'healthy' ? 'Service connected' : 'Checking service';
  const navItems = [
    { href: '/', label: 'Evidence chat', icon: MessageCircle },
    { href: '/dashboard', label: 'Evaluation dashboard', icon: Gauge },
    { href: '/sources', label: 'Source library', icon: Library },
  ];

  return (
    <aside className={`sidebar ${mobileOpen ? 'mobile-sidebar-open' : ''}`}>
      <div className="brand">
        <AppMark />
        <div>
          <span className="brand-name" style={{ color: '#ffffff' }}>Maternal AI Chat</span>
          <span className="brand-sub" style={{ color: '#cbd5e1' }}>Evidence companion</span>
        </div>
        {mobileOpen && <button className="icon-button mobile-close" onClick={onClose}><X size={16} /></button>}
      </div>

      <div style={{ padding: '12px 16px', margin: '8px 16px', background: 'rgba(137, 205, 187, 0.15)', borderRadius: '8px', cursor: 'pointer' }} onClick={() => onTogglePersona(persona === 'mother' ? 'doctor' : 'mother')}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', fontWeight: 'bold', color: '#ffffff', marginBottom: '4px' }}>
          <UserCheck size={16} style={{ color: '#4ade80' }} />
          <span>{userName}</span>
        </div>
        <div style={{ fontSize: '11px', color: '#94a3b8', display: 'flex', justifyContent: 'space-between' }}>
          <span>Role: {persona.toUpperCase()}</span>
          <span style={{ textDecoration: 'underline' }}>Switch Role</span>
        </div>
      </div>

      <div className="nav-label">Workspace</div>
      <nav className="nav-list">
        {navItems.map(({ href, label, icon: Icon }) => (
          <Link key={href} href={href} className={`nav-item ${location === href ? 'active' : ''}`} onClick={onClose}>
            <Icon size={16} strokeWidth={1.8} /><span>{label}</span>
          </Link>
        ))}
      </nav>
      <div className="sidebar-spacer" />
      <div className="side-footer"><span className="status-dot" style={{ display: 'inline-block', marginRight: 7 }} />{healthLabel}</div>
    </aside>
  );
}

function MobileTopbar({ onMenu }: { onMenu: () => void }) {
  const [location] = useLocation();
  return (
    <div className="mobile-topbar">
      <button className="icon-button" onClick={onMenu} aria-label="Open navigation"><Menu size={18} /></button>
      <div className="mobile-brand"><AppMark small /> <span style={{ color: '#ffffff' }}>Maternal AI Chat</span></div>
      <div className="mobile-links">
        <Link href="/" className={location === '/' ? 'active' : ''}><MessageCircle size={17} /></Link>
        <Link href="/dashboard" className={location === '/dashboard' ? 'active' : ''}><Gauge size={17} /></Link>
      </div>
    </div>
  );
}

function Shell({ children, persona, onTogglePersona }: { children: ReactNode; persona: 'mother' | 'doctor'; onTogglePersona: (p: 'mother'|'doctor')=>void }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  return (
    <div className="app-shell">
      <MobileTopbar onMenu={() => setMobileOpen(true)} />
      {mobileOpen && <div className="mobile-overlay" onClick={() => setMobileOpen(false)} aria-hidden="true" />}
      <Sidebar mobileOpen={mobileOpen} onClose={() => setMobileOpen(false)} persona={persona} onTogglePersona={onTogglePersona} />
      <main className="main">{children}</main>
    </div>
  );
}

function CitationCard({ citation }: { citation: Citation }) {
  return (
    <a className="citation" href={citation.url} target="_blank" rel="noreferrer">
      <div className="citation-head"><span className="citation-title">{citation.sourceTitle}</span><span className="citation-score">{Math.round(citation.relevance * 100)}% match</span></div>
      <div className="citation-meta">{citation.publisher} · {citation.year} · {citation.chunkId}</div>
      {citation.excerpt && <p className="citation-excerpt">“{citation.excerpt}”</p>}
    </a>
  );
}

function AnswerCard({ answer }: { answer: RagAnswer }) {
  const direction = answer.language === 'ar' ? 'rtl' : 'ltr';
  return (
    <div className="assistant-message">
      <div className="assistant-mark" aria-hidden="true"><Sparkles size={14} /></div>
      <div className="answer-body">
        <div className="answer-label">Evidence assistant · {answer.language === 'ar' ? 'العربية' : 'English'}</div>

        {answer.abstained ? (
          <div className="abstention" dir={direction}>
            {answer.answer}
          </div>
        ) : (
          <div className="answer-text markdown-body" dir={direction} style={{ whiteSpace: 'normal' }}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {answer.answer}
            </ReactMarkdown>
          </div>
        )}

        {answer.safetyNote && <div className="safety-note" dir={direction}><strong>Safety note</strong>{answer.safetyNote}</div>}
        <div className="trace-row">
          <span><Network size={12} /> {answer.trace?.method ?? 'hybrid'}</span>
          <span><Activity size={12} /> {answer.trace?.latencyMs ?? 0} ms</span>
          <span className={`confidence-${answer.trace?.confidence ?? 'high'}`}><ShieldCheck size={12} /> {answer.trace?.confidence ?? 'high'} confidence</span>
          <span><Database size={12} /> {(answer.trace?.indexedChunks ?? 100).toLocaleString()} chunks indexed</span>
        </div>
        {answer.citations && answer.citations.length > 0 && <div className="citation-list">{answer.citations.map((citation, idx) => <CitationCard key={idx} citation={citation} />)}</div>}
      </div>
    </div>
  );
}

function ChatPage({ persona }: { persona: 'mother' | 'doctor' }) {
  const [question, setQuestion] = useState('');
  const [items, setItems] = useState<ChatItem[]>([]);
  const [isPending, setIsPending] = useState(false);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const token = persona === 'doctor' ? DOCTOR_TOKEN : MOTHER_TOKEN;
        const res = await fetch('http://127.0.0.1:8000/api/v1/chat/history', {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        if (res.ok) {
          const data = await res.json();
          setItems(data.data?.messages || data.messages || []);
        } else {
          setItems([]);
        }
      } catch (err) {
        setItems([]);
      }
    };
    fetchHistory();
    setQuestion('');
  }, [persona]);

  const clearHistory = async () => {
    try {
      const token = persona === 'doctor' ? DOCTOR_TOKEN : MOTHER_TOKEN;
      const res = await fetch('http://127.0.0.1:8000/api/v1/chat/history', {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        setItems([]);
      }
    } catch (err) {
      console.error('Failed to clear history', err);
    }
  };

  const submit = (value = question) => {
    const clean = value.trim();
    if (clean.length < 3 || isPending) return;
    const id = Date.now();
    setQuestion('');
    setItems((current) => [...current, { id, question: clean }]);
    setIsPending(true);

    const fetchAnswer = async () => {
      try {
        const token = persona === 'doctor' ? DOCTOR_TOKEN : MOTHER_TOKEN;

        const resAsk = await fetch('http://127.0.0.1:8000/api/v1/chat/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
          body: JSON.stringify({
            question: clean,
            language: persona === 'doctor' ? 'en' : 'ar',
            persona: persona
          })
        });
        const answer = await resAsk.json();
        setItems((current) => current.map((item) => item.id === id ? { ...item, answer } : item));

        await fetch('http://127.0.0.1:8000/api/v1/chat/message', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
          body: JSON.stringify({ question: clean, answer: answer.answer })
        });

      } catch (err) {
        console.error('Error:', err);
      } finally {
        setIsPending(false);
      }
    };

    fetchAnswer();
  };

  const hasConversation = items.length > 0;
  const isDoctor = persona === 'doctor';

  return (
    <section className="chat-page">
      <header className="chat-header">
        <div className="chat-header-left">
          <div className="chat-orb"><Stethoscope size={19} /></div>
          <div>
            <div className="chat-title">{isDoctor ? 'Clinical Evidence Assistant' : 'المساعد الطبي للأمومة'}</div>
            <div className="chat-caption">{isDoctor ? 'Access verified medical guidelines instantly' : 'إجابات طبية موثوقة ومبسطة لرعايتك'}</div>
          </div>
        </div>
        <div className="language-toggle">
          <button
            className="lang-button"
            onClick={clearHistory}
            style={{ color: '#f87171', fontWeight: 600 }}
          >
            Clear History
          </button>
        </div>
      </header>
      <div className="chat-scroll">
        <div className="chat-inner">
          {!hasConversation && <div className="welcome">
            <div className="welcome-kicker">{isDoctor ? 'Clinical Reference' : 'بداية آمنة وموثوقة'}</div>
            <h1>{isDoctor ? 'Search medical guidelines & literature.' : 'إجابات موثوقة لأسئلتك حول الأمومة.'}</h1>
            <p>Active Profile: <strong>{isDoctor ? 'Doctor (Clinical English)' : 'Mother (Simple Arabic)'}</strong></p>
            <div className="suggestions">
              {suggestedQuestions[persona].map((item, index) => (
                <button key={index} className="suggestion" onClick={() => submit(item.text)}>
                  <small>{item.label}</small>{item.text}
                </button>
              ))}
            </div>
          </div>}
          {hasConversation && <div className="thread">
            {items.map((item) => <div key={item.id}>
              <div className="user-message"><div className="user-bubble" dir={isDoctor ? 'ltr' : 'rtl'}>{item.question}</div></div>
              {item.answer ? <div style={{ marginTop: 19 }}><AnswerCard answer={item.answer} /></div> : isPending && item.id === items[items.length - 1].id ? <div className="assistant-message" style={{ marginTop: 19 }}><div className="assistant-mark"><Sparkles size={14} /></div><div><div className="answer-label">Searching...</div><div className="skeleton" style={{ height: 13, width: '88%', marginBottom: 9 }} /><div className="skeleton" style={{ height: 13, width: '62%' }} /></div></div> : null}
            </div>)}
          </div>}
        </div>
      </div>
      <div className="composer-wrap">
        <form className="composer" onSubmit={(e) => { e.preventDefault(); submit(); }} dir={isDoctor ? 'ltr' : 'rtl'}>
          <div className="composer-box">
            <textarea value={question} onChange={(e) => setQuestion(e.target.value)} placeholder={isDoctor ? 'Ask clinical maternal health question...' : 'اكتبي سؤالك هنا...'} rows={1} />
            <button className="send-button" type="submit" disabled={question.trim().length < 3 || isPending}><Send size={15} /></button>
          </div>
        </form>
      </div>
    </section>
  );
}

function DashboardPage() {
  const statsQuery = useGetRagStats({ query: { queryKey: getGetRagStatsQueryKey() } });
  const evaluationQuery = useGetRagEvaluation({ query: { queryKey: getGetRagEvaluationQueryKey() } });
  const runEvaluation = useRunRagEvaluation();
  const client = useQueryClient();
  const [runMessage, setRunMessage] = useState('');
  const stats = statsQuery.data as RagStats | undefined;
  const report = evaluationQuery.data as EvaluationReport | undefined;
  const summary = report?.summary ?? stats?.evaluation;

  const run = () => {
    setRunMessage('');
    runEvaluation.mutate(undefined, {
      onSuccess: (next) => {
        client.setQueryData(getGetRagEvaluationQueryKey(), next);
        setRunMessage(`Evaluation complete · ${next.summary.total} questions checked`);
      },
      onError: () => setRunMessage('The evaluation could not be completed. Try again when the service is available.'),
    });
  };

  const results = report?.results ?? [];

  return (
    <div className="page-frame">
      <div className="topline">
        <div>
          <div className="eyebrow">Retrieval observatory</div>
          <h1 className="page-title">Evaluation dashboard</h1>
          <p className="page-lead">A transparent view of the evidence index behind every answer.</p>
        </div>
        <div className="toolbar">
          <button className="primary-button" onClick={run} disabled={runEvaluation.isPending}>
            <RefreshCw size={14} className={runEvaluation.isPending ? 'animate-spin' : ''} />
            {runEvaluation.isPending ? 'Running evaluation' : 'Run evaluation'}
          </button>
        </div>
      </div>

      {runMessage && <div className="error-banner success-banner">{runMessage}</div>}

      <div className="metric-grid">
        <div className="card metric"><div className="metric-label">Indexed sources</div><div className="metric-value">{stats?.indexedSources || '—'}</div><div className="metric-sub">public-health publishers</div></div>
        <div className="card metric"><div className="metric-label">Indexed chunks</div><div className="metric-value">{stats?.indexedChunks.toLocaleString() || '—'}</div><div className="metric-sub">retrievable passages</div></div>
        <div className="card metric"><div className="metric-label">Top-1 accuracy</div><div className="metric-value teal">{summary ? formatPercent(summary.top1Accuracy) : '—'}</div><div className="metric-sub">{summary ? `${summary.total} evaluation questions` : 'Awaiting first run'}</div></div>
        <div className="card metric"><div className="metric-label">Median feeling</div><div className="metric-value">{summary ? `${Math.round(summary.averageLatencyMs)} ms` : '—'}</div><div className="metric-sub">average retrieval latency</div></div>
      </div>
    </div>
  );
}

function SourcesPage() {
  const sourcesQuery = useListRagSources({ query: { queryKey: getListRagSourcesQueryKey() } });
  const sources = sourcesQuery.data ?? [];
  const reviewedCount = sources.filter((s) => s.reviewed).length;

  return (
    <div className="page-frame">
      <div className="topline">
        <div>
          <div className="eyebrow">What the assistant knows</div>
          <h1 className="page-title">Source library</h1>
          <p className="page-lead">The public guidance currently indexed for retrieval.</p>
        </div>
        <div className="status-pill"><span className="status-dot" />{reviewedCount} of {sources.length} reviewed</div>
      </div>

      <div className="source-grid">
        {sources.map((source) => (
          <article className="card source-card" key={source.id}>
            <div className="source-top">
              <div className="source-title">{source.title}</div>
            </div>
            <div className="source-publisher">{source.publisher}</div>
            <div className="source-bottom">
              <span>{source.year} · {source.chunkCount.toLocaleString()} chunks</span>
              <a className="external-link" href={source.url} target="_blank" rel="noreferrer">Original <ExternalLink size={11} /></a>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}

function Router({ persona, onTogglePersona }: { persona: 'mother' | 'doctor', onTogglePersona: (p:'mother'|'doctor')=>void }) {
  return (
    <ErrorBoundary resetKey={useLocation()[0] + persona}>
      <Switch>
        <Route path="/" component={() => <Shell persona={persona} onTogglePersona={onTogglePersona}><ChatPage persona={persona} /></Shell>} />
        <Route path="/dashboard" component={() => <Shell persona={persona} onTogglePersona={onTogglePersona}><DashboardPage /></Shell>} />
        <Route path="/sources" component={() => <Shell persona={persona} onTogglePersona={onTogglePersona}><SourcesPage /></Shell>} />
        <Route component={NotFound} />
      </Switch>
    </ErrorBoundary>
  );
}

export function App() {
  const [persona, setPersona] = useState<'mother' | 'doctor'>('mother');

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}>
          <Router persona={persona} onTogglePersona={setPersona} />
        </WouterRouter>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
