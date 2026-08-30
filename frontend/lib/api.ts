import type {
  AnalyticsOverview,
  ChatHistoryMessage,
  ChatMessageResponse,
  ChatSessionHistory,
  Decision,
  Experiment,
  ExperimentResult,
  KBDocument,
  Metric,
  MLRun,
  Persona,
  PracticeFeedbackResponse,
  SampleDatasetDetail,
  SampleDatasetSummary,
  User,
} from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const TOKEN_KEY = "abtesting_token";

// sessionStorage, not localStorage: business and learner are meant to be usable
// side by side in two tabs at once (they're genuinely separate accounts). localStorage
// is shared across every tab of the same origin, so logging into one account in one
// tab would silently overwrite the token the other tab is using underneath it —
// the other tab's UI keeps showing its own persona's chrome while its API calls
// start authenticating as the other account. sessionStorage is isolated per tab,
// which is exactly the isolation two concurrently-open accounts need.
export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.sessionStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  window.sessionStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  window.sessionStorage.removeItem(TOKEN_KEY);
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers = new Headers(options.headers);
  headers.set("Content-Type", "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const res = await fetch(`${API_URL}${path}`, { ...options, headers });

  if (!res.ok) {
    if (res.status === 429) {
      throw new ApiError(res.status, "Too many attempts — please wait a minute and try again.");
    }
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // ignore JSON parse failure, fall back to statusText
    }
    throw new ApiError(res.status, detail);
  }

  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

// --- Auth ---

export function signup(email: string, password: string, persona: Persona, fullName?: string) {
  return request<{ access_token: string }>("/api/auth/signup", {
    method: "POST",
    body: JSON.stringify({ email, password, full_name: fullName, persona }),
  });
}

export function login(email: string, password: string, persona: Persona) {
  return request<{ access_token: string }>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password, persona }),
  });
}

export function getMe() {
  return request<User>("/api/auth/me");
}

// --- Experiments ---

export interface SimpleTestPayload {
  name: string;
  metric_type: "conversion" | "continuous";
  domain?: string;
  hypothesis?: string;
  control_conversions?: number;
  control_total?: number;
  treatment_conversions?: number;
  treatment_total?: number;
  control_values?: number[];
  treatment_values?: number[];
}

export function runSimpleTest(payload: SimpleTestPayload) {
  return request<Experiment>("/api/experiments/simple", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export interface AdvancedTestPayload {
  name: string;
  domain?: string;
  hypothesis?: string;
  group_col: string;
  metric_col: string;
  test_type?: "auto" | "ttest" | "chi_square" | "mann_whitney";
  guardrail_cols?: string[];
  rows: Record<string, unknown>[];
}

export function runAdvancedTest(payload: AdvancedTestPayload) {
  return request<Experiment>("/api/experiments/advanced", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function listExperiments() {
  return request<Experiment[]>("/api/experiments");
}

export function getExperiment(id: string) {
  return request<Experiment>(`/api/experiments/${id}`);
}

export function updateExperimentDecision(id: string, decision: Decision) {
  return request<Experiment>(`/api/experiments/${id}/decision`, {
    method: "PATCH",
    body: JSON.stringify({ decision }),
  });
}

// --- Metrics catalog ---

export function listMetrics() {
  return request<Metric[]>("/api/metrics");
}

export function createMetric(payload: { name: string; description?: string; column_name: string; is_guardrail?: boolean }) {
  return request<Metric>("/api/metrics", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function deleteMetric(id: string) {
  return request<void>(`/api/metrics/${id}`, { method: "DELETE" });
}

// --- Datasets ---

export function listSampleDatasets() {
  return request<SampleDatasetSummary[]>("/api/datasets/samples");
}

export function getSampleDataset(key: string) {
  return request<SampleDatasetDetail>(`/api/datasets/samples/${key}`);
}

export function listGeneratorDomains() {
  return request<string[]>("/api/datasets/generator/domains");
}

export function generateDataset(domain: string, nSamples: number) {
  return request<{ domain: string; row_count: number; rows: Record<string, unknown>[]; truncated: boolean }>(
    "/api/datasets/generator/generate",
    { method: "POST", body: JSON.stringify({ domain, n_samples: nSamples }) }
  );
}

export function detectColumns(rows: Record<string, unknown>[]) {
  return request<Record<string, string[]>>("/api/datasets/detect-columns", {
    method: "POST",
    body: JSON.stringify(rows),
  });
}

// --- ML ---

export interface TrainModelPayload {
  experiment_id?: string;
  rows: Record<string, unknown>[];
  target_col: string;
  group_col?: string;
  model_type?: "auto" | "classification" | "regression";
  task?: "predictive" | "uplift";
}

export function trainModel(payload: TrainModelPayload) {
  return request<MLRun>("/api/ml/train", { method: "POST", body: JSON.stringify(payload) });
}

export function listMLRuns() {
  return request<MLRun[]>("/api/ml/runs");
}

export function getMLRun(id: string) {
  return request<MLRun>(`/api/ml/runs/${id}`);
}

export function predict(mlRunId: string, rows: Record<string, unknown>[]) {
  return request<{ predictions: number[] }>("/api/ml/predict", {
    method: "POST",
    body: JSON.stringify({ ml_run_id: mlRunId, rows }),
  });
}

// --- Chat ---

export function sendChatMessage(message: string, sessionId?: string, experimentId?: string) {
  return request<ChatMessageResponse>("/api/chat/message", {
    method: "POST",
    body: JSON.stringify({ message, session_id: sessionId, experiment_id: experimentId }),
  });
}

export function getChatHistory(sessionId: string) {
  return request<ChatHistoryMessage[]>(`/api/chat/sessions/${sessionId}/history`);
}

export function getLatestChatSession() {
  return request<ChatSessionHistory>("/api/chat/sessions/latest");
}

// --- Knowledge base ---

export function getKbDocument(slug: string) {
  return request<KBDocument>(`/api/kb/${slug}`);
}

// --- Practice Lab ---

export function submitPracticeFeedback(scenarioName: string, learnerConclusion: string, results: ExperimentResult) {
  return request<PracticeFeedbackResponse>("/api/practice/feedback", {
    method: "POST",
    body: JSON.stringify({ scenario_name: scenarioName, learner_conclusion: learnerConclusion, results }),
  });
}

// --- Analytics ---

export function getAnalyticsOverview() {
  return request<AnalyticsOverview>("/api/analytics/overview");
}
