export interface SampleRatioMismatch {
  passed: boolean;
  p_value: number;
  statistic?: number;
  observed_ratio: number | null;
  expected_ratio: number;
  n_control?: number;
  n_treatment?: number;
}

export interface GuardrailResult {
  test_name: string;
  p_value: number;
  effect_size: number;
  uplift_percentage: number;
  is_significant: boolean;
  metric: string;
  mean_control?: number;
  mean_treatment?: number;
  p_control?: number;
  p_treatment?: number;
}

export interface ExperimentResult {
  test_name: string;
  p_value: number;
  effect_size: number;
  uplift_percentage: number;
  is_significant: boolean;
  mean_control?: number;
  mean_treatment?: number;
  p_control?: number;
  p_treatment?: number;
  n_control?: number;
  n_treatment?: number;
  domain?: string;
  metric?: string;
  test_type?: string;
  ai_summary?: string;
  health_checks?: { sample_ratio_mismatch: SampleRatioMismatch };
  guardrails?: GuardrailResult[];
}

export type Decision = "shipped" | "rolled_back" | null;

export interface Experiment {
  id: string;
  name: string;
  mode: "simple" | "advanced";
  domain: string;
  test_type: string;
  hypothesis: string | null;
  group_col: string | null;
  metric_col: string | null;
  decision: Decision;
  results: ExperimentResult;
  created_at: string;
}

export interface Metric {
  id: string;
  name: string;
  description: string | null;
  column_name: string;
  is_guardrail: boolean;
  created_at: string;
}

export interface SampleDatasetSummary {
  key: string;
  name: string;
  description: string;
  group_col: string;
  metric_col: string;
  row_count: number;
}

export interface SampleDatasetDetail extends SampleDatasetSummary {
  rows: Record<string, unknown>[];
}

export interface MLRun {
  id: string;
  task_type: "train" | "uplift";
  status: "pending" | "running" | "done" | "failed";
  target_col: string;
  group_col: string | null;
  model_type: string;
  results: Record<string, unknown> | null;
  error_message: string | null;
  created_at: string;
}

export interface ChatSource {
  slug: string;
  title: string;
  similarity: number;
}

export interface ChatMessageResponse {
  session_id: string;
  role: string;
  content: string;
  sources: ChatSource[];
}

export interface ChatHistoryMessage {
  role: string;
  content: string;
  created_at: string;
  sources: ChatSource[] | null;
}

export interface ChatSessionHistory {
  session_id: string | null;
  messages: ChatHistoryMessage[];
}

export interface KBDocument {
  slug: string;
  title: string;
  content: string;
}

export type Persona = "business" | "learner";

export interface User {
  id: string;
  email: string;
  full_name: string | null;
  persona: Persona;
}

export interface PracticeFeedbackResponse {
  feedback: string;
  sources: ChatSource[];
}

export interface TrendPoint {
  week: string;
  count: number;
  significant: number;
}

export interface AnalyticsOverview {
  total_experiments: number;
  significant_count: number;
  significance_rate: number;
  experiments_this_week: number;
  test_type_breakdown: Record<string, number>;
  guardrail_failure_rate: number | null;
  trend: TrendPoint[];
  ai_summary: string;
  sources: ChatSource[];
}
