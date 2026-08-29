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
}

export interface Experiment {
  id: string;
  name: string;
  mode: "simple" | "advanced";
  domain: string;
  test_type: string;
  group_col: string | null;
  metric_col: string | null;
  results: ExperimentResult;
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
}

export type Persona = "business" | "learner";

export interface User {
  id: string;
  email: string;
  full_name: string | null;
  persona: Persona;
}
