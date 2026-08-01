// ~5 tipos escritos a mano que espejan el backend (ADR-008).
// No hay codegen: se mantienen sincronizados manualmente con app/schemas.py.

export type RiskLevel = "NORMAL" | "ALTO" | "CRÍTICO";

export interface Source {
  document: string;
  page: number;
  chunk_id: string;
  score: number;
}

export interface TurnResponse {
  conversation_id: string;
  turn_id: string;
  response: string;
  risk_level: RiskLevel;
  triggered_rules: string[];
  symptoms: Record<string, unknown>;
  sources: Source[];
  critical_override: boolean;
  alert_id: string | null;
}

export interface DocumentRow {
  id: string;
  filename: string;
  status: string;
  n_chunks: number;
  created_at: string;
  updated_at: string;
}

export interface AlertRow {
  id: string;
  conversation_id: string;
  risk_level: RiskLevel;
  triggered_rules: string[];
  symptoms: Record<string, unknown>;
  transcript: string;
  status: string;
  created_at: string;
}
