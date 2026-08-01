"use client";

import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import type { AlertRow, DocumentRow, TurnResponse } from "../lib/types";

type Tab = "call" | "documents" | "alerts";

export default function Home() {
  const [tab, setTab] = useState<Tab>("call");
  return (
    <div className="container">
      <h1>Agente Clínico Post-operatorio — Consola</h1>
      <p className="muted">
        Scaffold inicial. Modo texto para probar el bucle del turno (RAG +
        decisión + resumen) sin audio. La voz (Pipecat) se activa con credenciales.
      </p>
      <div className="tabs">
        <button className={`tab ${tab === "call" ? "active" : ""}`} onClick={() => setTab("call")}>
          Llamada (texto)
        </button>
        <button className={`tab ${tab === "documents" ? "active" : ""}`} onClick={() => setTab("documents")}>
          Conocimiento
        </button>
        <button className={`tab ${tab === "alerts" ? "active" : ""}`} onClick={() => setTab("alerts")}>
          Alertas
        </button>
      </div>
      {tab === "call" && <CallPanel />}
      {tab === "documents" && <DocumentsPanel />}
      {tab === "alerts" && <AlertsPanel />}
    </div>
  );
}

function RiskBadge({ level }: { level: string }) {
  return <span className={`badge ${level}`}>{level}</span>;
}

function CallPanel() {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<{ role: "user" | "agent"; text: string; meta?: TurnResponse }[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  async function send() {
    if (!input.trim()) return;
    const text = input.trim();
    setInput("");
    setMessages((m) => [...m, { role: "user", text }]);
    setLoading(true);
    try {
      const res = await api.turn(text, conversationId);
      setConversationId(res.conversation_id);
      setMessages((m) => [...m, { role: "agent", text: res.response, meta: res }]);
    } catch (e) {
      setMessages((m) => [...m, { role: "agent", text: `Error: ${String(e)}` }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="card chat">
        {messages.length === 0 && (
          <p className="muted">
            Escribe como si fueras el paciente. Ej: "Tengo dolor 9 y la pastilla no
            me hace nada" o "Estoy sangrando mucho".
          </p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`bubble ${m.role}`}>
            <div>{m.text}</div>
            {m.meta && (
              <div className="muted" style={{ marginTop: 6 }}>
                <RiskBadge level={m.meta.risk_level} />{" "}
                {m.meta.triggered_rules.length > 0 && `reglas: ${m.meta.triggered_rules.join(", ")}`}
                {m.meta.critical_override && " · guion de seguridad (LLM descartado)"}
                {m.meta.sources.length > 0 && (
                  <div>fuentes: {m.meta.sources.map((s) => `${s.document} p.${s.page} (${s.score})`).join("; ")}</div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="row">
        <textarea
          style={{ flex: 1 }}
          rows={2}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && (e.preventDefault(), send())}
          placeholder="Mensaje del paciente…"
        />
        <button onClick={send} disabled={loading}>
          {loading ? "…" : "Enviar"}
        </button>
      </div>
    </div>
  );
}

function DocumentsPanel() {
  const [docs, setDocs] = useState<DocumentRow[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);

  async function refresh() {
    setDocs(await api.listDocuments());
  }
  useEffect(() => {
    refresh();
  }, []);

  async function upload() {
    const file = fileRef.current?.files?.[0];
    if (!file) return;
    await api.uploadDocument(file);
    if (fileRef.current) fileRef.current.value = "";
    await refresh();
  }

  return (
    <div>
      <h2>Documentos indexados</h2>
      <div className="card row">
        <input type="file" ref={fileRef} accept=".pdf,.md,.txt" />
        <button onClick={upload}>Subir e indexar</button>
      </div>
      {docs.map((d) => (
        <div key={d.id} className="card row" style={{ justifyContent: "space-between" }}>
          <div>
            <strong>{d.filename}</strong>
            <div className="muted">
              {d.n_chunks} chunks · {d.status} · {new Date(d.updated_at).toLocaleString()}
            </div>
          </div>
          <button
            className="danger"
            onClick={async () => {
              await api.deleteDocument(d.id);
              await refresh();
            }}
          >
            Eliminar
          </button>
        </div>
      ))}
      {docs.length === 0 && <p className="muted">Sin documentos. Corre `python seed.py` o sube uno.</p>}
    </div>
  );
}

function AlertsPanel() {
  const [alerts, setAlerts] = useState<AlertRow[]>([]);

  async function refresh() {
    setAlerts(await api.listAlerts());
  }
  useEffect(() => {
    refresh();
    // Polling cada 4s (sin WebSocket, ADR §3.6).
    const t = setInterval(refresh, 4000);
    return () => clearInterval(t);
  }, []);

  return (
    <div>
      <h2>Alertas (polling cada 4s)</h2>
      {alerts.map((a) => (
        <div key={a.id} className="card">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <div>
              <RiskBadge level={a.risk_level} /> {a.status === "attended" ? "· atendida" : ""}
              <div className="muted">reglas: {a.triggered_rules.join(", ")}</div>
              <div className="muted">"{a.transcript}"</div>
            </div>
            {a.status !== "attended" && (
              <button
                className="secondary"
                onClick={async () => {
                  await api.attendAlert(a.id);
                  await refresh();
                }}
              >
                Marcar atendida
              </button>
            )}
          </div>
        </div>
      ))}
      {alerts.length === 0 && <p className="muted">Sin alertas todavía.</p>}
    </div>
  );
}
