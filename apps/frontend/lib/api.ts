import type { AlertRow, DocumentRow, TurnResponse } from "./types";

// El backend corre en :8000. En el navegador usamos localhost; configurable
// vía NEXT_PUBLIC_API_URL para el despliegue.
export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const API = API_BASE;

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const api = {
  async turn(text: string, conversationId: string | null): Promise<TurnResponse> {
    return json(
      await fetch(`${API}/conversation/turn`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, conversation_id: conversationId }),
      })
    );
  },

  async closeConversation(id: string): Promise<unknown> {
    return json(await fetch(`${API}/conversation/${id}/close`, { method: "POST" }));
  },

  async listDocuments(): Promise<DocumentRow[]> {
    return json(await fetch(`${API}/knowledge/documents`));
  },

  async uploadDocument(file: File, procedure?: string): Promise<unknown> {
    const form = new FormData();
    form.append("file", file);
    if (procedure) form.append("procedure", procedure);
    return json(
      await fetch(`${API}/knowledge/documents`, { method: "POST", body: form })
    );
  },

  async deleteDocument(id: string): Promise<unknown> {
    return json(await fetch(`${API}/knowledge/documents/${id}`, { method: "DELETE" }));
  },

  async listAlerts(): Promise<AlertRow[]> {
    return json(await fetch(`${API}/console/alerts`));
  },

  async attendAlert(id: string): Promise<unknown> {
    return json(await fetch(`${API}/console/alerts/${id}/attend`, { method: "POST" }));
  },
};
