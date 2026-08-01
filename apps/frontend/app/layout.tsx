import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "Consola — Agente Clínico Post-operatorio",
  description: "Consola de administración del agente de voz clínico",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  );
}
