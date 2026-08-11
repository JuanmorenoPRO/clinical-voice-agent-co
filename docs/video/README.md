# Video — entregable 04

Demo funcional con grabación de pantalla, más las dos preguntas de cierre respondidas
frente a cámara.

## Enlace

**[Ver el video en Google Drive](https://drive.google.com/file/d/1GCVi5ZZR75QMm16tJiz4r79IkLU49BbG/view?usp=drive_link)**

| Campo | Valor |
|---|---|
| URL | https://drive.google.com/file/d/1GCVi5ZZR75QMm16tJiz4r79IkLU49BbG/view?usp=drive_link |
| Duración | 31 min 33 s |
| Grabado el | 10 de agosto de 2026 |
| Commit del repositorio que se demuestra | [`a96d788`](https://github.com/JuanmorenoPRO/clinical_assistant/commit/a96d788) |

El commit importa: la rúbrica comprueba que el demo **corresponda al repositorio
entregado**, y un demo que no corresponde levanta bandera de integridad.

## Qué contiene

| Bloque | Qué se ve |
|---|---|
| Problema | Para qué existe el agente |
| Arranque y modelo | `/health` declarando `llama-3.3-70b-versatile` en Groq — compuerta **G3** |
| La llamada | Conversación de voz real: guion, jerga colombiana, silencios, inyección de prompt, petición fuera de misión — compuerta **G4** |
| Escalamiento | Bandera roja, guion de seguridad, alerta persistida, traza del turno y resumen de cierre |
| Conocimiento vivo | Subir un documento, usarlo con cita, borrarlo y comprobar que lo olvida — compuerta **G5** |
| Cuándo dice «no sé» | Pregunta del corpus con cita, pregunta inventada con abstención |
| Métricas | `scripts/report_metrics.py` en vivo |
| Pregunta 1 | El problema, la solución y su valor frente a las alternativas |
| Pregunta 2 | La decisión técnica: alternativas, riesgos y qué haría con dos semanas más |

## Por qué el archivo no está en el repositorio

El `.mov` pesa 768 MB: supera con creces el límite de 100 MB por archivo de GitHub y
ensuciaría el historial para siempre. El video se aloja en Drive y aquí queda el
enlace; los `.mp4` y `.mov` de esta carpeta están en `.gitignore`.
