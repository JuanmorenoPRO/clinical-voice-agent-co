# Capturas del demo

Las cuatro imágenes que referencia [`INFORME.md`](../../INFORME.md) § 9. Los nombres
son fijos: el informe las enlaza por ruta.

| Archivo | Qué debe verse |
|---|---|
| `01-alerta.png` | La consola con una alerta creada tras un escalamiento: nivel de riesgo, la regla que la disparó (p. ej. `herida_purulenta`) y el paciente asociado. |
| `02-traza-turno.png` | La respuesta de `GET /console/conversations/{id}` para un turno: síntomas extraídos, reglas disparadas, fragmentos citados con documento y página, tokens y latencia. Es la evidencia de que las métricas del README salen de estas filas. |
| `03-resumen.png` | El resumen de cierre: paciente, procedimiento, síntomas reportados, decisión turno a turno, referencias usadas y próximos pasos. |
| `04-conocimiento-vivo.png` | Tres momentos del ciclo G5, en una imagen o en un collage: el documento subido desde la consola, la respuesta del agente citándolo, y la abstención después de borrarlo. |

Formato: PNG, ancho mínimo 1.200 px para que el texto se lea al imprimir a PDF.

Al añadirlas, regenerar el HTML del informe:

```bash
pandoc INFORME.md -o informe-final.html --standalone --embed-resources --toc \
       --metadata title="Informe final — Agente de voz postoperatorio"
```
