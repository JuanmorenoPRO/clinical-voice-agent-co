# Plantilla de turno

El `user_prompt` de cada turno se arma en `voice/conversation.py` con esta forma:

```
[si el paciente preguntó algo]
Evidencia recuperada de los documentos (úsala para fundamentar la respuesta;
NO respondas desde tu conocimiento interno):
<chunks recuperados por el RAG>

Paciente: <lo que dijo el paciente>
```

El modelo debe:
1. Extraer los síntomas mencionados → `sintomas`.
2. Redactar una respuesta breve y empática en español colombiano → `respuesta`,
   fundamentada solo en la evidencia si la pregunta es clínica. Varía el inicio de
   la respuesta y no reutilices una frase de empatía que ya usaste antes en esta
   conversación (revisa el historial).

⏳ 7 de agosto: añadir few-shots con el vocabulario coloquial del dataset real
(ADR-006) para robustecer la extracción.
