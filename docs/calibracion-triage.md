# Calibración del motor de decisión contra el ground truth

Los umbrales de `apps/backend/app/decision/thresholds.yaml` no salen de la intuición
clínica ni de los documentos de ejemplo: se derivaron de las **160 trayectorias** de
`trayectorias_postop_silver.xlsx` contrastadas con el `label_ground_truth` del dataset
oficial. Este documento explica cómo, para que el número del README sea auditable.

Reproducible con:

```bash
.venv/bin/python scripts/load_dataset.py                    # genera el fixture
cd apps/backend && ../../.venv/bin/python -m pytest app/tests/test_triage_from_trajectories.py -q
```

---

## El punto de partida

El dataset describe cada llamada con seis variables, que son exactamente las seis que
el agente de referencia pregunta en su guion canónico:

| Variable | Rango observado |
|---|---|
| `dolor_nrs` | 0–9 |
| `fiebre_c` | 36.2–39.5 |
| `movilidad` | `normal` · `limitada_esperada` · `incapacitante_nueva` |
| `herida` | `normal` · `eritema_leve` · `secrecion_purulenta` |
| `apetito` | `normal` · `levemente_disminuido` · `muy_disminuido` |
| `sueno` | `normal` · `levemente_alterado` · `muy_alterado` |

La distribución de criticidad está desbalanceada, como en la realidad: **123 verde,
25 amarillo, 12 rojo**.

Conviene saber de dónde salen los rojos, aunque el agente no pueda verlo: el label se
determina por `arquetipo_trayectoria = complicacion_real` **y** `dia_postop ∈ {7, 14}`.
Los 24 casos de complicación real en días 1 y 3 son la complicación aún no manifestada,
y por eso se reparten entre verde y amarillo. El agente solo ve los síntomas, así que la
calibración tiene que reconstruir esa frontera a partir de las seis variables.

---

## Rojo: separación perfecta

```
rojo  ⟺  fiebre_c ≥ 38.0
      ∨  herida = secrecion_purulenta
      ∨  movilidad = incapacitante_nueva
      ∨  dolor_nrs ≥ 8
```

**12/12 de recall, 0 falsos positivos** sobre los 148 casos no-rojos. Los cuatro
predicados son disyuntivos a propósito: cada uno es un motivo suficiente e independiente
para que alguien mire al paciente hoy.

El umbral de fiebre es el punto delicado. El máximo de un caso no-rojo es 37.9 y el
mínimo de un rojo es 37.9 también, así que el corte en 37.9 mete un falso positivo y uno
de amarillo; el corte en **38.0** captura 11 de los 12 rojos sin ningún falso positivo, y
el rojo que se escapa por fiebre lo recogen las otras tres condiciones. De ahí la
disyunción: ninguna condición por separado alcanza, las cuatro juntas dan separación
perfecta.

Comparación de candidatos, medida sobre los 160 casos:

| Regla | Recall rojo | FP verde | FP amarillo |
|---|---:|---:|---:|
| `fiebre ≥ 38.0` sola | 11/12 | 0 | 0 |
| `fiebre ≥ 37.9` sola | 12/12 | 1 | 1 |
| `herida purulenta` sola | 3/12 | 0 | 0 |
| `movilidad incapacitante` sola | 4/12 | 0 | 0 |
| `dolor ≥ 8` sola | 2/12 | 0 | 0 |
| **Disyunción de las cuatro** | **12/12** | **0** | **0** |

---

## Amarillo: score aditivo

```
score = [dolor_nrs ≥ 5] + [fiebre_c ≥ 37.3] + [herida = eritema_leve]
      + [apetito = muy_disminuido] + [sueno = muy_alterado]

amarillo  ⟺  score ≥ 2
```

**25/25 de recall, 8/123 falsos positivos** sobre verde (93.5% de especificidad).

Ninguna señal aislada sirve: la mejor individual (`herida = eritema_leve`) da 19/25 con
11 falsos positivos. La lógica clínica del score es que el amarillo del dataset es
precisamente el cuadro donde *varias cosas menores coinciden*, no donde una se dispara.
Bajar el corte a 1 mantendría el recall pero subiría los falsos positivos a 44; subirlo a
3 los dejaría en 0 pero perdería 6 amarillos, y perder un amarillo es un falso negativo.

---

## Resultado

```
  real \ predicho   verde  amarillo   rojo
  verde               115         8      0
  amarillo              0        25      0
  rojo                  0         0     12

  exactitud 152/160 (95.0%)
```

**Cero falsos negativos en ambos niveles.** Los 8 errores son todos verdes escalados a
amarillo, es decir, falsos positivos. Es la asimetría que pide la rúbrica: en salud, no
alertar cuando había que alertar es la falla catastrófica, y un seguimiento de más cuesta
una llamada.

---

## Qué mide esto y qué no

Este test alimenta las **reglas** con el cuadro clínico real, como si la extracción
hubiera sido perfecta. Aísla a propósito el error del motor del error del extractor: si
mañana el agente falla un caso, este test dice si la culpa es de los umbrales o de la
comprensión del lenguaje. La evaluación de la cadena completa —conversación ruidosa
incluida— es `scripts/run_dataset_eval.py`, que corre los mismos 160 casos en sus dos
capas a través del agente entero.

## Lo que se conserva del motor anterior

Las seis banderas de emergencia —sangrado abundante, dificultad respiratoria, pérdida de
consciencia, dolor torácico, estado mental alterado y convulsión— **no aparecen en el
dataset**, porque describe el curso postoperatorio esperable y no urgencias vitales. Se
conservan íntegras porque sí aparecen en los escenarios que el jurado interpreta en vivo,
y escalan por una vía distinta: `emergencia_123` en vez de `enfermeria_prioritaria`. Son
dos guiones con tono distinto, porque alarmar de más a alguien que lo que necesita es una
consulta hoy también es un daño.

## La política de incertidumbre

Un slot en `None` nunca reduce el riesgo: los predicados son igualdad explícita o
`is True`, jamás `not x`. Además, al cerrar la llamada (`evaluate(..., final=True)`) se
fuerza ALTO con la regla `informacion_insuficiente` si se respondió menos de la mitad del
guion, o si un slot capaz de disparar rojo quedó sin respuesta habiendo ya una señal
amarilla. Es un falso positivo comprado deliberadamente para cerrar una vía de falso
negativo: el paciente evasivo o minimizador no puede irse de la llamada como verde por no
haber contestado.
