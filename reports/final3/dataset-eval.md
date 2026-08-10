# Evaluación sobre los 160 casos del dataset

160 conversaciones reproducidas a través del agente completo (comprensión del lenguaje, acumulación de síntomas y decisión).


## capa2_ruidosa  (160 casos)

```
  real \ predicho   verde  amarillo   rojo
  verde                54       31       38
  amarillo              2       11       12
  rojo                  0        0       12

  exactitud 77/160 (48.1%)
```

- **Falsos negativos: 2** (de los cuales 0 en rojo, que es la falla catastrófica)
  - `caso_tray_pac_42_00015_3` esperaba **amarillo**, dijo *verde* · completitud 1.0
  - `caso_tray_pac_42_00027_3` esperaba **amarillo**, dijo *verde* · completitud 0.667

## Exactitud de la extracción, por slot

Compara lo que el agente entendió con el cuadro real de la trayectoria. Si un caso falla, esto dice si la culpa fue de entender al paciente o de los umbrales.

| Slot | Acierto | Sin extraer |
|---|---:|---:|
| dolor_nrs | 46% | 5% |
| movilidad | 71% | 16% |
| herida | 59% | 22% |
| apetito | 68% | 15% |
| sueno | 61% | 19% |

## Consumo por llamada

| Métrica | Valor |
|---|---:|
| Turnos por llamada | 6.9 |
| Invocaciones al modelo por llamada | 3.7 |
| Invocaciones al modelo por turno | 0.54 |
| Tokens de entrada por llamada | 2310 |
| Tokens de salida por llamada | 122 |
| Latencia por turno P50 | 452 ms |
| Latencia por turno P95 | 4890 ms |

## Por estilo de paciente

| Estilo | Casos | Exactitud | Falsos negativos |
|---|---:|---:|---:|
| ansioso | 27 | 48% | 0 |
| colaborativo | 32 | 47% | 0 |
| confundido | 35 | 40% | 1 |
| evasivo | 29 | 21% | 0 |
| minimizador_sintomas | 37 | 78% | 1 |
