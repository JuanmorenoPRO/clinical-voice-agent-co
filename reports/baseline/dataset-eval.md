# Evaluación sobre los 160 casos del dataset

160 conversaciones reproducidas a través del agente completo (comprensión del lenguaje, acumulación de síntomas y decisión).


## capa2_ruidosa  (160 casos)

```
  real \ predicho   verde  amarillo   rojo
  verde                47       33       43
  amarillo              1        9       15
  rojo                  0        0       12

  exactitud 68/160 (42.5%)
```

- **Falsos negativos: 1** (de los cuales 0 en rojo, que es la falla catastrófica)
  - `caso_tray_pac_42_00015_3` esperaba **amarillo**, dijo *verde* · completitud 1.0

## Exactitud de la extracción, por slot

Compara lo que el agente entendió con el cuadro real de la trayectoria. Si un caso falla, esto dice si la culpa fue de entender al paciente o de los umbrales.

| Slot | Acierto | Sin extraer |
|---|---:|---:|
| dolor_nrs | 46% | 5% |
| movilidad | 72% | 12% |
| herida | 64% | 13% |
| apetito | 69% | 12% |
| sueno | 61% | 17% |

## Consumo por llamada

| Métrica | Valor |
|---|---:|
| Turnos por llamada | 6.9 |
| Invocaciones al modelo por llamada | 2.9 |
| Invocaciones al modelo por turno | 0.42 |
| Tokens de entrada por llamada | 1712 |
| Tokens de salida por llamada | 72 |
| Latencia por turno P50 | 2 ms |
| Latencia por turno P95 | 5033 ms |

## Por estilo de paciente

| Estilo | Casos | Exactitud | Falsos negativos |
|---|---:|---:|---:|
| ansioso | 27 | 37% | 0 |
| colaborativo | 32 | 47% | 0 |
| confundido | 35 | 34% | 0 |
| evasivo | 29 | 14% | 0 |
| minimizador_sintomas | 37 | 73% | 1 |
