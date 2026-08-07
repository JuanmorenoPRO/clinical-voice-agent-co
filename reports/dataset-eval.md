# Evaluación sobre los 160 casos del dataset

160 conversaciones reproducidas a través del agente completo (comprensión del lenguaje, acumulación de síntomas y decisión).


## capa1_limpia  (160 casos)

```
  real \ predicho   verde  amarillo   rojo
  verde                79       34       10
  amarillo              1       13       11
  rojo                  0        1       11

  exactitud 103/160 (64.4%)
```

- **Falsos negativos: 2** (de los cuales 1 en rojo, que es la falla catastrófica)
  - `caso_tray_pac_42_00015_3` esperaba **amarillo**, dijo *verde* · completitud 0.833
  - `caso_tray_pac_42_00030_7` esperaba **rojo**, dijo *amarillo* · completitud 0.667

## Exactitud de la extracción, por slot

Compara lo que el agente entendió con el cuadro real de la trayectoria. Si un caso falla, esto dice si la culpa fue de entender al paciente o de los umbrales.

| Slot | Acierto | Sin extraer |
|---|---:|---:|
| dolor_nrs | 66% | 6% |
| movilidad | 75% | 11% |
| herida | 70% | 17% |
| apetito | 73% | 8% |
| sueno | 67% | 11% |

## Consumo por llamada

| Métrica | Valor |
|---|---:|
| Turnos por llamada | 6.0 |
| Invocaciones al modelo por llamada | 1.6 |
| Invocaciones al modelo por turno | 0.26 |
| Tokens de entrada por llamada | 788 |
| Tokens de salida por llamada | 42 |
| Latencia por turno P50 | 2 ms |
| Latencia por turno P95 | 10345 ms |

## Por estilo de paciente

| Estilo | Casos | Exactitud | Falsos negativos |
|---|---:|---:|---:|
| ansioso | 27 | 59% | 0 |
| colaborativo | 32 | 81% | 0 |
| confundido | 35 | 63% | 0 |
| evasivo | 29 | 17% | 1 |
| minimizador_sintomas | 37 | 92% | 1 |
