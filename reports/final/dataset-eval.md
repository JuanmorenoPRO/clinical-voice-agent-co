# Evaluación sobre los 160 casos del dataset

160 conversaciones reproducidas a través del agente completo (comprensión del lenguaje, acumulación de síntomas y decisión).


## capa2_ruidosa  (160 casos)

```
  real \ predicho   verde  amarillo   rojo
  verde                55       33       35
  amarillo              1       11       13
  rojo                  0        0       12

  exactitud 78/160 (48.8%)
```

- **Falsos negativos: 1** (de los cuales 0 en rojo, que es la falla catastrófica)
  - `caso_tray_pac_42_00015_3` esperaba **amarillo**, dijo *verde* · completitud 1.0

## Exactitud de la extracción, por slot

Compara lo que el agente entendió con el cuadro real de la trayectoria. Si un caso falla, esto dice si la culpa fue de entender al paciente o de los umbrales.

| Slot | Acierto | Sin extraer |
|---|---:|---:|
| dolor_nrs | 46% | 5% |
| movilidad | 71% | 16% |
| herida | 59% | 23% |
| apetito | 69% | 14% |
| sueno | 63% | 18% |

## Consumo por llamada

| Métrica | Valor |
|---|---:|
| Turnos por llamada | 6.9 |
| Invocaciones al modelo por llamada | 3.2 |
| Invocaciones al modelo por turno | 0.46 |
| Tokens de entrada por llamada | 1852 |
| Tokens de salida por llamada | 77 |
| Latencia por turno P50 | 4 ms |
| Latencia por turno P95 | 5001 ms |

## Por estilo de paciente

| Estilo | Casos | Exactitud | Falsos negativos |
|---|---:|---:|---:|
| ansioso | 27 | 44% | 0 |
| colaborativo | 32 | 44% | 0 |
| confundido | 35 | 43% | 0 |
| evasivo | 29 | 21% | 0 |
| minimizador_sintomas | 37 | 84% | 1 |
