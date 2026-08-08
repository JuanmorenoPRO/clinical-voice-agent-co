# Evaluación sobre los 160 casos del dataset

160 conversaciones reproducidas a través del agente completo (comprensión del lenguaje, acumulación de síntomas y decisión).


## capa2_ruidosa  (160 casos)

```
  real \ predicho   verde  amarillo   rojo
  verde                64       31       28
  amarillo              1       12       12
  rojo                  0        1       11

  exactitud 87/160 (54.4%)
```

- **Falsos negativos: 2** (de los cuales 1 en rojo, que es la falla catastrófica)
  - `caso_tray_pac_42_00015_3` esperaba **amarillo**, dijo *verde* · completitud 1.0
  - `caso_tray_pac_42_00030_7` esperaba **rojo**, dijo *amarillo* · completitud 0.833

## Exactitud de la extracción, por slot

Compara lo que el agente entendió con el cuadro real de la trayectoria. Si un caso falla, esto dice si la culpa fue de entender al paciente o de los umbrales.

| Slot | Acierto | Sin extraer |
|---|---:|---:|
| dolor_nrs | 59% | 11% |
| movilidad | 72% | 9% |
| herida | 66% | 17% |
| apetito | 69% | 13% |
| sueno | 61% | 18% |

## Consumo por llamada

| Métrica | Valor |
|---|---:|
| Turnos por llamada | 6.9 |
| Invocaciones al modelo por llamada | 3.1 |
| Invocaciones al modelo por turno | 0.45 |
| Tokens de entrada por llamada | 1435 |
| Tokens de salida por llamada | 65 |
| Latencia por turno P50 | 3 ms |
| Latencia por turno P95 | 5856 ms |

## Por estilo de paciente

| Estilo | Casos | Exactitud | Falsos negativos |
|---|---:|---:|---:|
| ansioso | 27 | 44% | 0 |
| colaborativo | 32 | 69% | 0 |
| confundido | 35 | 51% | 0 |
| evasivo | 29 | 17% | 1 |
| minimizador_sintomas | 37 | 81% | 1 |
