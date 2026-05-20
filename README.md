# 🔬 Caminata Autoevitante (Self-Avoiding Walk)

## 🎯 ¿Qué es?
Modelo de simulación de **polímeros** o **cadenas moleculares** que no pueden cruzarse a sí mismas. Es un problema fundamental en física estadística y ciencia de materiales.

## 📊 Qué mide
- **<dₙ>**: Distancia promedio desde el origen después de n pasos
- **<dₙ²>**: Distancia cuadrática promedio
- **Exponente de Flory ν**: Caracteriza cómo crece el tamaño de la cadena con el número de monómeros

## 🔬 Teoría
Para una caminata autoevitante en 2D, se espera:
<dₙ²> ∝ n^(2ν) con ν = 0.75
Este exponente es **mayor** que en caminata simple (ν = 0.5), reflejando la "hinchazón" de la cadena por el impedimento estérico.

## 🛠️ Tecnologías
- Python (NumPy, Matplotlib, Seaborn)
- Regresión lineal en escala log-log
- Método de Monte Carlo

## 📈 Resultados obtenidos
| Parámetro | Valor simulado | Teórico |
|-----------|---------------|---------|
| ν (de <dₙ²>) | 0.74 | 0.75 |
| R² | 0.998 | - |

## ▶️ Cómo ejecutar
```bash
python self_avoiding_walk.py
