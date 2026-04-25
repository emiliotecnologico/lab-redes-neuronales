# Laboratorio de Redes Neuronales – Binarización e Interpretabilidad Biológica

## Descripción
Aplicación con interfaz gráfica (tkinter) que compara el rendimiento de redes neuronales **continuas (32-bit)** vs **binarizadas (1-bit)** para predecir demanda sintética.  
Analiza métricas clave: error cuadrático medio (MSE), tiempo de inferencia, ahorro de memoria e **interpretabilidad biológica** (activación de neuronas ocultas).

## Tecnologías utilizadas
- Python 3.x
- scikit-learn
- matplotlib
- numpy
- tkinter (interfaz gráfica nativa)

## Características principales
- ✅ Generación de datos sintéticos con ruido.
- ✅ Entrenamiento y comparación de dos modelos (MLPRegressor).
- ✅ Cálculo de reducción de memoria (hasta 32x).
- ✅ Visualización de activaciones de neuronas ocultas (selectividad).
- ✅ Gráficos interactivos de predicciones y rendimiento.
- ✅ Exportación de resultados a CSV.

## Cómo ejecutar
1. Asegúrate de tener Python 3 instalado.
2. Instala las dependencias necesarias (recomendado usar un entorno virtual):
   ```bash
   pip install scikit-learn matplotlib numpy
