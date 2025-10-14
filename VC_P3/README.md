# Detección y reconocimiento de formas

## Descripción del trabajo
Este repositorio con tiene dos ejercicios de visión por computador desarrollados en **Python** con la librería **OpenCV**. Cada uno trata un problema distinto de análisis de imágenes. Los ejercicios realizados son los siguientes:

#### 1. **Contador de dinero** en imágenes
En este ejercicio se ha desarrollado un sistema capaz de calcular la cantidad total de dinero presente en una imagen con monedas.

Instrucciones de uso:
1. Ejecutar
2. Introducir los parámetros que se solicitan:
    - **Escala**: 0.1 para imágenes grandes
    - **Método de umbralizado**: 1 para Otsu (mejor con iluminación uniforme) o 2 para Adaptativo (mejor con mala iluminación)
3. Cuando aparezca la ventana de las monedas, **hacer click en la de 1 euro**
4. Se mostrará en pantalla el valor total

Funcionamiento:

1. **Preprocesamiento**: Convierte la imagen a escala de grises, mejora el contraste y aplica filtros para reducir el ruido
2. **Detección de contornos**: Utiliza umbralización de Otsu o un umbral adaptativo para detectar los bordes circulares de las monedas
3. **Calibración**: El usuario hace clic en una moneda de 1€ como referencia
4. **Identificación**: Compara el tamaño de cada moneda detectada con los tamaños relativos conocidos de las monedas de euro
5. **Cálculo**: Suma el valor total y muestra el resultado en pantalla
<p align="center">
  <img src="Monedas.jpg" alt="Imagen de las monedas" height="350"/>
  <img src="Detector_monedas.jpg" alt="Ventana del detector de monedas" height="350"/>
</p>

#### 2. **Clasificador heurístico de partículas**
Este proyecto se enfoca en la construcción de un clasificador para distintos tipos de microplásticos (FRA, PEL, TAR) según sus características intrínsecas.

Funcionamiento:

1. **Entrenamiento**:
    - Se analiza una imagen de referencia para cada tipo de partícula (FRA, PEL, TAR).
    - Para cada partícula, se extraen las características más relevantes (área, perímetro, brillo, saturación, ...)
    - Se calcula el **vector medio** de todas las partículas de la misma clase para crear un "prototipo" representativo de cada tipo de microplástico.
2. **Normalización y ponderación**:
    - Se **normalizan** los vectores de características para que todas las métricas puedan ser comparadas, evitando valores muy grandes o muy pequeños.
    - Se asignan **pesos** a cada característica según su capacidad para distinguir entre clases, dando más importancia a las más distintivas.
3. **Clasificación y comparación**:
    - Se procesa una **imagen de test** con todas las clases mezcladas para predecir las etiquetas gracias a lo aprendido en el entrenamiento.
    - Para cada partícula detectada se calcula su **vector de características** y se **compara** con el vector prototipo.
    - La partícula se asignará a la clase cuyo prototipo sea más similar.
4. **Resultados**:
    - Finalmente, se comparan las predicciones con las etiquetas reales para evaluar el rendimiento con medidas como la **precisión** o la **matriz de confusión**.

<p>
    <img src="Comparacion_clasificaciones.jpg" alt="Comparación de la clasigicación real con la obtenida"/>
</p>

---

## Autoría

Este trabajo ha sido realizado por **Juan Francisco del Rosario Machin** y **Mario García Abellán**, como parte de una práctica de procesamiento de imágenes en el entorno académico.

---

## Fuentes y referencias

Durante el desarrollo de la práctica se consultaron o utilizaron las siguientes fuentes:

- Documentación oficial de OpenCV: [https://docs.opencv.org](https://docs.opencv.org)
- [Stack Overflow](https://stackoverflow.com/)
  - [setMouseCallback arguments](https://stackoverflow.com/questions/47114360/what-should-be-the-arguments-of-cv2-setmousecallback#50193432)
- Consultas a [ChatGPT](https://chatgpt.com/)