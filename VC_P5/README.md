# Thermal Vision Filter

Filtro de visión térmica en tiempo real que detecta rostros y aplica efectos de mapa de calor con animación de escaneo.

## Descripción

Aplicación de visión por computador que simula una cámara térmica. Detecta tu rostro usando MTCNN y aplica diferentes colormaps solo a la región facial, dejando el fondo en colores normales.

## Controles

- `D` - Cambiar colormap (inicia escaneo)
- `Q` - Salir

## Flujo del Programa

El programa comienza cargando el detector facial MTCNN y preparando cinco colormaps térmicos diferentes (JET, HOT, OCEAN, RAINBOW, DEEPGREEN). Una vez inicializado, espera a que aparezcas frente a la cámara.

Cuando detecta tu rostro por primera vez, inicia automáticamente un escaneo visual. Una línea blanca desciende de arriba hacia abajo sobre tu cara, revelando progresivamente el efecto térmico aplicado solo a la región facial mientras el fondo permanece en colores normales.

En cada frame del bucle principal, el programa captura la imagen de la cámara y detecta tu cara usando MTCNN. Crea una máscara elíptica alrededor del rostro detectado y la suaviza con un filtro gaussiano para conseguir bordes difusos. Luego convierte el frame a escala de grises y aplica el colormap térmico actual.

Si se está ejecutando un escaneo, el programa realiza una mezcla progresiva mostrando la línea de escaneo y una barra de progreso. Cuando no hay escaneo activo, muestra el efecto térmico completo aplicado al rostro. Cada vez que presionas la tecla 'D', cambia al siguiente colormap e inicia un nuevo escaneo que revela el efecto gradualmente.

En resumen, el programa detecta tu cara, aplica el efecto térmico solo al rostro manteniendo el fondo normal, y cada vez que cambias de colormap o apareces por primera vez, ejecuta un escaneo visual de arriba hacia abajo revelando el nuevo efecto.

## Ejemplo

![Ejemplo del filtro térmico](termica_ejemplo.png)

---
