# Transferencia secuencial desde imágenes de laboratorio para clasificar enfermedades foliares de la papa en campo

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StevenSntg/papa-transferencia-secuencial/blob/main/Modelo_Investigacion.ipynb)

Código del trabajo final del **Grupo 3** del curso Redes Neuronales 2026-II (Escuela Profesional de Ingeniería
de Sistemas, Universidad Nacional Tecnológica de Lima Sur). Docente: Mg. Javier A. Azabache Asmat.

Integrantes: Jaqueline María Álvarez Rocca · Diego Armando Gamero Palacios · Carlos Alberto de Jesús Meza Peláez ·
Carlos Steven Santiago Flores.

El cuaderno entrena una CNN **EfficientNetV2B0** con tres estrategias de inicialización (desde cero, transferencia
directa desde ImageNet y transferencia secuencial ImageNet–PlantVillage–campo) y responde las cuatro preguntas del
artículo: la brecha laboratorio–campo (E1), la comparación de estrategias (E2), el efecto de la partición
depurada frente a la aleatoria (E3) y el análisis de errores con Grad-CAM (E4).

## Contenido

| Ruta | Qué es |
|---|---|
| `Modelo_Investigacion.ipynb` | Cuaderno principal: datos, preprocesamiento, modelo, entrenamiento, evaluación de E1–E4 y **DEMO** |
| `requirements.txt` | Librerías y versiones |
| `scripts/descargar_datasets.py` | Descarga los dos conjuntos de fuentes públicas y verifica el SHA-256 de cada imagen |
| `scripts/auditar_y_particionar.py` | Auditoría de duplicados (SHA-256 y dHash) y particiones 70/15/15 |
| `scripts/preparar_demo.py` | Copia a `demo/` las cinco fotos de la DEMO (todas de la partición de prueba) |
| `particiones/` | Particiones de referencia (`campo_particion.csv`, `laboratorio_particion.csv`) y su resumen |
| `demo/` | Fotos de la DEMO y `casos_demo.csv` con la salida esperada de cada una |
| `resultados/` | Métricas, predicciones y figuras de la corrida que respalda el artículo (se agrega tras la corrida) |

## Requisitos de ejecución

| Elemento | Valor |
|---|---|
| Entorno | Google Colab (recomendado) o Jupyter local |
| Python | 3.10 o superior (la versión exacta de la corrida queda en `resultados/entorno.json`) |
| Hardware | **GPU** (T4 de Colab). En CPU el entrenamiento completo tardaría días; solo sirve el modo de prueba |
| RAM | unos 6 GB |
| Disco | unos 2 GB en Google Drive (caché de imágenes y 15 modelos de ~72 MB) más 0,8 GB temporales en Colab |
| Tiempo | entre 1,5 y 3 h para la corrida completa con GPU T4 (estimación; el tiempo real queda en `resultados/entorno.json`) |
| Librerías | TensorFlow / Keras 3, NumPy, pandas, scikit-learn, SciPy, Matplotlib, Pillow (ver `requirements.txt`) |

## Datos

| Conjunto | Fuente | Licencia | Tamaño |
|---|---|---|---|
| Campo | Shabrina *et al.*, *Potato Leaf Disease Dataset in Uncontrolled Environment*, Mendeley Data, DOI [10.17632/ptz377bwb8.1](https://doi.org/10.17632/ptz377bwb8.1); artículo en *Data in Brief* 52:109955 (2024) | CC BY 4.0 | 3076 JPG de 1500 × 1500 px, 7 clases, ~730 MB |
| Laboratorio | Subconjunto de papa de PlantVillage, repositorio [spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset) (`raw/color`) | CC BY-SA 3.0 | 2152 JPG de 256 × 256 px, 3 clases, ~44 MB |

El cuaderno los descarga solo con `scripts/descargar_datasets.py` (API pública de Mendeley Data y *sparse checkout*
de GitHub), en `/content/datos/` en Colab o en `../datos/` en local. Después guarda las imágenes redimensionadas a
224 × 224 en la caché de Drive y no vuelve a descargarlas. Las fotos de `demo/` son cinco imágenes de esos dos
conjuntos y se redistribuyen bajo sus licencias, con la atribución anterior.

## Credenciales

**No se necesita ninguna.** Los conjuntos son públicos. El único acceso es el montaje de Google Drive, que Colab
pide con su propia ventana de autorización; ninguna clave ni token aparece en el código.

## Cómo ejecutarlo

1. Abrir el cuaderno con el botón **Abrir en Colab** de arriba.
2. *Entorno de ejecución → Cambiar tipo de entorno de ejecución →* **GPU T4**.
3. *Entorno de ejecución →* **Ejecutar todas**. La primera celda clona este repositorio y la segunda instala las
   dependencias que falten.
4. Autorizar el acceso a Google Drive. Todo se guarda en `MyDrive/TF_Grupo3_RN/` (`cache/`, `modelos/`, `resultados/`).
5. Mantener la pestaña abierta mientras entrena. Si Colab se desconecta, volver a *Ejecutar todas*: las corridas
   terminadas se saltan y solo se repite la que estaba a medias.
6. Al terminar aparece «Corrida completa.». Los resultados quedan en `MyDrive/TF_Grupo3_RN/resultados/` y en
   `resultados_TF_Grupo3.zip`.

**Modo de prueba.** Con la variable de entorno `MODO_CUADERNO=prueba`, el cuaderno usa unas pocas fotos, una época
por fase y una semilla; sirve para comprobar que todo funciona en CPU, no para obtener resultados. En local:

```
cd 02_CODIGO
set MODO_CUADERNO=prueba        (Windows; en Linux/macOS: export MODO_CUADERNO=prueba)
jupyter nbconvert --to notebook --execute Modelo_Investigacion.ipynb --output prueba.ipynb
```

## DEMO

La sección 13 del cuaderno carga el modelo guardado, **sin reentrenar**, y clasifica cinco fotos de `demo/`: tres
de clases frecuentes, un caso límite (nematodos, la clase más rara) y un caso atípico (una foto de PlantVillage).
También muestra dos errores controlados (archivo inexistente y archivo que no es una imagen). Si el modelo no está
en Drive, lo descarga de la *release* `modelo-v1` de este repositorio.

## Configuración del modelo

EfficientNetV2B0 (270 capas, 5 919 312 parámetros en el extractor, activación SiLU) + promedio global + *dropout*
0,2 + capa densa *softmax* (7 o 3 neuronas). Entrada 224 × 224 × 3 con píxeles 0–255 (la red incluye su
normalización). Entropía cruzada categórica con pesos por clase inversos a la frecuencia, Adam, lote 32. Con
transferencia: 5 épocas solo la cabeza (tasa 0,001) y hasta 50 épocas todas las capas (tasa 0,0001), con la
normalización por lotes en modo inferencia. Desde cero: hasta 100 épocas (tasa 0,001). Parada temprana con
paciencia 10 sobre la pérdida de validación. Tres semillas (1, 2 y 3) por configuración. Precisión mixta en GPU.
