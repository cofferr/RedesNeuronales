# Iris neural network: C++ (Eigen) vs Python (scikit-learn)

Final project for Redes Neuronales (Universidad Sergio Arboleda). The same
small neural network is implemented twice and compared:

- **C++17 + Eigen**: written from scratch (forward and backward propagation).
- **Python + scikit-learn**: `MLPClassifier`.

Dataset: Iris (150 samples, 4 features, 3 classes).
Architecture: 4 inputs -> 8 hidden neurons (ReLU) -> 3 outputs (softmax + cross-entropy), trained with gradient descent.
Both versions use identical hyperparameters and read the same train/test CSV files exported by Python to `data/`.

Comparison criteria: accuracy, loss, execution time, ease of implementation, pros and cons.

## Structure

```
.
├── pyproject.toml      Python project and dependencies (uv)
├── run_all.py          Runs the whole pipeline
├── python/             Data export, sklearn model, C++ runner, plots
├── cpp/                CMakeLists.txt + src/ (standalone executable, opens in Qt Creator)
├── data/               Train/test CSV files (versioned)
├── results/            Benchmark outputs and plots (not versioned)
└── README.md
```

## Setup

Requirements: [uv](https://docs.astral.sh/uv/), CMake >= 3.16 and a C++17 compiler.
Eigen 3.4 is downloaded automatically by CMake if it is not installed.

```sh
uv sync
```

## Build and run

```sh
# C++ only
cmake -B cpp/build -S cpp && cmake --build cpp/build
cpp/build/nn_eigen

# Full pipeline (export data, sklearn, build + run C++, plots)
uv run run_all.py
```

To use Qt Creator, open `cpp/CMakeLists.txt`.

## Pipeline (`uv run run_all.py`)

1. Export the stratified, standardized Iris split, shared initial weights and reference values to `data/`.
2. Train scikit-learn (`partial_fit` per epoch for the equivalent trajectory; one `fit()` call, `warm_start` from the same weights, for timing).
3. Build the C++ project in Release and run it (`nn_eigen --train`).
4. Compare C++ vs scikit-learn: fails if losses differ by more than 1e-9 or accuracy/predictions differ.
5. Benchmarks (`--no-bench` skips them): interleaved timing blocks and clean-process memory runs, written to `results/benchmark.json`.
6. Figures (PNG + PDF) in `results/figures/` and the summary table in `results/summary_table.{csv,md}`.

7. `report/` (LaTeX, Spanish): `generated/` holds the macros, tables and code listings produced from the results, and `report_overleaf.zip` (git-ignored) packs the folder for Overleaf.

`nn_eigen` also has `--gradcheck` (numerical gradient check) and runs forward/training checks against scikit-learn when called without arguments.
Tests: `uv run pytest`.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/) (`feat`, `fix`, `docs`, `chore`, ...).

## Report

`report/` is self-contained: `main.tex`, `references.bib`, `figures/` (PDF copies) and `generated/`.
Every number in the text comes from `generated/numbers.tex`, produced by `python/report_data.py` (called by `run_all.py`).
To compile locally: `cd report && tectonic -X compile main.tex` (or `latexmk -pdf main.tex`).
For Overleaf, upload `report_overleaf.zip` (created by `run_all.py`) as a new project and use the pdfLaTeX compiler with `main.tex` as the main file.


# Sustentación --- Red neuronal para Iris: C++ (Eigen) vs. scikit-learn

> **Guía de estudio y exposición oral**\
> Proyecto final de Redes Neuronales --- Universidad Sergio Arboleda\
> **Autores:** Camilo López y Juan José Márquez\
> **Docente:** Oscar Andrés Arias M.\

------------------------------------------------------------------------

# 1. ¿Qué pedía el proyecto?

La guía del proyecto final establece cuatro fases y cuatro componentes
de evaluación.

## 1.1 Requisitos técnicos de la guía

  -----------------------------------------------------------------------
  Fase                    Requisito               Cómo lo abordamos
  ----------------------- ----------------------- -----------------------
  Diseño del modelo       Elegir un conjunto de   Dataset Iris; red
                          datos y definir         4--8--3; ReLU; softmax
                          arquitectura,           con entropía cruzada;
                          activación, pérdida y   descenso de gradiente
                          optimizador             

  Implementación en C++   Usar Eigen; implementar Implementación
                          forward, backward,      explícita de
                          entrenamiento,          propagación hacia
                          validación y métricas   adelante y
                                                  retropropagación con
                                                  Eigen

  Implementación en       Usar scikit-learn o     `MLPClassifier`
  Python                  TensorFlow/Keras y      configurado para
                          mantener                reproducir la red de
                          hiperparámetros         C++
                          equivalentes            

  Análisis comparativo    Comparar precisión,     Comparación de
                          tiempo, facilidad,      exactitud, pérdida,
                          ventajas y desventajas  tiempo, memoria y
                                                  líneas de código
  -----------------------------------------------------------------------

## 1.2 Rúbrica de evaluación

  ------------------------------------------------------------------------
  Componente                                  Puntos Qué debemos demostrar
  --------------------- ---------------------------- ---------------------
  Código funcional                                40 Que ambas
                                                     implementaciones
                                                     funcionan y que el
                                                     código está
                                                     estructurado y
                                                     documentado

  Reporte técnico                                 30 Introducción,
                                                     metodología,
                                                     resultados,
                                                     conclusiones,
                                                     gráficas y análisis

  Presentación oral                               20 Organización,
                                                     claridad, dominio del
                                                     tema y respuestas a
                                                     preguntas

  Trabajo en equipo                               10 Participación
                                                     equitativa,
                                                     coordinación y
                                                     colaboración

  **Total**                                  **100** 
  ------------------------------------------------------------------------

**Prioridad para la sustentación:** no basta con mostrar que el programa
ejecuta. Debemos explicar por qué la comparación es justa, cómo funciona
la red y qué significan los resultados.

------------------------------------------------------------------------

# 2. El problema que resolvimos

## 2.1 ¿Qué es Iris?

Iris es un conjunto de datos de clasificación. Cada registro describe
una flor mediante cuatro características numéricas y pertenece a una de
tres especies:

-   Setosa
-   Versicolor
-   Virginica

El conjunto contiene:

-   150 muestras en total.
-   4 características de entrada por muestra.
-   3 clases balanceadas.
-   120 muestras para entrenamiento.
-   30 muestras para prueba.

La división fue estratificada, con semilla 42. Esto conserva la
representación de las clases en entrenamiento y prueba.

**¿Qué queremos que aprenda la red?** A partir de las cuatro
características de una flor, predecir a cuál de las tres especies
pertenece.

## 2.2 ¿Por qué este dataset?

Iris permite trabajar con un problema de clasificación multiclase
pequeño y controlado. Eso resulta útil para estudiar y comparar dos
implementaciones de una misma red neuronal sin que el tamaño del
conjunto de datos sea el principal obstáculo.

No significa que Iris represente todos los problemas reales de
aprendizaje automático. Es un caso de estudio para comparar
implementaciones bajo condiciones controladas.

------------------------------------------------------------------------

# 3. Diseño de la red neuronal

## 3.1 Arquitectura 4--8--3

La red tiene una capa de entrada, una capa oculta y una capa de salida.

``` text
        ENTRADA              CAPA OCULTA              SALIDA
       4 variables            8 neuronas              3 clases

      x1 ─────────────┐
      x2 ─────────────┼──────►  ReLU  ──────────────► Setosa
      x3 ─────────────┼──────►  8 neuronas ─────────► Versicolor
      x4 ─────────────┘                             ► Virginica
```

* **Longitud del sépalo** (*sepal length*)
* **Ancho del sépalo** (*sepal width*)
* **Longitud del pétalo** (*petal length*)
* **Ancho del pétalo** (*petal width*)

  -------------------------------------------------------------------------
  Elemento                Configuración           Función
  ----------------------- ----------------------- -------------------------
  Entrada                 4 neuronas              Recibe las cuatro
                                                  características de cada
                                                  flor

  Capa oculta             8 neuronas              Aprende combinaciones y
                                                  patrones de las
                                                  características

  Activación oculta       ReLU                    Introduce no linealidad

  Salida                  3 neuronas              Produce una
                                                  puntuación/probabilidad
                                                  para cada clase

  Activación de salida    Softmax                 Convierte los valores de
                                                  salida en probabilidades

  Pérdida                 Entropía cruzada media  Penaliza las predicciones
                                                  alejadas de la clase
                                                  correcta

  Optimizador             Descenso de gradiente   Actualiza los parámetros
                          por lote completo       para reducir la pérdida
  -------------------------------------------------------------------------

La arquitectura tiene **67 parámetros entrenables**:

-   Pesos de entrada a capa oculta: 4 × 8 = 32.
-   Sesgos de la capa oculta: 8.
-   Pesos de capa oculta a salida: 8 × 3 = 24.
-   Sesgos de salida: 3.
-   Total: 32 + 8 + 24 + 3 = **67**.

## 3.2 ¿Qué hace cada función?

**ReLU**

\[ `\operatorname{ReLU}`{=tex}(z)=`\max`{=tex}(0,z) \]

Si el valor es negativo, devuelve cero; si es positivo, conserva el
valor. Se utiliza en la capa oculta.

**Softmax**

\[ P(y=k`\mid `{=tex}x)=`\frac{e^{z_k}}{\sum_j e^{z_j}}`{=tex} \]

Transforma los valores de salida en probabilidades que suman uno. La
clase predicha es la que tiene la probabilidad más alta.

**Entropía cruzada**

Mide qué tan lejos están las probabilidades predichas de la clase
correcta. Una pérdida menor indica que, en promedio, la red asigna más
probabilidad a las clases verdaderas.

## 3.3 Hiperparámetros

  Hiperparámetro                          Valor
  ---------------------- ----------------------
  Neuronas ocultas                            8
  Tasa de aprendizaje                       0.1
  Épocas                                   1000
  Tamaño de lote                            120
  Optimizador                               SGD
  Momentum                                    0
  Regularización L2                           0
  Barajado de muestras              Desactivado
  Precisión numérica       `float64` / `double`

**¿Qué es una época?** Una pasada completa por las 120 muestras de
entrenamiento. Como usamos lote completo, cada actualización de
parámetros utiliza las 120 muestras.

**¿Qué es la tasa de aprendizaje?** El tamaño del paso con el que se
actualizan los parámetros. En nuestro caso es 0.1.

------------------------------------------------------------------------

# 4. Preparación de los datos y equivalencia

Esta es una de las partes más importantes del proyecto. Si las
implementaciones no reciben los mismos datos o no parten de los mismos
parámetros, la comparación puede ser engañosa.

## 4.1 Preparación

1.  Se divide Iris en 120 muestras de entrenamiento y 30 de prueba.
2.  La división es estratificada y utiliza la semilla 42.
3.  Se estandarizan las características usando la media y desviación
    estándar calculadas **solo con entrenamiento**.
4.  Python exporta las particiones a CSV con 17 cifras significativas.
5.  C++ lee exactamente esos mismos archivos.

**¿Por qué estandarizar solo con entrenamiento?** Para evitar fuga de
información (*data leakage*). El conjunto de prueba no debe participar
en el cálculo de los parámetros usados para transformar los datos de
entrenamiento.

## 4.2 Pesos iniciales compartidos

Las dos implementaciones parten de los mismos pesos:

-   Capa oculta: inicialización He.
-   Capa de salida: inicialización Glorot.
-   Sesgos: cero.

Los pesos se generan una sola vez y se exportan a CSV. Ambas versiones
los cargan para comenzar desde el mismo estado.

**¿Por qué es importante?** Si cada implementación comenzara con pesos
aleatorios diferentes, las diferencias en pérdida o predicción podrían
deberse a la inicialización y no al lenguaje o biblioteca.

## 4.3 ¿Cómo comprobamos que sí son equivalentes?

No nos limitamos a configurar parámetros con nombres parecidos.
Construimos verificaciones numéricas:

  Comprobación                                   Diferencia máxima reportada
  ------------------------------------------- ------------------------------
  Pérdida inicial                               (6.7`\times10`{=tex}\^{-16})
  Probabilidades iniciales                      (2.2`\times10`{=tex}\^{-16})
  Pesos después de 20 épocas                    (1.1`\times10`{=tex}\^{-16})
  Curva de pérdida durante el entrenamiento     (6.7`\times10`{=tex}\^{-16})

Además, al finalizar las ejecuciones se comprueba que:

-   La pérdida final coincida dentro de la tolerancia establecida.
-   Las curvas de pérdida coincidan.
-   La exactitud sea idéntica.
-   Las predicciones del conjunto de prueba sean idénticas.

La diferencia máxima entre las curvas fue (6.7`\times10`{=tex}\^{-16}),
cercana a la precisión de punto flotante `float64`.

**Idea para explicar al profesor:** no estamos afirmando que C++ y
Python sean idénticos internamente. Estamos demostrando que, para este
experimento, sus resultados numéricos siguen la misma trayectoria dentro
de tolerancias extremadamente pequeñas.

------------------------------------------------------------------------

# 5. ¿Cómo implementamos la red?

## 5.1 Versión C++ con Eigen

En C++ implementamos explícitamente las operaciones de la red. Eigen se
utiliza para las operaciones matriciales.

El entrenamiento se divide en dos procesos principales:

### A. Forward propagation --- propagación hacia adelante

La información avanza desde las entradas hasta las probabilidades de
salida.

1.  Se multiplican las entradas por los pesos de la capa oculta y se
    suman los sesgos.
2.  Se aplica ReLU.
3.  Se multiplican las activaciones por los pesos de salida y se suman
    los sesgos.
4.  Se aplica softmax.
5.  Se calcula la entropía cruzada.

En forma resumida:

\[ Z_1=XW_1+b_1 \]

\[ H=`\operatorname{ReLU}`{=tex}(Z_1) \]

\[ Z_2=HW_2+b_2 \]

\[ P=`\operatorname{softmax}`{=tex}(Z_2) \]

### B. Backpropagation --- retropropagación

La retropropagación calcula cuánto contribuye cada parámetro al error y
obtiene los gradientes para actualizar los pesos.

La combinación de softmax y entropía cruzada simplifica el gradiente de
salida:

\[ dZ_2=`\frac{P-Y}{N}`{=tex} \]

Luego el error se propaga hacia la capa oculta, considerando la derivada
de ReLU. Finalmente, se actualizan los parámetros:

\[
`\theta`{=tex}`\leftarrow`{=tex}`\theta`{=tex}-`\eta `{=tex}d`\theta`{=tex}
\]

Donde:

-   (`\theta`{=tex}) representa los pesos y sesgos.
-   (`\eta`{=tex}) es la tasa de aprendizaje (0.1).
-   (d`\theta`{=tex}) es el gradiente calculado.

**Cómo explicarlo sin memorizar las ecuaciones:** el forward produce una
predicción y calcula el error; el backward distribuye ese error hacia
atrás para saber cómo modificar cada peso; el descenso de gradiente
aplica esos cambios.

## 5.2 Decisiones de ingeniería en C++

**1. Workspace y búferes preasignados**

Se creó una estructura `Workspace` que reserva una sola vez las matrices
intermedias, aproximadamente 45 KB. En cada época se reutilizan esos
espacios, en lugar de solicitar memoria repetidamente.

**¿Por qué importa?** Reduce las asignaciones dinámicas dentro del ciclo
de entrenamiento y evita trabajo innecesario.

**2. Uso de `.noalias()` en Eigen**

Se utiliza en productos matriciales para indicar que el destino no se
solapa con los operandos y evitar temporales innecesarios.

**3. Verificación de asignaciones**

El programa `noalloc_check` ejecuta las 1000 épocas con
`EIGEN_RUNTIME_NO_MALLOC` y falla si se produce una reserva de memoria
durante el ciclo que se está verificando.

**4. Estabilidad numérica**

-   En softmax se resta el máximo de cada fila antes de aplicar la
    exponencial.
-   La pérdida se calcula con `log-sum-exp`.

Estas técnicas ayudan a evitar desbordamientos numéricos al calcular
exponenciales y logaritmos.

**5. Referencias constantes**

Las funciones de propagación reciben datos y parámetros mediante
referencias constantes cuando corresponde. Así se evitan copias
innecesarias y se protege la entrada frente a modificaciones
accidentales.

## 5.3 Verificación del gradiente

Una implementación manual de backpropagation puede tener errores. Por
eso se añadió el modo `--gradcheck`.

Se compara:

-   El gradiente analítico, calculado mediante retropropagación.
-   El gradiente numérico, aproximado con diferencias finitas centrales.

\[ `\frac{\partial L}{\partial\theta_k}`{=tex} `\approx`{=tex}
`\frac{L(\theta_k+\epsilon)-L(\theta_k-\epsilon)}`{=tex}
{2`\epsilon`{=tex}} \]

Se usó (`\epsilon=10`{=tex}\^{-6}). La prueba se realizó en dos puntos:
con los pesos iniciales y después de 20 épocas.

**Resultado:** el peor error relativo fue (2.9`\times10`{=tex}\^{-9}),
por debajo del umbral (10\^{-6}).

**Qué demuestra:** que el gradiente implementado en C++ coincide
estrechamente con una aproximación numérica independiente en los puntos
probados. No demuestra por sí solo que todo el programa esté libre de
errores, pero sí valida una parte crítica del entrenamiento.

## 5.4 Versión Python con scikit-learn

En Python se utilizó `MLPClassifier`, configurado con los mismos
hiperparámetros:

-   Una capa oculta de 8 neuronas.
-   ReLU.
-   SGD.
-   Tasa de aprendizaje constante de 0.1.
-   Lote de 120 muestras.
-   Sin momentum.
-   Sin regularización L2.
-   Sin barajado.
-   1000 épocas.

La dificultad fue que scikit-learn no permite establecer directamente
los pesos iniciales desde el constructor. La solución fue inicializar el
modelo y sobrescribir sus arreglos `coefs_` e `intercepts_` **in situ**
con los valores exportados a CSV.

El informe advierte que reemplazar las listas completas sería
incorrecto, porque el optimizador conserva referencias a los arreglos
originales.

## 5.5 ¿Por qué `fit()` con `warm_start=True`?

Se evaluaron dos rutas:

-   **`partial_fit()` una vez por época:** permite seguir la
    trayectoria, pero cada llamada vuelve a validar la entrada y añade
    sobrecarga de Python. Su mediana fue 423.7 ms.
-   **`fit()` con `warm_start=True`:** continúa desde los pesos cargados
    sin reinicializarlos. Con los criterios de parada configurados para
    evitar una detención anticipada, permite entrenar las 1000 épocas en
    una sola llamada.

Para la comparación de rendimiento se utilizó `fit()` con
`warm_start=True`, cuya mediana fue 50.75 ms.

**Punto importante:** los 423.7 ms de `partial_fit()` no se comparan
como si fueran el tiempo propio del algoritmo. Se presentan aparte para
mostrar el costo de llamar a Python repetidamente.

------------------------------------------------------------------------

# 6. Resultados: ¿qué obtuvimos?

Esta sección debe ser el centro de la sustentación. No basta con leer
los números: debemos explicar qué significan.

## 6.1 Calidad de clasificación

  Métrica                        C++ (Eigen)   scikit-learn
  ---------------------------- ------------- --------------
  Exactitud en entrenamiento         97.50 %        97.50 %
  Exactitud en prueba                96.67 %        96.67 %
  Pérdida en entrenamiento          0.049544       0.049544
  Pérdida en prueba                 0.078916       0.078916

En prueba se clasificaron correctamente **29 de las 30 muestras**.

La matriz de confusión fue:

  -----------------------------------------------------------------------
  Clase real        Predicha Setosa           Predicha Predicha Virginica
                                            Versicolor 
  -------------- ------------------ ------------------ ------------------
  Setosa                         10                  0                  0

  Versicolor                      0                  9                  1

  Virginica                       0                  0                 10
  -----------------------------------------------------------------------

El único error fue una muestra de **Versicolor** clasificada como
**Virginica**.

Las dos implementaciones produjeron exactamente las mismas predicciones
de prueba.

### Cómo interpretar las métricas por clase

  Clase          Precisión   Exhaustividad (recall)       F1
  ------------ ----------- ------------------------ --------
  Setosa            1.0000                   1.0000   1.0000
  Versicolor        1.0000                   0.9000   0.9474
  Virginica         0.9091                   1.0000   0.9524

-   **Precisión:** de las predicciones realizadas para una clase,
    cuántas fueron correctas.
-   **Exhaustividad o recall:** de las muestras que realmente
    pertenecían a una clase, cuántas detectó el modelo.
-   **F1:** equilibrio entre precisión y recall.

Por ejemplo, Versicolor tiene recall de 0.90 porque identificó
correctamente 9 de sus 10 muestras reales.

## 6.2 Curva de pérdida

La pérdida comienza en **1.3618** en la primera época y disminuye
durante el entrenamiento.

La diferencia máxima entre las curvas de C++ y scikit-learn es:

\[ 6.7`\times10`{=tex}\^{-16} \]

En 153 épocas la diferencia fue exactamente cero.

**Qué debemos decir al mostrar la gráfica:** las curvas prácticamente se
superponen. Esto es evidencia de que ambas implementaciones están
ejecutando el mismo proceso de aprendizaje bajo la configuración
establecida.

## 6.3 Tiempo de entrenamiento

```{=html}
<details>
```
```{=html}
<summary>
```
Tabla de tiempos (medianas y rangos)
```{=html}
</summary>
```
  Métrica                 C++ (Eigen)   scikit-learn
  --------------------- ------------- --------------
  Mediana                     4.35 ms       50.75 ms
  Media                       4.35 ms       50.72 ms
  Desviación estándar         0.09 ms        1.11 ms
  Mínimo                      4.16 ms       48.79 ms
  Máximo                      4.49 ms       52.39 ms

Se midieron 20 repeticiones por lado, después de descartar
calentamientos.

```{=html}
</details>
```
La aceleración se calcula como la razón entre las medianas:

\[
`\text{Aceleración}`{=tex}=`\frac{50.75}{4.35}`{=tex}`\approx11.7`{=tex}
\]

**Resultado:** C++ fue aproximadamente **11.7 veces más rápido** en este
benchmark.

### ¿Por qué ocurre?

La red es muy pequeña: tiene 67 parámetros y utiliza 120 muestras de
entrenamiento. Cada época realiza una cantidad reducida de cálculo
matricial.

En estas condiciones, el tiempo de scikit-learn está influido
considerablemente por costos fijos del entorno Python y de las llamadas
y validaciones de la biblioteca. El resultado no prueba que C++ sea
siempre 11.7 veces más rápido.

## 6.4 Memoria

  Métrica                                           C++ (Eigen)   scikit-learn
  ----------------------------------------------- ------------- --------------
  Memoria pico del proceso                              2.03 MB         160 MB
  Crecimiento del pico durante el entrenamiento         49.2 KB           0 KB

La relación aproximada es:

\[ `\frac{160}{2.03}`{=tex}`\approx78.8`{=tex} \]

Es decir, el proceso C++ utilizó aproximadamente **79 veces menos
memoria pico**.

**Advertencia importante:** se midió el proceso completo, no solamente
la memoria de la red. En Python se incluye el intérprete y las
bibliotecas NumPy, SciPy y pandas. Por eso no debemos afirmar que la red
neuronal en sí necesita 160 MB.

El crecimiento durante el entrenamiento fue muy pequeño en ambos casos.
En C++ fue de 49.2 KB y en scikit-learn de 0 KB según la resolución de
la medición RSS utilizada.

## 6.5 Cantidad de código

  Componente                                           C++          Python
  --------------------------------------- ---------------- ---------------
  Definición del modelo y entrenamiento          80 líneas       18 líneas
  Lectura de datos y pesos desde CSV             46 líneas        8 líneas
  Verificación numérica de gradientes            46 líneas             ---
  Equivalencia con C++ y `warm_start`                  ---       14 líneas
  **Total contabilizado**                   **172 líneas**   **40 líneas**

C++ requiere aproximadamente **4.3 veces más líneas**.

Esta cifra debe interpretarse con cuidado: Python necesita código
adicional para garantizar la equivalencia con C++, y C++ incluye la
lectura de CSV. El conteo no representa una comparación universal de
productividad.

------------------------------------------------------------------------

# 7. ¿Cómo se midió el rendimiento?

El benchmark debe explicarse porque respalda la credibilidad de los
resultados.

## 7.1 Tiempo

-   Se cronometra únicamente el entrenamiento.
-   C++ recarga los pesos iniciales antes de cada repetición, fuera del
    cronómetro.
-   Python crea un clasificador nuevo por repetición y mide la llamada a
    `fit()`.
-   Se descartan dos repeticiones de calentamiento por bloque.
-   Se miden cinco repeticiones por bloque.
-   Se alternan ocho bloques en orden C++/scikit-learn para reducir el
    efecto de cambios térmicos o temporales.
-   Se obtienen 20 mediciones por implementación.
-   Se comprueba que la pérdida final coincida con la referencia.

Se informa mediana, media, desviación estándar, mínimo y máximo.

## 7.2 Memoria

-   Se utilizan procesos separados y limpios.
-   Se hacen cinco ejecuciones por lado, alternadas.
-   Se mide el máximo RSS (*Resident Set Size*) del proceso completo.
-   También se estima el crecimiento del pico durante el entrenamiento.

## 7.3 Entorno experimental

El informe registra:

  Componente     Entorno
  -------------- ----------------------------------
  Equipo         Apple M4, 10 CPU lógicas, 24 GiB
  Sistema        macOS 27.0.1 arm64
  Compilador     Apple clang 21.0.0
  C++            C++17, Release, `-O3 -DNDEBUG`
  Eigen          3.4.0
  Python         3.14.8
  NumPy          2.5.4
  scikit-learn   1.9.1

**Por qué importa:** los tiempos dependen del hardware, el sistema
operativo, el compilador, las bibliotecas y la configuración. Para
reproducir el benchmark hay que considerar el entorno documentado.

