# Red neuronal para Iris: C++ (Eigen) vs Python (scikit-learn)

La misma red neuronal pequeña se implementa dos veces y se compara:

- **C++17 + [Eigen](https://eigen.tuxfamily.org)**: propagación hacia adelante y hacia atrás escritas desde cero.
- **Python + [scikit-learn](https://scikit-learn.org)**: `MLPClassifier`.

Proyecto final de *Redes Neuronales* (Universidad Sergio Arboleda).

| | |
|---|---|
| Datos | Iris: 150 muestras, 4 características, 3 clases (partición estratificada 80/20, estandarizada) |
| Arquitectura | 4 entradas → 8 neuronas ocultas (ReLU) → 3 salidas (softmax + entropía cruzada) |
| Entrenamiento | Descenso de gradiente con todo el conjunto por época; hiperparámetros en [python/config.py](python/config.py) |

Ambas versiones parten de **los mismos datos, los mismos pesos iniciales y los mismos hiperparámetros**, por lo que siguen la misma trayectoria de entrenamiento. El pipeline lo verifica: falla si las curvas de pérdida difieren en más de `1e-9` o si las predicciones de prueba son distintas.

La comparación cubre exactitud, pérdida, tiempo de entrenamiento, memoria, facilidad de implementación y ventajas y desventajas. El informe escrito está en [report/](report/) (LaTeX, en español).

## Requisitos

- [uv](https://docs.astral.sh/uv/) (instala Python 3.14 y las dependencias)
- CMake ≥ 3.16 y un compilador de C++17
- Conexión a internet en la primera compilación: CMake descarga Eigen 3.4 si no está instalado
- Opcional, para compilar el informe: [tectonic](https://tectonic-typesetting.github.io/) o una distribución de LaTeX

## Inicio rápido

```sh
uv sync                        # instala las dependencias de Python
uv run run_all.py              # ejecuta todo el pipeline
uv run run_all.py --no-bench   # igual, sin los benchmarks (más lentos)
uv run pytest                  # ejecuta las pruebas
```

## Qué hace `uv run run_all.py`

| # | Etapa | Salida |
|---|---|---|
| 1 | Exporta la partición, los pesos iniciales compartidos y los valores de referencia | `data/` |
| 2 | Entrena scikit-learn (`partial_fit` por época para la trayectoria, un `fit()` para el tiempo) | `results/sklearn_*` |
| 3 | Compila el proyecto de C++ en Release (`cmake`) | `cpp/build/nn_eigen` |
| 4 | Ejecuta el binario de C++ (`nn_eigen --train`) | `results/cpp_*` |
| 5 | Compara C++ y scikit-learn (curvas de pérdida, exactitud, predicciones); se detiene ante cualquier diferencia | consola |
| 6 | Benchmarks: bloques de tiempo intercalados y ejecuciones de memoria en procesos limpios | `results/benchmark.json` |
| 7 | Figuras y tabla resumen | `results/figures/`, `results/summary_table.{csv,md}` |
| 8 | Macros, tablas y listados del informe, más el zip para Overleaf | `report/generated/`, `report_overleaf.zip` |

Python dirige el proceso: [python/cpp_runner.py](python/cpp_runner.py) compila y lanza el programa de C++, que lee los CSV que Python escribió en `data/`.
`--no-bench` omite las etapas 6 y 8.

## Ejecutar solo la parte de C++

```sh
cmake -B cpp/build -S cpp && cmake --build cpp/build
cpp/build/nn_eigen --train        # entrena y escribe results/cpp_*
cpp/build/nn_eigen --gradcheck    # verificación numérica del gradiente
cpp/build/nn_eigen                # comprobaciones de forward y entrenamiento contra scikit-learn
cpp/build/noalloc_check           # comprueba que el bucle de entrenamiento no reserva memoria
```

`data/` ya contiene los CSV, así que el programa de C++ corre sin Python. Para usar Qt Creator, abre `cpp/CMakeLists.txt`.

## Estructura del repositorio

```
.
├── run_all.py          Punto de entrada del pipeline
├── pyproject.toml      Proyecto Python y dependencias (uv); uv.lock fija las versiones
├── python/             Lado Python: datos, modelo sklearn, ejecutor de C++, benchmarks, gráficas
├── cpp/                Lado C++: CMakeLists.txt y src/
├── data/               CSV de entrenamiento y prueba, pesos iniciales, valores de referencia (versionados)
├── results/            Salidas del pipeline y figuras (se generan, no se versionan)
├── tests/              Pruebas con pytest
└── report/             Informe en LaTeX: main.tex, references.bib, figures/, generated/
```

## Archivos importantes

Para entender el proyecto, empieza por aquí.

**C++** (`cpp/src/`)

| Archivo | Función |
|---|---|
| [network.hpp](cpp/src/network.hpp) | La red: forward, retropropagación, paso de descenso de gradiente y buffers preasignados |
| [main.cpp](cpp/src/main.cpp) | Punto de entrada: bucle de entrenamiento, modos de línea de comandos, medición de tiempo y memoria |
| [csv.hpp](cpp/src/csv.hpp) | Lectura de los CSV |
| [gradcheck.hpp](cpp/src/gradcheck.hpp) | Compara los gradientes analíticos con los numéricos |
| [noalloc_check.cpp](cpp/src/noalloc_check.cpp) | Comprueba que el entrenamiento no reserva memoria en Eigen |

**Python** (`python/`)

| Archivo | Función |
|---|---|
| [config.py](python/config.py) | Hiperparámetros y rutas compartidos por ambas versiones |
| [data.py](python/data.py) | Partición, estandarización y exportación a CSV |
| [weights.py](python/weights.py) | Pesos iniciales compartidos |
| [sklearn_model.py](python/sklearn_model.py) | El modelo de scikit-learn y su trayectoria de referencia |
| [compare.py](python/compare.py) | Verificación de equivalencia entre C++ y scikit-learn |
| [cpp_runner.py](python/cpp_runner.py) | Compila y ejecuta el binario de C++ desde Python |
| [bench.py](python/bench.py), [bench_worker.py](python/bench_worker.py) | Benchmarks de tiempo y memoria |
| [plots.py](python/plots.py) | Figuras y tabla resumen |
| [report_data.py](python/report_data.py) | Genera las macros, tablas y listados del informe, y el zip de Overleaf |

## Pruebas

`uv run pytest` cubre la equivalencia entre C++ y scikit-learn, la comprobación de que no hay reservas de memoria, la exportación de datos y configuración, los benchmarks, las gráficas y la generación del informe.
La prueba del informe lo compila y comprueba que no pase de diez páginas; se omite si `tectonic` no está instalado.

## Informe

`report/` es autocontenido. Cada número del texto sale de `report/generated/numbers.tex`, que produce [python/report_data.py](python/report_data.py) a partir de los resultados del pipeline.

```sh
cd report && tectonic -X compile main.tex     # o: latexmk -pdf main.tex
uv run python -m python.report_data --zip     # report_overleaf.zip, para subir a Overleaf
```

En Overleaf, usa el compilador pdfLaTeX con `main.tex` como archivo principal.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/) (`feat`, `fix`, `docs`, `test`, `chore`, ...).
