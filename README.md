[![Documentation Status](https://readthedocs.org/projects/caospy/badge/?version=latest)](https://caospy.readthedocs.io/en/latest/?badge=latest)
[![Caospy](https://github.com/Colman-Bertolo/Caospy/actions/workflows/caospy_ci.yml/badge.svg)](https://github.com/Colman-Bertolo/Caospy/actions/workflows/caospy_ci.yml)
[![MIT License](https://img.shields.io/npm/l/caos)](https://caospy.readthedocs.io/en/latest/license.html)
[![Coverage Status](https://coveralls.io/repos/github/Colman-Bertolo/Caospy/badge.svg?branch=main)](https://coveralls.io/github/Colman-Bertolo/Caospy?branch=main)
[![https://github.com/leliel12/diseno_sci_sfw](https://img.shields.io/badge/DiSoftCompCi-FAMAF-ffda00)](https://github.com/leliel12/diseno_sci_sfw)

# Caospy
Librería destinada al análisis de sistemas dinámicos. 

## Descripción

Presenta las siguientes funcionalidades:
- Resolver numéricamente EDO o sistema de EDOs temporal.
- Análisis y clasificación de puntos fijos.
- Exponente de Lyapunov.
- Generación de gráficos 2D y 3D. 
- Trazado de mapas de Poincare
- Estabilidad del sistema

Para su utilización, el usuario debe definir la función requerida, en caso contrario la propia librería cuenta con las ecuaciones de Lorenz y de Dufing integradas, debido a que dentro del estudio de sistema dinámicos dichas expresiones son muy recurrentes.

La documentación se encuentra en:

## Requerimientos 

Se necesita Python 3.9+ para correr Caospy, en el archivo requierements.txt

## Recomendaciones 

Para agilizar su utilización se recomienda importar de la siguiente manera.
```Python
import caospy as cp
```
