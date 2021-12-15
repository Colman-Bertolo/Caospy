.. Caospy documentation master file, created by
   sphinx-quickstart on Wed Nov  3 09:40:49 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Caospy's documentation!
==================================

**Caospy**  is a Python package to analyze continuous dynamical systems and chaos.

.. note::
    You'll need a Python 3.9+ version to run Caospy.



Its utilities are:

-  Solve systems of ODEs.
-  Eigenvalues, eigenvectors and roots of equations.
-  Classification of fixed points in 1D and 2D.
-  Poincare maps.


Some well studied systems are available in the library, like Lorenz's system, the Logistic
equation, Duffing's system and the Rosslers-Chaos systems.

Motivation
-----------

Dynamic sisytems are one of the most researched niches of knowledge, their understanding is crucial in order to interact more effectively with our environment. In order to study any dynamical systems to an acceptable level of detail, the classical qualitative analysis fall short and different heuristic methods along with numerical approaches where born to better understand their behavior. Properties like fixed points stability or chaotic behavior, and phenomena like bifurcations are commonly obtained by means of using these so called heuristic methods. Caospy attempts to bring this different tools of analysis together in one Python package, and achieve the unification and regularization of their use in a common developing context, hoping it will provide an easier and more comprehensive analysis of the subject in question.

**Authors**:
Juan Colman, Sebastián Bertolo.

Repository 
-----------

https://github.com/Colman-Bertolo/Caospy




.. toctree::
   :maxdepth: 1
   :caption: Contents:


   install
   tutorial.ipynb
   license
   Api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
