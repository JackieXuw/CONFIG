# Introduction
 CONFIG (CONstrained efFIcient Global optimization) is a data-driven efficient global optimization toolbox. 
 By sequentially and adaptively evaluting different candidate solutions under the guidance of Gaussian process surrogates, 
 CONFIG algorithm can efficiently identify globally optimal solution for
 constrained optimization problems with potentially non-convex and multi-modal
 functions.

# Use Cases
CONFIG toolbox can be applied for general black-box optimization problems with
a compact set of candidate solutions. It is particularly useful when the objective
and constraint functions are expensive to evaluate. 
Typical application domains include set-points tuning in process control and PI
controller tuning for energy systems, when constraint violations during the
tuning process are not safety critical.

# Supported Algorithms
We recommend the CONFIG algorithm, which has demonstrated 
good convergence property to global optimal solution both in theory and in
practice. But to allow more flexibility for user's choice, we also implement the
following popular algorithms.
* Constrained EI.
* Primal-dual.
* EPBO.

# Install
Under the directory where [README.md](./README.md) is, run `pip install .`. 

# Usage
In the directory "./examples", we provide a notebook to demonstrate the usage of the toolbox. 

# Citation
If you use this toolbox for your research that leads to publications, we would
appreciate your recognition by citing the following paper. 
* [Xu, W., Jiang, Y., and Jones, C.N. (2022a). Con-
strained efficient global optimization of expensive black-
box functions. doi:10.48550/ARXIV.2211.00162. URL
https://arxiv.org/abs/2211.00162.](https://arxiv.org/abs/2211.00162)

