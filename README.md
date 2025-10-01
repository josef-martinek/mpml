# Mixed Precision Multilevel Monte Carlo
Implementation of the standard multilevel Monte Carlo method (MLMC) and the mixed precision multilevel Monte Carlo method (MPML) for computing expectations. MPML adaptively chooses computational accuracy on each level. We suggest starting with the provided **[examples](./examples/)**.
## Related publications
* J. Martínek, E. Carson, and R. Scheichl. **[Exploiting Inexact Computations in Multilevel Sampling Methods](https://arxiv.org/abs/2503.05533)**. arXiv preprint 3/2025.

Scripts generating data from figures in arXiv preprint 3/2025 can be found in **[examples/figures/](./examples/figures/)**. To reproduce the data from a particular figure, run the script located in the appropriate folder. Note that the script has to be run from the root folder of the repository. The file "git_commit.txt" in each folder contains the hash of the commit used to generate the data. It might be necessary to adjust the number of workers for parallel execution. For each MLMC and MPML algorithm run, the script outputs the resulting estimate, the number of samples used on each level, and the cost per sample on each level. From this data the desired algorithm statistics can be straightforwardly computed.
## Installation
We recommend using a Linux-based OS for installation and execution. The list of dependencies can be found in **[`requirements.yml`](./requirements.yml)**. To create the required environment using Conda, clone this repository and run the following command from the root folder:

```conda env create -f environment.yml ```