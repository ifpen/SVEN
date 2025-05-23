# SVEN
A simple unsteady Free Vortex Wake lifting-line filament flow solver 

**SVEN**: (S)olver (V)ortex (E)olie(N)nes

# SCOPE

The objective of this project is to provide a simple and modular implementation of a Free Vortex Wake (FVW) method with a Lifting Line approach and filament discretization, developed in Python. The focus is on simplicity and accessibility for academic use while leveraging the power of GPUs via CUDA for fast computations.

This code is designed to enable the exploration and testing of different models and aspects of the method, without aiming to offer a complete or highly optimized implementation. The emphasis is on flexibility and ease of adaptation to meet the specific needs of users, whether they are researchers, engineers, or students.

# How to use it

To be completed.

# Implemented models



This code implements a vortex-based aerodynamic model using a lifting-line formulation with a filament-discretized free wake. It computes bound circulation along the blade and tracks shed and trailing filaments to capture unsteady aerodynamic effects.

The implementation follows the principles of the Free Vortex Wake method, using the Biot–Savart law and the Kutta–Joukowski theorem to resolve induced velocities and aerodynamic forces.

More information on the implemented models can be found in [Le Guern's Thesis](https://theses.fr/2024BORD0313).





# Dependencies 

**1. Install CUDA Toolkit** 

This library requires CUDA for GPU acceleration. To set up CUDA: 

- Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) from NVIDIA.
- Ensure the installed version matches the one specified in `environment.yaml` (e.g., `11.8.0`).


**2. Prepare the environment**

Run the following command to create the Conda environment:

```bash
conda env create -f environment.yaml
```

**3. Set up environment variables for CUDA**

After installing CUDA, add it to your system's path. Append the following lines to your shell configuration file (e.g.,~/.bashrc):

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
Then reload your shell: 

```bash
source ~/.bashrc
```

**4. Activate the environment**

```bash
conda activate sven
```

# Authors 

Caroline Le Guern, Frédéric Blondel [@ IFP Energies nouvelles](https://www.ifpenergiesnouvelles.com)


