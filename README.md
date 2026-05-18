# Publication:
This repository introduces and provides BRAID (**Behavioraly relevant Analysis of Intrinsic Dynamics**)

Parsa Vahidi, Omid G. Sani, and Maryam Shanechi. *BRAID: Input-driven nonlinear dynamical modeling of neural-behavioral data.* ***In The Thirteenth International Conference on Learning
Representations***, 2025. URL https://openreview.net/forum?id=3usdM1AuI3.

# Usage examples
The following notebook contains usage examples of BRAID for several use-cases:
[source/BRAID/example/BRAID_tutorial.ipynb](source/BRAID/example/BRAID_tutorial.ipynb)

A .py scripty version of the same notebook is also available in the same directory [source/BRAID/example/BRAID_tutorial.py](source/BRAID/example/BRAID_tutorial.py)

# Usage examples
The following are the key classes that are used to implement BRAID formulation as explained in [source/BRAID/BRAIDModelDoc.md](source/BRAID/BRAIDModelDoc.md) (the code for these key classes is also available in the same directory):

- `BRAIDModel` [./source/BRAID/BRAIDModel.py](./source/BRAID/BRAIDModel.py): The full BRAID model class for fitting and inference including the optional preprocessing stage and post-learning stage 3. BRAID's main 2 stages are implemented in a separate class named MainModel in [source/BRAID/MainModel.py].

- `MainModel` [./source/BRAID/MainModel.py](./source/BRAID/MainModel.py): BRAID's main 2 stages implemented in a separate class. The BRAIDModel object build a MainModel object internally to perform stage 1 and 2.

- `RNNModel` [./source/BRAID/RNNModel.py](./source/BRAID/RNNModel.py): The custom RNN class, which implements the RNNs that are trained in stages 1, 2 (and the preprocessing/stage 3 stpes if used). 

- `RegressionModel` [./source/BRAID/RegressionModel.py](./source/BRAID/RegressionModel.py): The class internally used by both RNNModel and MainModel to build the general multilayer feed-forward neural networks that are used to implement each model parameter.


# License
Copyright (c) 2025 University of Southern California\
See full notice in [LICENSE.md](LICENSE.md)\
Parsa Vahidi, Omid G. Sani and Maryam M. Shanechi\
Shanechi Lab, University of Southern California
