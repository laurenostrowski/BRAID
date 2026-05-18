# BRAIDModel formulation
The model learning for `BRAIDModel` is done in 2 main stages and an optional preprocessing stage as follows: 


BRAID has two main stages, each incorporating two recursions implemented by `MainModel` class, as explained below:

Note BRAID can learn two sets of recursions termed ''predictior'' and ''generator''. The predictor ($RNN$) learns inference of 1-step-ahead predicted latents'observations as outlined in this document. The generative model ($RNN_{fw}$) propagates the latents inferred by $RNN$ ahead in time to generate data accroding to the learned intrinsic dynamics of mapping $A_{fw}$. Here for brevity we provide formulations in terms of the predictor model. Please refer to the manuscript for more details.

1. In stage 1, we learn the parameters $A^{(1)}(\cdot)$/$A^{(1)}_{fw}(\cdot)$, $K^{(1)}(\cdot)$/$K^{(1)}_{fw}(\cdot)$, $C^{(1)}_z(\cdot)$ of the following $RNN$/$RNN_{fw}$:

    RNN 1:

$$x^{(1)}_{k+1} = A^{(1)}(x^{(1)}_k) + K^{(1)}( y_k, u_k )$$

$$\hat{z}^{(1)}_k = C_z^{(1)}( x^{(1)}_k, u_k)$$

and estimate its latent state $x^{(1)}_k\in\mathbb{R}^{n_1}$, while minimizing the negative log-likelihood (NLL) of predicting the behavior $z_k$ as $\hat{z}^{(1)}_k$. This $RNN$/$RNN_{fw}$ is implemented as an `RNNModel` object with $y_k$ and $u_k$ as the inputs and $\hat{z}^{(1)}_k$ as the output and the state dimension of $n_1$ as specified by the user. `RNNModel` implements each of the $RNN$/$RNN_{fw}$ parameters, $A^{(1)}(\cdot)$/$A^{(1)}_{fw}(\cdot)$, $K^{(1)}(\cdot)$/$K^{(1)}_{fw}(\cdot)$, and $C^{(1)}_z(\cdot)$, as a multilayer feed-forward neural network implemented via the `RegressionModel` class. 

To also predict neural activity $y_k$, After RNN 1 is fitted, MainModel fits a readout from the extracted latent states as

$$\hat{y}_k = C_y^{(1)}( x^{(1)}_k, u_k )$$

2. In the third optimization step, we learn the parameters $A^{(2)}(\cdot)$/$A^{(2)}_{fw}(\cdot)$, $K^{(2)}(\cdot)$/$K^{(2)}_{fw}(\cdot)$, and $C^{(2)}_y(\cdot)$ of the following $RNN$/$RNN_{fw}$:  

   RNN 2:

$$x^{(2)}_{k+1} = A^{(2)}(x^{(2)}_k) + K^{(2)}( y_k, u_k, x^{(1)}_{k} )$$

$$\hat{y}_k = C_y^{(2)}( x^{(2)}_k, u_{k})$$

and estimate its latent state $x^{(2)}_k$ while minimizing the aggregate neural prediction negative log-likelihood, which also takes into account the negative log-likelihood (NLL) obtained from stage 1 via the $C_y^{(1)}( x^{(1)}_k, u_k)$ and computed using the previously learned parameter $C_y^{(1)}$ and the previously extracted states $x_k^{(1)}$. This RNN is also implemented as an `RNNModel` object with the concatenation of $y_k$, $u_k$ and $x^{(1)}_k$ as the inputs and the predicted neural activity as the output. The NLL for predicting neural activity from stage 1 also provided as input, to allow formation of aggregate neural prediction NLL as the loss. `RNNModel` again implements each of the RNN parameters, $A^{(2)}(\cdot)$/$A^{(2)}_{fw}(\cdot)$, $K^{(2)}(\cdot)$/$K^{(2)}_{fw}(\cdot)$, and $C^{(2)}_y(\cdot)$, as a multilayer feed-forward neural network implemented via the `RegressionModel` class. 


Next, another readout from the extracted latent states i.e., $x^{(2)}_k$ can be fitted as:

$$\hat{z}_k = C_z^{(2)}(x^{(2)}_k, u_k)$$

while minimizing the aggregate behavior prediction negative log-likelihood taking into account the negative log-likelihood (NLL) obtained from stage 1 via the $C_z^{(1)}( x^{(2)}_k, u_k)$. This step can complement behavior predictions from stage 1 in case $n_1$ is set to a low value.

For additional details in these steps, please read **Methods** in the paper.


To exclude non-encoded dynamics that only explain behavior $z_k$, we may use an additioanl preprocessing stage. In this case, `BRAIDModel`, before starting main stages explained above, fits a preprocessing $RNN$/$RNN_{fw}$:

$$x^{(0)}_{k+1} = A^{(0)}(x^{(0)}_k) + K^{(0)}( y_k, u_k)$$

$$\hat{y}_k = C_y^{(0)}( x^{(0)}_k, u_{k})$$

to learn all neural-relevant dynamics i.e., x^{(0)}_k. Then a readout from these states give the preprocessed behavior signal to be used in stage 1

$$\hat{z}_k = C_z^{(0)}( x^{(0)}_k)$$

# Forcasting with generative RNNs learned via multi-step-ahead optimization
MainModel can simultaneously optimizes predictions of multiple steps ahead in the future. For example if the parameter step_ahead is set to [1,2,3], then the loss function includes 1-step, 2-step and 3-step ahead predicition of $z_k$ and $y_k$. This holds for the second stage and preprocessing stage explained below as well.In case user specifies optimization of multi-steps-ahead (parameter steps_ahead), then a generative recursion will be additionally learned for each starge to propagate dynamics forward in time with paramteres $A_{fw}(.)$ and $K_{fw}(.)$  complementing the inferred 1-step-ahead predictions. Please refer to **Methods** section in the manuscript for details.

# Objective function of each optimization step
For each optimization step, we minimized the mean-squared-error loss between prediction and true values which is equivalent to the negative log-likelihood for isotropic Gaussian-distributed observations.


# BRAIDModel arguments

Here we explain the main inputs BRAIDModel takes as arguments set by the user.

Initialize an BRAIDModel as model = BRAIDModel()

block_samples=128 # RNN sequence length
batch_size=32,  # Training batch size

Training the model:


Y: neural data (dimension X time)  
Z=None: Behavior data (dimension X time)  
U=None: External input data (dimension X time)  
nx=None: Total state dimension 
n1=None: stage dimension in stage 1
n3=None: behavior specific dynamics dimension (RNN 3, x3). This only applies in case nx > n1.  
n_pre=None: State dimension of the preprocessing RNN (x0)  
create_val_from_training = False: If True, will create the validation data by cutting it off of the training data  

Optionally, can provide validation data instead:  
Y_validation = None: if provided will use to compute loss on validation (unless create_val_from_training=True)  
Z_validation = None: if provided will use to compute loss on validation (unless create_val_from_training=True)  
U_validation = None: if provided will use to compute loss on validation (unless create_val_from_training=True)   
steps_ahead=None, # List of ints (None take as [1]), indicating the number of steps ahead to generate from the model. used to construct training loss and predictions.  

Setting for parameters:  
A_args = {}, K_args = {}, Cy_args = {}, Cz_args = {},    # Both stage 1 and 2 params  
A1_args = None, K1_args = None, Cy1_args = None, Cz1_args = None, # Stage 1 params  
A2_args = None, K2_args = None, Cy2_args = None, Cz2_args = None, # Stage 2 params  

args_base = None: a dictionary containing remaining arguments as defined in MainModel.fit(). Please check the method or details  
noFT = False: If True, will omit the feedthrough term (u in the readouts)
