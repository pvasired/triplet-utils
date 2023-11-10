# Utilities for fitting electrical stimulation spike sorting data

import numpy as np
import statsmodels.api as sm
import multiprocessing as mp
import copy
import collections
import jax
import jax.numpy as jnp
import optax
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multisite.fitting as fitting

@jax.jit
def activation_probs(x, w):
    """
    Activation probabilities using hotspot model.

    Parameters:
    w (n x d jnp.DeviceArray): Site weights
    x (c x d jnp.DeviceArray): Current levels

    Returns:
    p (c x 1 jnp.DeviceArray): Predicted probabilities
    """
    # w : site weights, n x d
    # x : current levels, c x d
    site_activations = jnp.dot(w, jnp.transpose(x)) # dimensions: n x c
    p_sites = jax.nn.sigmoid(site_activations) # dimensions : n x c
    p = 1 - jnp.prod(1 - p_sites, 0)  # dimensions: c

    return p

@jax.jit
def fisher_loss_array(probs_vec, transform_mat, jac_full, trials):
    """
    Compute the Fisher loss across the entire array summed across cells.

    Parameters:
    probs_vec (jnp.DeviceArray): The flattened array of probabilities across all meaningful
                            (cell, pattern) combinations
    transform_mat (jnp.DeviceArray): The transformation matrix to convert the trials array
                                to the transformed trials array for multiple cells on
                                the same pattern
    jac_full (jnp.DeviceArray): The full precomputed Jacobian matrix
    trials (jnp.DeviceArray): The input trials vector to be optimized

    Returns:
    loss (float): The whole array Fisher information loss
    """
    p_model = jnp.clip(probs_vec, a_min=1e-5, a_max=1-1e-5) # need to clip these to prevent
                                                            # overflow errors
    t = jnp.dot(transform_mat, trials).flatten()
    I_p = t / (p_model * (1 - p_model))

    # Avoiding creating the large diagonal matrix and storing in memory
    I_w = jnp.dot((jac_full.T * I_p), jac_full) / len(p_model)
    
    # Avoiding multiplying the matrices out and calculating the trace explicitly
    return jnp.sum(jnp.multiply(jac_full.T, jnp.linalg.solve(I_w, jac_full.T)))

def sample_spikes(p_true, t, error_rate_0=0, error_rate_1=0):
    """
    Helper function to sample spikes from a Bernoulli distribution.

    Parameters:
    p_true (np.ndarray AMPLITUDES X 1): True probabilities of spiking across amplitudes
                         for a given (cell, pattern)
    t (np.ndarray AMPLITUDES X 1): Number of trials across amplitudes for a given (cell, pattern)

    Returns:
    p_empirical_array (np.ndarray): Empirical probability of a spike across
                              amplitude for a given (cell, pattern)
    """
    p_true, t = np.array(p_true), np.array(t).astype(int)
    
    p_empirical = []
    for i in range(len(p_true)):
        # If there are no trials, set the empirical probability to 0.5
        if t[i] == 0:
            p_empirical += [0.5]
        
        # Else, sample from a Bernoulli distribution
        else:
            spikes = np.random.choice(np.array([0, 1]), 
                                                 p=np.array([(1-p_true[i])*(1-error_rate_0) + p_true[i]*error_rate_1, 
                                                             p_true[i]*(1-error_rate_1) + (1-p_true[i])*error_rate_0]), 
                                                 size=t[i])

            p_empirical += [np.mean(spikes)]
        
    p_empirical_array = np.array(p_empirical)

    return p_empirical_array

def sample_spikes_array(true_probs, trials, error_rate_0=0, error_rate_1=0,
                        NUM_THREADS=24):
    """
    Sample spikes across all cells and patterns using multiprocessing.

    Parameters:
    true_probs (np.ndarray CELLS X PATTERNS X AMPLITUDES): True probabilities of spikes
    trials (np.ndarray PATTERNS X AMPLITUDES): Number of trials
    NUM_THREADS (int): Number of threads to use for multiprocessing

    Returns:
    p_empirical_array (np.ndarray CELLS X PATTERNS X AMPLITUDES): Empirical probability of a spike across
                                    all cells and patterns
    """

    # Set up a list for multiprocessing
    input_list = []
    for i in range(len(true_probs)):
        for j in range(len(true_probs[i])):
            input_list += [(true_probs[i][j], trials[j], error_rate_0, error_rate_1)]
    
    # Run the multiprocessing
    pool = mp.Pool(processes=NUM_THREADS)
    results = pool.starmap_async(sample_spikes, input_list)
    mp_output = results.get()
    pool.close()

    return np.array(mp_output).reshape(true_probs.shape)

def get_performance_array(true_params, curr_probs, true_probs):
    """
    Helper function to compute performance across all cells and patterns.

    Parameters:
    true_params (np.ndarray CELLS X PATTERNS): True parameters
    curr_probs (np.ndarray CELLS X PATTERNS X AMPLITUDES): Current probabilities
    true_probs (np.ndarray CELLS X PATTERNS X AMPLITUDES): True probabilities

    Returns:
    error (float): RMSE across all cells and patterns
    """
    
    error = 0
    cnt = 0
    for i in range(len(true_params)):
        for j in range(len(true_params[i])):
            # If the true parameter is an integer, it is a degenerate case
            if type(true_params[i][j]) != int:
                error += np.sqrt(np.sum((curr_probs[i][j] - true_probs[i][j])**2) / len(true_probs[i][j]))
                cnt += 1

    return error / cnt

@jax.jit
def fisher_loss_max(probs_vec, transform_mat, jac_full, trials):
    """
    Compute the Fisher loss across the entire array, taking logsumexp()
    to minimize the worst case.

    Parameters:
    probs_vec (jnp.DeviceArray): The flattened array of probabilities across all meaningful
                            (cell, pattern) combinations
    transform_mat (jnp.DeviceArray): The transformation matrix to convert the trials array
                                to the transformed trials array for multiple cells on
                                the same pattern
    jac_full (jnp.DeviceArray): The full precomputed Jacobian matrix
    trials (jnp.DeviceArray): The input trials vector to be optimized

    Returns:
    loss (float): The whole array Fisher information loss, taking logsumexp() across
                  cells to minimize the worst case.
    """
    p_model = jnp.clip(probs_vec, a_min=1e-5, a_max=1-1e-5) # need to clip these to prevent
                                                            # overflow errors
    t = jnp.dot(transform_mat, trials).flatten()
    I_p = t / (p_model * (1 - p_model))

    # Avoiding creating the large diagonal matrix and storing in memory
    I_w = jnp.dot((jac_full.T * I_p), jac_full) / len(p_model)
    
    # Avoiding multiplying the matrices out and calculating the trace explicitly
    sum_probs = jnp.sum(jnp.multiply(jac_full.T, jnp.linalg.solve(I_w, jac_full.T)), axis=0)
    sum_cells = jnp.reshape(sum_probs, (-1, trials.shape[1])).sum(axis=-1)

    return jax.scipy.special.logsumexp(sum_cells)

def optimize_fisher_array(jac_full, probs_vec, transform_mat, T_prev, T, reg=None, 
                          step_size=1, n_steps=100, T_budget=5000, verbose=True):
    """
    Fisher optimization loop using optax and AdamW optimizer.

    Parameters:
    jac_full (jnp.DeviceArray): Full precomputed Jacobian matrix
    probs_vec (jnp.DeviceArray): Flattened array of all probabilities
                                 for all non-degenerate (cell, pattern)
                                 combinations
    transform_mat (jnp.DeviceArray): transformation matrix to convert trials
                                     array into correct shape matrix for 
                                     multiple cells on the same pattern
    T_prev (jnp.DeviceArray): The previously sampled trials array
    T (jnp.DeviceArray): The initialization for the to-be-optimized trials

    Returns:
    losses (np.ndarray): An array of losses per iteration of the optimization routine
    T (jnp.DeviceArray): The optimized trials matrix 
    """
    # Exponential decay of the learning rate.
    # scheduler = optax.exponential_decay(
    #     init_value=step_size, 
    #     transition_steps=1000,
    #     decay_rate=0.99,
    #     staircase=False)

    # Initialize the optimizer
    optimizer = optax.adamw(step_size)
    # optimizer = optax.lion(step_size/3)
    # optimizer = optax.sgd(learning_rate=scheduler)
    opt_state = optimizer.init(T)

    # Automatically set the regularization parameter if none is passed in
    if reg is None:
        init_function = fisher_loss_max(probs_vec, transform_mat, jac_full, T_prev + jnp.absolute(T))
        reg = init_function / 20000 # 20000 worked, 100000 too large

    # Update function for computing the gradient
    @jax.jit
    def update(jac_full, probs_vec, transform_mat, T_prev, T):
        # Adding special l1-regularization term that controls the total trial budget
        fisher_lambda = lambda T, jac_full, probs_vec, transform_mat, T_prev: fisher_loss_max(probs_vec, transform_mat, jac_full, T_prev + jnp.absolute(T)) + reg * jnp.absolute(jnp.sum(jnp.absolute(T)) - T_budget)

        grads = jax.grad(fisher_lambda)(T, jac_full, probs_vec, transform_mat, T_prev)
        
        return grads
    
    # Optimization loop
    losses = []
    for step in range(n_steps):
        if verbose:
            print(step)

        # Update the optimizer
        # start_grad = time.time()
        grads = update(jac_full, probs_vec, transform_mat, T_prev, T)
        # print(time.time() - start_grad)

        # start_update = time.time()
        updates, opt_state = optimizer.update(grads, opt_state, params=T)
        # print(time.time() - start_update)

        # start_apply = time.time()
        T = optax.apply_updates(T, updates)
        # print(time.time() - start_apply)
        
        # start_verbose = time.time()
        # If desired, compute the losses and store them
        if verbose:
            loss = fisher_loss_max(probs_vec, transform_mat, jac_full, T_prev + jnp.absolute(T))
            loss_tuple = (loss, jnp.sum(jnp.absolute(T)), loss + reg * jnp.absolute(jnp.sum(jnp.absolute(T)) - T_budget),
                            reg * jnp.absolute(jnp.sum(jnp.absolute(T)) - T_budget))
            print(loss_tuple)
            losses += [loss_tuple]
        # print(time.time() - start_verbose)

    return np.array(losses), T

def optimize_fisher_array_projected(jac_full, probs_vec, transform_mat, T_prev, T,
                                    step_size=1, n_steps=100, T_budget=5000, verbose=True):
    """
    Fisher optimization loop using optax and AdamW optimizer.
    Uses projected gradient descent on the L1-ball to enforce the trial budget.

    Parameters:
    jac_full (jnp.DeviceArray): Full precomputed Jacobian matrix
    probs_vec (jnp.DeviceArray): Flattened array of all probabilities
                                 for all non-degenerate (cell, pattern)
                                 combinations
    transform_mat (jnp.DeviceArray): transformation matrix to convert trials
                                     array into correct shape matrix for 
                                     multiple cells on the same pattern
    T_prev (jnp.DeviceArray): The previously sampled trials array
    T (jnp.DeviceArray): The initialization for the to-be-optimized trials

    Returns:
    losses (np.ndarray): An array of losses per iteration of the optimization routine
    T (jnp.DeviceArray): The optimized trials matrix 
    """
    # Exponential decay of the learning rate.
    # scheduler = optax.exponential_decay(
    #     init_value=step_size, 
    #     transition_steps=1000,
    #     decay_rate=0.99,
    #     staircase=False)

    # Initialize the optimizer
    optimizer = optax.adamw(step_size)
    # optimizer = optax.lion(step_size/3)
    # optimizer = optax.sgd(learning_rate=scheduler)
    opt_state = optimizer.init(T)

    # Update function for computing the gradient
    @jax.jit
    def update(jac_full, probs_vec, transform_mat, T_prev, T):
        fisher_lambda = lambda T, jac_full, probs_vec, transform_mat, T_prev: fisher_loss_max(probs_vec, transform_mat, jac_full, T_prev + jnp.absolute(T))
        grads = jax.grad(fisher_lambda)(T, jac_full, probs_vec, transform_mat, T_prev)
        return grads
    
    # Optimization loop
    losses = []
    for step in range(n_steps):
        if verbose:
            print(step)

        # Update the optimizer
        # start_grad = time.time()
        grads = update(jac_full, probs_vec, transform_mat, T_prev, T)
        # print(time.time() - start_grad)

        # start_update = time.time()
        updates, opt_state = optimizer.update(grads, opt_state, params=T)
        # print(time.time() - start_update)

        # start_apply = time.time()
        T = optax.apply_updates(T, updates)
        # print(time.time() - start_apply)

        # Project the trials onto the budget
        mu = jnp.flip(jnp.sort(jnp.absolute(T).flatten()))
        for j in range(len(mu)):
            if mu[j] - (jnp.sum(mu[:j+1]) - T_budget) / (j + 1) < 0:
                break

        theta = (jnp.sum(mu[:j]) - T_budget) / j
        T = jnp.maximum(jnp.absolute(T) - theta, 0)

        # start_verbose = time.time()
        # If desired, compute the losses and store them
        if verbose:
            loss = fisher_loss_max(probs_vec, transform_mat, jac_full, T_prev + jnp.absolute(T))
            loss_tuple = (loss, jnp.sum(jnp.absolute(T)))
            print(loss_tuple)
            losses += [loss_tuple]
        # print(time.time() - start_verbose)

    return np.array(losses), T

# Need to check modificiation of input arrays in this
def generate_input_list(all_probs_, amps_, trials_, w_inits_array, min_prob=0):
    """
    Generate input list for multiprocessing fitting of sigmoids
    to an entire array.
    
    Parameters:
    all_probs (cells x patterns x amplitudes np.ndarray): Probabilities
    amps (patterns x amplitudes x stimElecs np.ndarray): Amplitudes
    trials (patterns x amplitudes np.ndarray): Trials
    w_inits_array (cells x patterns np.ndarray of objects): Initial guesses
                                                            of parameters
                                                            
    Returns:
    input_list (list): formatted list ready for multiprocessing
    """
    all_probs = copy.deepcopy(all_probs_)
    amps = copy.deepcopy(amps_)
    trials = copy.deepcopy(trials_)

    input_list = []
    for i in range(len(all_probs)):
        for j in range(len(all_probs[i])):
            probs = all_probs[i][j]
            T = trials[j]
            X = amps[j]

            # Remove amplitudes that were not sampled
            good_T_inds = np.where(T > 0)[0]
            probs, X, T = copy.deepcopy(probs[good_T_inds]), copy.deepcopy(X[good_T_inds]), copy.deepcopy(T[good_T_inds])

            # If there are no valid probabilities, skip this (cell, pattern)
            if len(probs[probs > min_prob]) == 0:
                probs = np.array([])
                X = np.array([])
                T = np.array([])

            input_list += [(X, probs, T, w_inits_array[i][j])]

    return input_list

def fisher_sampling(probs_empirical, T_prev, amps, w_inits_array=None, t_final=None, 
                          budget=10000, reg=None, T_step_size=0.05, T_n_steps=5000, ms=[1, 2],
                          verbose=True, R2_cutoff=0, return_probs=False, trial_cap=25, entropy_buffer=0.5, 
                          entropy_samples=1, exploit_factor=0.75, zero_prob=0.01, slope_bound=20, NUM_THREADS=24,
                          min_prob=0):

    """
    Parameters:
    probs_empirical: cells x patterns x amplitudes numpy.ndarray of probabilities from g-sort
    T_prev: patterns x amplitudes numpy.ndarray of trials that have already been done
    amps: patterns x amplitudes x stimElecs numpy.ndarray of current amplitudes applied
    w_inits_array: cells x patterns numpy.ndarray(dtype=object) of lists containing initial guesses for fits
    t_final: numpy.ndarray of last optimal trial allocation
    budget: int of total number of trials to allocate
    reg: float of regularization parameter for Fisher optimization
    T_step_size: float of step size for Fisher optimization
    T_n_steps: int of number of steps for Fisher optimization
    ms: list of ints of number of sites to fit
    verbose: bool of whether to print out losses
    R2_cutoff: float of minimum R2 value to consider a fit valid
    return_probs: bool of whether to return the probabilities
    trial_cap: int of maximum number of trials to allocate to a single amplitude
    entropy_buffer: float of buffer around 0.5 to consider an amplitude as entropic
    entropy_samples: int of number of samples to allocate to entropic amplitudes
    exploit_factor: float of fraction of budget to allocate to Fisher optimization
    zero_prob: float of required probability at the origin
    slope_bound: float of maximum absolute value of slope
    NUM_THREADS: int of number of threads to use for multiprocessing
    min_prob: float of minimum probability to consider a valid trial

    Returns:
    T_new: patterns x amplitudes numpy.ndarray of new trials to perform
    w_inits_array: cells x patterns numpy.ndarray(dtype=object) of lists containing new initial guesses for fits
    t_final: numpy.ndarray of new optimal trial allocation

    OR if return_probs is True:
    T_new: patterns x amplitudes numpy.ndarray of new trials to perform
    w_inits_array: cells x patterns numpy.ndarray(dtype=object) of lists containing new initial guesses for fits
    t_final: numpy.ndarray of new optimal trial allocation
    probs_curr: cells x patterns x amplitudes numpy.ndarray of probabilities from new trial allocation
    params_curr: cells x patterns numpy.ndarray(dtype=object) of new parameters
    """

    print('Setting up data...')

    # Create the array of all initial guesses if none is passed in
    if w_inits_array is None:
        w_inits_array = np.zeros((probs_empirical.shape[0], probs_empirical.shape[1]), dtype=object)
        for i in range(len(w_inits_array)):
            for j in range(len(w_inits_array[i])):
                w_inits = []

                for m in ms:
                    w_init = np.array(np.random.normal(size=(m, amps[j].shape[1]+1)))
                    z = 1 - (1 - zero_prob)**(1/len(w_init))
                    w_init[:, 0] = np.clip(w_init[:, 0], None, np.log(z/(1-z)))
                    w_init[:, 1:] = np.clip(w_init[:, 1:], -slope_bound, slope_bound)
                    w_inits.append(w_init)

                w_inits_array[i][j] = w_inits

    print('Generating input list...')

    # Set up the data for multiprocess fitting
    input_list = generate_input_list(probs_empirical, amps, T_prev, w_inits_array, min_prob=min_prob)

    print('Fitting dataset...')
    pool = mp.Pool(processes=NUM_THREADS)
    results = pool.starmap_async(fitting.fit_surface, input_list)
    mp_output = results.get()
    pool.close()

    params_curr = np.zeros((probs_empirical.shape[0], probs_empirical.shape[1]), dtype=object)
    w_inits_array = np.zeros((probs_empirical.shape[0], probs_empirical.shape[1]), dtype=object)
    R2s = np.zeros((probs_empirical.shape[0], probs_empirical.shape[1]))
    probs_curr = np.zeros(probs_empirical.shape)

    cnt = 0
    for i in range(len(probs_empirical)):
        for j in range(len(probs_empirical[i])):
            params_curr[i][j] = mp_output[cnt][0][0]
            w_inits_array[i][j] = mp_output[cnt][1]
            R2s[i][j] = mp_output[cnt][0][2]
            
            probs_curr[i][j] = fitting.sigmoidND_nonlinear(
                                    sm.add_constant(amps[j], has_constant='add'), 
                                    params_curr[i][j])

            cnt += 1

    print('Calculating Jacobian...')

    jac_dict = collections.defaultdict(dict)
    transform_mat = []
    probs_vec = []
    num_params = 0
    entropy_inds = []

    for i in range(len(params_curr)):
        for j in range(len(params_curr[i])):
            if ~np.all(params_curr[i][j][:, 0] == -np.inf) and R2s[i][j] >= R2_cutoff:
                X = jnp.array(sm.add_constant(amps[j], has_constant='add'))
                jac_dict[i][j] = jax.jacfwd(activation_probs, argnums=1)(X, jnp.array(params_curr[i][j])).reshape(
                                                (len(X), params_curr[i][j].shape[0]*params_curr[i][j].shape[1]))  # c x l
                num_params += jac_dict[i][j].shape[1]

                entropy_inds_j = np.where((probs_curr[i][j] >= 0.5 - entropy_buffer) & (probs_curr[i][j] <= 0.5 + entropy_buffer))[0]
                for ind in entropy_inds_j:
                    entropy_inds.append((j, ind))

                transform = jnp.zeros(len(T_prev))
                transform = transform.at[j].set(1)
                transform_mat.append(transform)     # append a e-vector (512)

                probs_vec.append(probs_curr[i][j])  # append a c-vector (80)

    if len(probs_vec) == 0:
        raise ValueError("No valid probabilities found.")
    
    entropy_inds = np.array(entropy_inds)
    transform_mat = jnp.array(transform_mat, dtype='float32')
    probs_vec = jnp.array(jnp.hstack(probs_vec), dtype='float32')

    jac_full = jnp.zeros((len(probs_vec), num_params))
    counter_axis0 = 0
    counter_axis1 = 0
    for i in jac_dict.keys():
        for j in jac_dict[i].keys():
            next_jac = jac_dict[i][j]

            jac_full = jac_full.at[counter_axis0:counter_axis0+next_jac.shape[0], counter_axis1:counter_axis1+next_jac.shape[1]].set(next_jac)

            counter_axis0 += next_jac.shape[0]
            counter_axis1 += next_jac.shape[1]

    jac_full = jnp.array(jac_full, dtype='float32')
    print('Optimizing trials...')

    if t_final is None:
        random_init = np.random.choice(len(T_prev.flatten()), size=int(budget*exploit_factor))
        T_new_init = jnp.array(np.bincount(random_init, minlength=len(T_prev.flatten())).astype(int).reshape(T_prev.shape), dtype='float32')

    else:
        T_new_init = jnp.array(jnp.absolute(jnp.array(t_final)), dtype='float32')

    losses, t_final = optimize_fisher_array(jac_full, probs_vec, transform_mat, jnp.array(T_prev, dtype='float32'), T_new_init, 
                                                    step_size=T_step_size, n_steps=T_n_steps, reg=reg, T_budget=budget*exploit_factor,
                                                    verbose=verbose)

    if verbose:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].plot(losses[:, 0])
        axs[0].set_ylabel('Fisher Loss (A-optimality)')
        axs[1].plot(losses[:, 1])
        axs[1].set_ylabel('Total Trials')
        axs[2].plot(losses[:, 2])
        axs[2].set_ylabel('Regularized Loss, reg=' + str(reg))

        fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
        plt.savefig(f'plots_CL.png', dpi=300)
        plt.show(block=False)

    T_new = jnp.round(jnp.absolute(t_final), 0)

    T_new = np.array(T_new)
    capped_inds = np.where(T_new + T_prev >= trial_cap)
    T_new[capped_inds[0], capped_inds[1]] = np.clip(trial_cap - T_prev[capped_inds[0], capped_inds[1]],
                                                    0, None)

    if np.sum(T_new) < budget:
        random_entropy = np.random.choice(len(entropy_inds), 
                                          size=int((budget - np.sum(T_new))/entropy_samples))
        for ind in random_entropy:
            T_new[entropy_inds[ind][0]][entropy_inds[ind][1]] += entropy_samples

    capped_inds = np.where(T_new + T_prev >= trial_cap)
    T_new[capped_inds[0], capped_inds[1]] = np.clip(trial_cap - T_prev[capped_inds[0], capped_inds[1]],
                                                    0, None)

    if return_probs:
        return T_new.astype(int), w_inits_array, np.array(t_final), probs_curr, params_curr
    
    else:
        return T_new.astype(int), w_inits_array, np.array(t_final)