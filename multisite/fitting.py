# Utilities for fitting electrical stimulation spike sorting data

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import sklearn.model_selection as model_selection
import statsmodels.api as sm
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from itertools import chain, combinations
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multisite.multielec_utils as mutils

def convertToBinaryClassifier(probs, num_trials, amplitudes, degree=1, 
                              interaction=True):
    """
    Converts input g-sort data of probabilities, trials, and 
    amplitudes. Includes a functionality for converting to data to a
    polynomial transformation which is now largely deprecated.

    Parameters:
    probs (N x 1 np.ndarray): The input probabilities
    num_trials (N x 1 np.ndarray): The input number of trials used for
                                   each probability
    amplitudes (N x k np.ndarray): The amplitude (vectors) of current
                                   stimulus. Supports multi-electrode
                                   stimulation.
    
    Optional Arguments:
    degree (int): The polynomial transformation degree, default 1
    interaction (bool): Whether or not to include cross terms in the
                        construction of the polynomial transform.
    
    Return:
    X (np.ndarray): The binary classifier inputs with constant term
                    and possibly polynomial tranformation terms added.
                    The shape of this array is related to the number
                    of trials per amplitude, the number of amplitudes,
                    the stimulation current vector sizes, and the 
                    polynomical transformation degree.

    y (np.ndarray): The binary classifier outputs consisting of 0s and
                    1s, with the same length as X.
    """
    
    y = []
    X = []
    num_trials = num_trials.astype(int) # convert to integer
    for j in range(len(amplitudes)):
        # Calculate the number of 1s and 0s for each probability
        num1s = int(np.around(probs[j] * num_trials[j], 0))
        num0s = num_trials[j] - num1s

        # Append all the amplitudes for this probability
        X.append(np.tile(amplitudes[j], (num_trials[j], 1)))

        # Append the 0s and 1s for this probability
        y.append(np.concatenate((np.ones(num1s), np.zeros(num0s))))
    
    # If desired, perform a polynomial transformation with cross terms
    if interaction == True:
        poly = PolynomialFeatures(degree)
        X = poly.fit_transform(np.concatenate(X))

    # If no cross terms are desired
    else:
        X = noInteractionPoly(np.concatenate(X), degree)
    
    y = np.concatenate(y)
    
    return X, y

def noInteractionPoly(amplitudes, degree):
    """
    Constructs a non-interacting polynomial transformation with no
    cross terms included.

    Parameters:
    amplitudes (np.ndarray): Raw amplitude vectors
    degree (int): Degree of polynomial transformation


    Returns:
    X (np.ndarray): Same length as amplitudes, but expanded along 
                    axis 1 according to the degree of the transform
    """
    # For each degree, add the next non-interacting polynomial
    higher_order = []
    for i in range(degree):
        higher_order.append(amplitudes**(i+1))
    
    # Add a constant column to the output
    return sm.add_constant(np.hstack(higher_order), has_constant='add')

def negLL(params, *args):
    """
    Compute the negative log likelihood for a logistic regression
    binary classification task.

    Parameters:
    params (np.ndarray): Weight vector to be fit, same dimension as
                         axis=1 dimension of X (see below)
    *args (tuple): X (np.ndarray) output of convertToBinaryClassifier,
                   y (np.ndarray) output of convertToBinaryClassifier,
                   verbose (bool) increases verbosity
                   method: regularization method. 'MAP' (maximum a 
                           posteriori), 'l1', and 'l2' are supported.
    
    Returns:
    negLL (float): negative log likelihood of the data given the 
                   current parameters, possibly plus a regularization
                   term.
    """
    X, y, verbose, method = args
    
    w = params
    
    # Get predicted probability of spike using current parameters
    yPred = 1 / (1 + np.exp(-X @ w))
    yPred[yPred == 1] = 0.999999     # some errors when yPred is 
                                     # exactly 1 due to taking 
                                     # log(1 - 1)
    yPred[yPred == 0] = 0.000001

    # Calculate negative log likelihood
    NLL = -np.sum(y * np.log(yPred) + (1 - y) * np.log(1 - yPred))     

    if method == 'MAP':
        # penalty term according to MAP regularization
        penalty = 0.5 * (w - mu) @ np.linalg.inv(cov) @ (w - mu)
    elif method == 'l1':
        # penalty term according to l1 regularization
        penalty = l1_reg*np.linalg.norm(w, ord=1)
    elif method == 'l2':
        # penalty term according to l2 regularization
        penalty = l2_reg*np.linalg.norm(w)
    else:
         penalty = 0

    if verbose:
        print(NLL, penalty)

    return(NLL + penalty)

def negLL_hotspot(params, *args):
    """
    Compute the negative log likelihood for a logistic regression
    binary classification task assuming the hotpot model of activation.

    Parameters:
    params (np.ndarray): Weight vector to be fit, same dimension as
                         axis=1 dimension of X (see below)
    *args (tuple): X (np.ndarray) output of convertToBinaryClassifier,
                   y (np.ndarray) output of convertToBinaryClassifier,
                   verbose (bool) increases verbosity
                   method (str): regularization method. 'l1', 'l2', and
                                 'MAP' with multivariate Gaussian prior 
                                 are supported.
                   reg (float): regularization parameter
                                In the case of MAP, reg consists of 
                                (regmap, mu, cov)
                                where regmap is a constant scalar
                                      mu is the mean vector
                                      cov is the covariance matrix

    Returns:
    negLL (float): negative log likelihood of the data given the 
                   current parameters, possibly plus a regularization
                   term.
    """
    X, y, verbose, method, reg = args
    
    w = params.reshape(-1, X.shape[-1]).astype(float)

    # Negative log-likelihood calculation for hotspot activation
    prod = np.ones(len(X))
    for i in range(len(w)):
        prod *= (1 + np.exp(X @ w[i].T))
    prod -= 1

    prod[prod < 1e-10] = 1e-10  # prevent divide by 0 errors

    ### Deprecated calculation (gives same results but slower) ###
    # yPred2 = 1 / (1 + np.exp(-np.log(prod)))

    # Get predicted probability of spike using current parameters
    # response_mat = 1 / (1 + np.exp(-X @ w.T))
    # yPred = 1 - np.multiply.reduce(1 - response_mat, axis=1)

    # print(np.sum(np.absolute(yPred - yPred2)))
    # yPred[yPred == 1] = 0.999999     # some errors when yPred is exactly 1 due to taking log(1 - 1)
    # yPred[yPred == 0] = 0.000001
    
    # negative log likelihood for logistic
    # NLL = -np.sum(y * np.log(yPred) + (1 - y) * np.log(1 - yPred)) 
    ###

    # Calculate negative log likelihood
    NLL2 = np.sum(np.log(1 + prod) - y * np.log(prod))

    # Add the regularization penalty term if desired
    if method == 'l1':
        # penalty term according to l1 regularization
        penalty = reg*np.linalg.norm(w.flatten(), ord=1)
    elif method == 'l2':
        # penalty term according to l2 regularization
        penalty = reg/2*np.linalg.norm(w.flatten())**2
    elif method == 'MAP':
        regmap, mu, cov = reg
        # penalty term according to MAP with Gaussian prior
        penalty = regmap * 0.5 * (params - mu) @ np.linalg.inv(cov) @ (params - mu)
    else:
        penalty = 0

    if verbose:
        print(NLL2, penalty)
        
    return(NLL2 + penalty)

def all_combos(iterable):
    """
    Compute the 'powerset' of an iterable defined as:
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    
    Parameters:
    iterable: An iterable list or np.ndarray

    Returns:
    powerset: The powerset, or another iterable consisting of all 
              combinations of elements from the input iterable.
    """
    s = list(iterable)    
    return list(chain.from_iterable(
                        combinations(s, r) for r in range(len(s)+1)))

def negLL_hotspot_jac(params, *args):
    """
    Manually computed jacobian of negative log likelihood function
    assuming a hotspot model of activation. Manual gradients greatly
    improve runtime.

    Parameters:
    params (np.ndarray): Weight vector to be fit, same dimension as
                         axis=1 dimension of X (see below)
    *args (tuple): X (np.ndarray) output of convertToBinaryClassifier,
                   y (np.ndarray) output of convertToBinaryClassifier,
                   verbose (bool) increases verbosity
                   method (str): regularization method. 'l2' 
                                 and MAP are supported.
                   reg (float): regularization parameter
                                In the case of MAP, reg is as above
                                in negLL_hotspot()

    Returns:
    grad (np.ndarray): jacobian of negative log likelihood, same shape
                       as params
    """
    X, y, verbose, method, reg = args
    w = params.reshape(-1, X.shape[-1]).astype(float)
    
    # Complicated manual jacobian calculation, was verified
    # to produce the same results as automated differentiation methods
    prod = np.ones(len(X))
    for i in range(len(w)):
        prod = prod * (1 + np.exp(X @ w[i].T))
    prod = prod - 1

    prod[prod < 1e-10] = 1e-10  # prevent divide by 0 errors

    factors = np.zeros((len(w), len(X)))
    for i in range(len(w)):
        other_weights = np.setdiff1d(np.arange(len(w), dtype=int), i)
        other_combos = all_combos(other_weights)
        for j in range(len(other_combos)):
            other_combo = np.array(other_combos[j])
            if len(other_combo) > 0:
                factors[i] = factors[i] + np.exp(X @ np.sum(
                                            w[other_combo], axis=0).T)

    factors = factors + 1

    grad = np.zeros_like(w, dtype=float)
    for i in range(len(w)):
        term1 = X.T @ (1 / (1 + np.exp(-X @ w[i].T)))
        term2 = -X.T @ (y * np.exp(X @ w[i].T) * factors[i] / prod)

        grad[i] = term1 + term2

    grad = grad.ravel()

    # penalty term according to l2 regularization
    if method == 'l2':
        grad += reg * params

    # penalty term according to MAP
    elif method == 'MAP':
        regmap, mu, cov = reg
        grad += regmap  * (np.linalg.inv(cov) @ (params - mu)).flatten()

    return grad

def get_monotone_probs_and_amps(amplitudes,probs_,trials,n_amps_blank=0, st=0.5):
    """
    A utility function that returns the set of amplitudes and probabilities
    that satisfy the monotone requirement.

    TODO: document.
    """
    probs = copy.deepcopy(probs_)

    # Zero out the first few amplitudes
    probs[0:n_amps_blank] = 0
    mono_inds = np.argwhere(enforce_noisy_monotonicity(probs, st=st)).flatten()

    return amplitudes[mono_inds],probs[mono_inds], trials[mono_inds]

def enforce_noisy_monotonicity(probs, st=.5, noise_limit=.8):
    """
    Enforces monotonicity in the raw probability data. Finds indices that 
    violate monotonicity and excludes them in a final set for fitting.

    Code written by Jeff Brown.

    TODO: document.
    """
    J_array = []
    max_value = st
    trigger = False

    for i in range(len(probs)):

        if probs[i] >= max_value*noise_limit:
            max_value = probs[i]
            trigger = True
            J_array += [1]
        else:

            if not trigger:
                J_array += [1]
            else:
                J_array += [0]

    J_array = np.array(J_array).astype(np.int16)

    if J_array[0] == 1 and sum(J_array) == 1:
        J_array[-1] = 1

    return J_array

def disambiguate_sigmoid(sigmoid_, noise_limit = 0.0, thr_prob=0.5):
    """
    Utility for disambiguating 0/1 probability for g-sort output.
    The function converts 0s to 1s if the maximum probability 
    exceeds some spontaneous limit. For all amplitudes with
    magnitude greater than the amplitude that reached the 
    spontaneous limit, all probabilities below some noise 
    threshold are converted from 0s to 1s.
    
    Parameters:
    sigmoid_ (np.ndarray): Array of probabilities sorted according to
                           increasing magnitudes of amplitudes
    spont_limit (float): Maximum spontaneous activity threshold
    noise_limit (float): Threshold below which 0s are converted to 1s
    thr_prob (float): Value for determining where to start flipping
                      probabilities from 0 to 1

    Returns:
    sigmoid (np.ndarray): 0/1 disambiguated sigmoid with same shape
                          as sigmoid_
    
    """
    sigmoid = copy.copy(sigmoid_)
    if np.max(sigmoid) < thr_prob:
        return sigmoid
    above_limit = np.argwhere(sigmoid >= thr_prob).flatten()

    min_ind = np.amin(above_limit)
    sigmoid[min_ind:][sigmoid[min_ind:] <= noise_limit] = 1
    
    return sigmoid

# Need to check modifying array in this:
def disambiguate_fitting(X_expt_, probs_, T_, w_inits,
                         verbose=False, thr=0.5, pm=0.3,
                         spont_limit=0.2, reg_method='none',
                         reg=0):

    X_expt = copy.deepcopy(X_expt_)
    probs = copy.deepcopy(probs_)
    T = copy.deepcopy(T_)

    good_inds = np.where((probs > spont_limit) & (probs != 1))[0]
    zero_inds = np.where((probs <= spont_limit) | (probs == 1))[0]

    if len(zero_inds) > 0:
        params, _, _ = fit_surface(X_expt[good_inds], probs[good_inds], 
                                    T[good_inds], w_inits, reg_method, reg,
                                    verbose=verbose)

        probs_pred_zero = sigmoidND_nonlinear(sm.add_constant(X_expt[zero_inds], 
                                                            has_constant='add'),
                                                params)
        
        good_zero_inds = zero_inds[np.where(probs_pred_zero < thr - pm)[0]]
        good_inds_tot = np.concatenate([good_inds, good_zero_inds])

        return X_expt[good_inds_tot], probs[good_inds_tot], T[good_inds_tot]

    else:
        return X_expt, probs, T

    # # fig = plt.figure()
    # # fig.clear()
    # # ax = Axes3D(fig, auto_add_to_figure=False)
    # # fig.add_axes(ax)
    # # plt.xlabel(r'$I_1$ ($\mu$A)', fontsize=16)
    # # plt.ylabel(r'$I_2$ ($\mu$A)', fontsize=16)
    # # plt.xlim(-1.8, 1.8)
    # # plt.ylim(-1.8, 1.8)
    # # ax.set_zlim(-1.8, 1.8)
    # # ax.set_zlabel(r'$I_3$ ($\mu$A)', fontsize=16)

    # # scat = ax.scatter(X_expt[:, 0], 
    # #             X_expt[:, 1],
    # #             X_expt[:, 2], marker='o', c=probs_pred, s=20, alpha=0.8, vmin=0, vmax=1)

    # # plt.show()

    # flip_inds = np.setdiff1d(np.arange(len(probs), dtype=int), mid_inds)
    # for i in flip_inds:
    #     LL_flip = (1 - probs[i]) * np.log(probs_pred[i]) + probs[i] * np.log(1 - probs_pred[i])
    #     LL_stay = probs[i] * np.log(probs_pred[i]) + (1 - probs[i]) * np.log(1 - probs_pred[i])

    #     if LL_flip > LL_stay:
    #         probs[i] = 1 - probs[i]

    # return probs

# Numpy version of multisite.closed_loop.activation_probs()
def sigmoidND_nonlinear(X, w):
    """
    N-dimensional nonlinear sigmoid computed according to multi-
    hotspot model.
    
    Parameters:
    X (np.ndarray): Input amplitudes
    w (np.ndarray): Weight vector matrix

    Returns:
    response (np.ndarray): Probabilities with same length as X
    """
    response_mat = 1 / (1 + np.exp(-X @ w.T))
    response = 1 - np.multiply.reduce(1 - response_mat, axis=1)
    return response

def selectivity_triplet(ws, targets, curr_min=-1.8, curr_max=1.8, num_currs=40):
    """
    Returns the selectivity score for a list of weights ws and a
    list of target indices targets. The selectivity score is
    product over target probabilities times 
    1 - (maximum non-target pobability) at a given current amplitude.

    Parameters:
    ws (list): List of weight matrices for each target
    targets (list): List of target indices
    curr_min (float): Minimum current amplitude
    curr_max (float): Maximum current amplitude
    num_currs (int): Number of current amplitudes to evaluate

    Returns:
    Tuple (selec_triplet, selec_1elec)
    selec_triplet (float): Selectivity score for triplet stimulation
    selec_1elec (float): Selectivity score for single electrode stimulation
    """
    # Create the amplitude matrices for triplet and single electrode data
    # on the same set of stimulating electrodes
    I1 = np.linspace(curr_min, curr_max, num_currs)
    I2 = np.linspace(curr_min, curr_max, num_currs)
    I3 = np.linspace(curr_min, curr_max, num_currs)

    # Triplet amplitude matrices are all combinations of currents
    X_triplet = sm.add_constant(np.array(np.meshgrid(I1, I2, I3)).T.reshape(-1,3), has_constant='add')

    # Single electrode amplitude matrices have only one nonzero column per row
    X_1elec = []
    for i in range(3):
        X_elec = np.zeros((len(I1), 3))
        X_elec[:, i] = I1
        X_1elec.append(X_elec)

    X_1elec = sm.add_constant(np.vstack(X_1elec), has_constant='add')

    # Compute the selectivity score for triplet and single electrode stimulation

    # Compute selectivity score for triplet stimulation
    selec_product_triplet_t = []
    selec_product_triplet_nt = []
    for i in range(len(ws)):
        if i in targets:
            # If target, compute the probability of the target
            selec_product_triplet_t.append(sigmoidND_nonlinear(X_triplet, ws[i]))
        else:
            # If non-target, compute 1 - probability of non-target
            selec_product_triplet_nt.append(1 - sigmoidND_nonlinear(X_triplet, ws[i]))

    selec_product_triplet_t = np.array(selec_product_triplet_t)
    selec_product_triplet_nt = np.array(selec_product_triplet_nt)

    # Take the product of all targets and multiply it by 
    # 1 - (maximum non-target probability) at each amplitude
    # Take the maximum of this product over all amplitudes
    selec_triplet = np.amax(np.multiply.reduce(selec_product_triplet_t) * 
                            np.amin(selec_product_triplet_nt, axis=0))

    # Compute selectivity score for single electrode with same algo
    selec_product_1elec_t = []
    selec_product_1elec_nt = []
    for i in range(len(ws)):
        if i in targets:
            selec_product_1elec_t.append(sigmoidND_nonlinear(X_1elec, ws[i]))
        else:
            selec_product_1elec_nt.append(1 - sigmoidND_nonlinear(X_1elec, ws[i]))

    selec_product_1elec_t = np.array(selec_product_1elec_t)
    selec_product_1elec_nt = np.array(selec_product_1elec_nt)
    selec_1elec = np.amax(np.multiply.reduce(selec_product_1elec_t) * 
                            np.amin(selec_product_1elec_nt, axis=0))

    return selec_triplet, selec_1elec

def enforce_3D_monotonicity(index, Xdata, ydata, k=2, 
                            percentile=1.0, num_points=20,
                            dist_thr=0.15):
    """
    Enforces monotonicity in the raw probability. For a given
    index in the raw data, determine if that index violates
    a monotonic radial constraint. Finds points along a line
    extending from the origin to the point at index and finds
    the nearest points to that line. If the probability at
    the index is greater than the percentile * maximum probability
    of the nearest points, then the point is considered to be
    monotonic.

    Parameters:
    index (int): Index of the point to be tested
    Xdata (np.ndarray): Input amplitudes
    ydata (np.ndarray): Output probabilities
    percentile (float): Percentile of the maximum probability of
                        the nearest points to be compared to the
                        probability at the index
    num_points (int): Number of points to sample along the line
    dist_thr (float): Distance threshold for determining the nearest
                      points

    Returns:
    bool: True if the point is monotonic, False otherwise
    """
    # If the point is the origin, it is automatically monotonic
    point = Xdata[index]
    if np.linalg.norm(point) == 0:
        return True

    # Get the direction of the line extending from the origin to the point    
    direction = point / np.linalg.norm(point)

    # Get the nearest points to the line
    scaling = np.linspace(0, np.linalg.norm(point), num_points)
    closest_line = []
    for j in range(len(scaling)):
        curr = scaling[j] * direction
        dists = cdist(Xdata, curr[:, None].T).flatten()
        closest_inds = np.setdiff1d(np.where(dists <= dist_thr)[0], index)
        # closest_inds = np.setdiff1d(np.argsort(dists)[:k], index)

        closest_line.append(closest_inds)

    # Get the unique indices of the nearest points
    line_inds = np.unique(np.concatenate(closest_line))

    # If there are no nearest points, return False
    if len(line_inds) > 0:
        # If the probability at the index is greater than the percentile
        # * maximum probability of the nearest points, return True
        if ydata[index] >= percentile * np.amax(ydata[line_inds]):
            return True
        
        else:
            return False
    
    else:
        return False

def fit_surface(X_expt, probs, T, w_inits_, reg_method='none', reg=0,
                        R2_thresh=0.05, zero_prob=0.01, verbose=False,
                        method='L-BFGS-B', jac=negLL_hotspot_jac,
                        opt_verbose=False, slope_bound=20):
    """
    Fitting function for fitting surfaces to nonlinear data with multi-hotspot model.
    This function is primarily a wrapper for calling get_w() in the framework of 
    early stopping using the McFadden pseudo-R2 metric.

    Parameters:
    X_expt (N x d np.ndarray): Input amplitudes
    probs (N x 1 np.ndarray): Probabilities corresponding to the amplitudes
    T (N x 1 np.ndarray): Trials at each amplitude
    w_inits (list): List of initial guessses for each number of hotspots. Each element
                    in the list is a (m x (d + 1)) np.ndarray with m the number of 
                    hotspots. This list should be generated externally.
    R2_thresh (float): Threshold used for determining when to stop adding hotspots
    zero_prob (float): Value for what the probability should be forced to be below
                       at an amplitude of 0-vector
    verbose (bool): Increases verbosity
    method (string): Method for optimization according to constrained optimization
                     methods available in scipy.optimize.minimize
    jac (function): Jacobian function if manually calculated
    reg_method (string): Regularization method. 'l2' is supported
    reg (float): Regularization parameter value
    opt_verbose (bool): Increases verbosity of optimization procedure
    slope_bound (float): Bound on the absolute value of the slope parameters

    Returns:
    opt (tuple): Tuple containing the optimal parameters, the negative log likelihood
                    of the optimal parameters, and the McFadden pseudo-R2 value of the
                    optimal parameters
    w_inits (list): The new initial guesses for each number of hotspots for the
                    next possible iteration of fitting
    """
    w_inits = copy.deepcopy(w_inits_)

    # If the probability never gets large enough, return the degenerate parameters
    # The degenerate parameters are a bias term of -np.inf and all slopes set to 0
    # These parameters cause probs_pred to be an array of all 0s
    if len(probs) == 0:
        deg_opt = np.zeros_like(w_inits[-1])
        deg_opt[:, 0] = np.ones(len(deg_opt)) * -np.inf

        return (deg_opt, 0, -1), w_inits
    
    # If a large enough probability was detected, begin fitting

    # Convert the data to binary classification data
    X_bin, y_bin = convertToBinaryClassifier(probs, T, X_expt)

    # Compute the negative log likelihood of the null model which only
    # includes an intercept
    ybar = np.mean(y_bin)
    beta_null = np.log(ybar / (1 - ybar))
    null_weights = np.concatenate((np.array([beta_null]), 
                                   np.zeros(X_expt.shape[-1])))
    nll_null = negLL_hotspot(null_weights, X_bin, y_bin, False, 'none', 
                             0)
    if verbose:
        print(nll_null)

    # Now begin the McFadden pseudo-R2 early stopping loop
    if reg_method == 'MAP':
        last_opt = get_w(w_inits[0], X_bin, y_bin, nll_null, zero_prob=zero_prob, 
                        method=method, jac=jac, reg_method=reg_method, 
                        reg=(reg[0], reg[1][0][0], reg[1][0][1]),
                        verbose=opt_verbose, slope_bound=slope_bound)
    else:
        last_opt = get_w(w_inits[0], X_bin, y_bin, nll_null, zero_prob=zero_prob, 
                        method=method, jac=jac, reg_method=reg_method, reg=reg, verbose=opt_verbose,
                        slope_bound=slope_bound)
    w_inits[0] = last_opt[0]
    last_R2 = last_opt[2]   # store the pseudo-R2 value for early stopping
                            # procedure
    BIC = len(w_inits[0].flatten()) * np.log(len(X_bin)) + 2 * last_opt[1]
    HQC = 2 * len(w_inits[0].flatten()) * np.log(np.log(len(X_bin))) + 2 * last_opt[1]
    if verbose:
        print(last_opt, last_R2, BIC, HQC)

    for i in range(1, len(w_inits)):
        # Refit with next number of sites
        if reg_method == 'MAP':
            new_opt = get_w(w_inits[i], X_bin, y_bin, nll_null, zero_prob=zero_prob,
                            method=method,
                            jac=jac,
                            reg_method=reg_method,
                            reg=(reg[0], reg[1][i][0], reg[1][i][1]),
                            verbose=opt_verbose,
                            slope_bound=slope_bound)
        else:
            new_opt = get_w(w_inits[i], X_bin, y_bin, nll_null, zero_prob=zero_prob,
                            method=method,
                            jac=jac,
                            reg_method=reg_method,
                            reg=reg,
                            verbose=opt_verbose,
                            slope_bound=slope_bound)
        w_inits[i] = new_opt[0]
        new_R2 = new_opt[2]
        BIC = len(w_inits[i].flatten()) * np.log(len(X_bin)) + 2 * new_opt[1]
        HQC = 2 * len(w_inits[i].flatten()) * np.log(np.log(len(X_bin))) + 2 * new_opt[1]

        if verbose:
            print(new_opt, new_R2, BIC, HQC)

        # If the pseudo-R2 improvement was too small, break and stop adding sites
        if last_R2 > 0 and (new_R2 - last_R2) / last_R2 <= R2_thresh:
            break

        last_opt = new_opt
        last_R2 = new_R2

    return last_opt, w_inits

def fit_surface_CV(X_expt, probs, T, w_inits_, reg_method='none', reg=0,
                        R2_thresh=0.05, zero_prob=0.01, verbose=False,
                        method='L-BFGS-B', jac=negLL_hotspot_jac,
                        opt_verbose=False, random_state=None, slope_bound=20):
    """
    Fitting function for fitting surfaces to nonlinear data with multi-hotspot model.
    This function is primarily a wrapper for calling get_w() but now uses cross validation
    to determine the optimal number of hotspots. This function is significantly
    slower than fit_surface(), but it is more robust to overfitting.

    Parameters:
    X_expt (N x d np.ndarray): Input amplitudes
    probs (N x 1 np.ndarray): Probabilities corresponding to the amplitudes
    T (N x 1 np.ndarray): Trials at each amplitude
    w_inits_ (list): List of initial guessses for each number of hotspots. Each element
                    in the list is a (m x (d + 1)) np.ndarray with m the number of 
                    hotspots. This list should be generated externally.
    R2_thresh (float): Threshold used for determining when to stop adding hotspots
    zero_prob (float): Value for what the probability should be forced to be below
                       at an amplitude of 0-vector
    verbose (bool): Increases verbosity
    method (string): Method for optimization according to constrained optimization
                     methods available in scipy.optimize.minimize
    jac (function): Jacobian function if manually calculated
    reg_method (string): Regularization method. 'l2' is supported
    reg (float): Regularization parameter value
    opt_verbose (bool): Increases verbosity of the optimization function
    random_state (int): Random seed for cross validation
    slope_bound (float): Bound on the absolute value of the slopes

    Returns:
    opt (tuple): The optimized set of parameters for the
                    optimized number of hotspots m using
                    cross validation. The tuple consists of
                    (w_opt, nll_opt, R2_opt) where w_opt is the
                    optimized weight vector, nll_opt is the
                    negative log likelihood of the optimized
                    weight vector, and R2_opt is the McFadden
                    pseudo-R2 value of the optimized weight vector.
    w_inits (list): The new initial guesses for each number of hotspots for the
                    next possible iteration of fitting
    """
    w_inits = copy.deepcopy(w_inits_)

    # If the probability never gets large enough, return the degenerate parameters
    # The degenerate parameters are a bias term of -np.inf and all slopes set to 0
    # These parameters cause probs_pred to be an array of all 0s
    if len(probs) == 0:
        deg_opt = np.zeros_like(w_inits[-1])
        deg_opt[:, 0] = np.ones(len(deg_opt)) * -np.inf

        return (deg_opt, 0, -1), w_inits
    
    # If a large enough probability was detected, begin fitting

    # Convert the data to binary classification data
    X_bin, y_bin = convertToBinaryClassifier(probs, T, X_expt)
    
    # Stratified K Fold Cross Validation to choose number of sites
    skf = model_selection.StratifiedKFold(n_splits=len(w_inits), shuffle=True, random_state=random_state)
    test_R2s = np.zeros(len(w_inits))
    for i, (train_index, test_index) in enumerate(skf.split(X_bin, y_bin)):
        X_train, X_test = X_bin[train_index], X_bin[test_index]
        y_train, y_test = y_bin[train_index], y_bin[test_index]

        # Compute the negative log likelihood of the null model which only
        # includes an intercept
        ybar_train = np.mean(y_train)
        beta_null_train = np.log(ybar_train / (1 - ybar_train))
        null_weights_train = np.concatenate((np.array([beta_null_train]), 
                                             np.zeros(X_expt.shape[-1])))
        nll_null_train = negLL_hotspot(null_weights_train, X_train, y_train, False, 'none', 0)

        if reg_method == 'MAP':
            train_params, _, _ = get_w(w_inits[i], X_train, y_train, nll_null_train, zero_prob=zero_prob,
                                        method=method, jac=jac, reg_method=reg_method,
                                        reg=(reg[0], reg[1][i][0], reg[1][i][1]),
                                        verbose=opt_verbose, slope_bound=slope_bound)
        else:
            train_params, _, _ = get_w(w_inits[i], X_train, y_train, nll_null_train, 
                                                        zero_prob=zero_prob, method=method, jac=jac, 
                                                        reg_method=reg_method, reg=reg, 
                                                        verbose=opt_verbose,
                                                        slope_bound=slope_bound)
        test_fun = negLL_hotspot(train_params, X_test, y_test, opt_verbose, reg_method, reg)

        # Compute the negative log likelihood of the null model which only
        # includes an intercept
        ybar_test = np.mean(y_test)
        beta_null_test = np.log(ybar_test / (1 - ybar_test))
        null_weights_test = np.concatenate((np.array([beta_null_test]), 
                                             np.zeros(X_expt.shape[-1])))
        nll_null_test = negLL_hotspot(null_weights_test, X_test, y_test, False, 'none', 0)

        test_R2 = 1 - test_fun / nll_null_test
        test_R2s[i] = test_R2

    breakFlag = 0
    for i in range(1, len(test_R2s)):
        if test_R2s[i-1] > 0 and (test_R2s[i] - test_R2s[i-1]) / test_R2s[i-1] <= R2_thresh:
            breakFlag = 1
            break
    
    if breakFlag:
        ind = i - 1
        w_init = w_inits[ind]
    else:
        ind = len(w_inits) - 1
        w_init = w_inits[ind]

    # Compute the negative log likelihood of the null model which only
    # includes an intercept
    ybar = np.mean(y_bin)
    beta_null = np.log(ybar / (1 - ybar))
    null_weights = np.concatenate((np.array([beta_null]), 
                                np.zeros(X_expt.shape[-1])))
    nll_null = negLL_hotspot(null_weights, X_bin, y_bin, False, 'none', 0)

    if verbose:
        print(nll_null)

    if reg_method == 'MAP':
        opt = get_w(w_init, X_bin, y_bin, nll_null, zero_prob=zero_prob,
                        method=method,
                        jac=jac,
                        reg_method=reg_method,
                        reg=(reg[0], reg[1][i][0], reg[1][i][1]),
                        verbose=opt_verbose,
                        slope_bound=slope_bound)
    else:
        opt = get_w(w_init, X_bin, y_bin, nll_null, zero_prob=zero_prob,
                        method=method,
                        jac=jac,
                        reg_method=reg_method,
                        reg=reg,
                        verbose=opt_verbose,
                        slope_bound=slope_bound)

    w_inits[ind] = opt[0]
    return opt, w_inits

def get_w(w_init, X, y, nll_null, zero_prob=0.01, method='L-BFGS-B', jac=None,
          reg_method='none', reg=0, slope_bound=20, bias_bound=None, verbose=False,
        #   options={'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxfun': 15000}):
          options={'maxiter': 20000, 'ftol': 1e-10, 'maxfun': 20000}):
    """
    Fitting function for fitting data with a specified number of hotspots
    
    Parameters:
    w_init (m x (d + 1) np.ndarray): Initial guesses on parameters for model
                                     with m hotspots
    X (N x (d + 1) np.ndarray): Binary classification input data with constant term
    y (N x 1 np.ndarray): Binary classification output data (0s or 1s)
    nll_null (float): The negative log likelihood for the null model to the data 
    zero_prob (float): The forced maximum probability at 0-vector
    method (string): Optimization method according to constrained optimization
                     methods available in scipy.optimize.minimize
    jac (function): Manual jacobian function
    reg_method (string): Regularization method, only 'none' is currently supported
    reg (float): Regularization parameter
    slope_bound (float): Bound on the absolute value of the slopes
    bias_bound (float): Bound on the bias term
    verbose (bool): Increases verbosity
    options (dict): Dictionary of options for scipy.optimize.minimize

    Returns:
    weights (m x (d + 1) np.ndarrray): Fitted weight vector
    opt.fun (float): Minimized value of negative log likelihood
    R2 (float): McFadden pseudo-R2 value
    """

    z = 1 - (1 - zero_prob)**(1/len(w_init))

    # Set up bounds for constrained optimization
    bounds = []
    for j in range(len(w_init)):
        bounds += [(bias_bound, np.log(z/(1-z)))]
        for i in range(X.shape[-1] - 1):
            bounds += [(-slope_bound, slope_bound)]

    # Optimize the weight vector with MLE
    opt = minimize(negLL_hotspot, x0=w_init.ravel(), bounds=bounds,
                       args=(X, y, verbose, reg_method, reg), method=method,
                        jac=jac, options=options)
    
    return opt.x.reshape(-1, X.shape[-1]), opt.fun, (1 - opt.fun / nll_null)
