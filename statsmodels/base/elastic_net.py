import numpy as np
from statsmodels.base.model import Results
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly

"""
Elastic net regularization.

Routines for fitting regression models using elastic net
regularization.  The elastic net minimizes the objective function

-llf / nobs + alpha((1 - L1_wt) * sum(params**2) / 2 + L1_wt * sum(abs(params)))

The algorithm implemented here closely follows the implementation in
the R glmnet package, documented here:

http://cran.r-project.org/web/packages/glmnet/index.html

and here:

http://www.jstatsoft.org/v33/i01/paper

This routine should work for any regression model that implements
loglike, score, and hess.
"""


def _gen_npfuncs(k, L1_wt, alpha, loglike_kwds, score_kwds, hess_kwds):
    """
    Negative penalized log-likelihood functions.

    Returns the negative penalized log-likelihood, its derivative, and
    its Hessian.  The penalty only includes the smooth (L2) term.

    All three functions have argument signature (x, model), where
    ``x`` is a point in the parameter space and ``model`` is an
    arbitrary statsmodels regression model.
    """

    def nploglike(params, model):
        nobs = model.nobs
        pen_llf = alpha[k] * (1 - L1_wt) * np.sum(params**2) / 2
        llf = model.loglike(np.r_[params], **loglike_kwds)
        return - llf / nobs + pen_llf

    def npscore(params, model):
        nobs = model.nobs
        pen_grad = alpha[k] * (1 - L1_wt) * params
        gr = -model.score(np.r_[params], **score_kwds)[0] / nobs
        return gr + pen_grad

    def nphess(params, model):
        nobs = model.nobs
        pen_hess = alpha[k] * (1 - L1_wt)
        h = -model.hessian(np.r_[params], **hess_kwds)[0,0] / nobs + pen_hess
        return h

    return nploglike, npscore, nphess



def fit_elasticnet(model, method="coord_descent", maxiter=100,
         alpha=0., L1_wt=1., start_params=None, cnvrg_tol=1e-7,
         zero_tol=1e-8, refit=False, check_step=True,
         loglike_kwds=None, score_kwds=None, hess_kwds=None):
    """
    Return an elastic net regularized fit to a regression model.

    Parameters
    ----------
    model : model object
        A statsmodels object implementing ``loglike``, ``score``, and
        ``hessian``.
    method :
        Only the coordinate descent algorithm is implemented.
    maxiter : integer
        The maximum number of iteration cycles (an iteration cycle
        involves running coordinate descent on all variables).
    alpha : scalar or array-like
        The penalty weight.  If a scalar, the same penalty weight
        applies to all variables in the model.  If a vector, it
        must have the same length as `params`, and contains a
        penalty weight for each coefficient.
    L1_wt : scalar
        The fraction of the penalty given to the L1 penalty term.
        Must be between 0 and 1 (inclusive).  If 0, the fit is
        a ridge fit, if 1 it is a lasso fit.
    start_params : array-like
        Starting values for `params`.
    cnvrg_tol : scalar
        If `params` changes by less than this amount (in sup-norm)
        in one iteration cycle, the algorithm terminates with
        convergence.
    zero_tol : scalar
        Any estimated coefficient smaller than this value is
        replaced with zero.
    refit : bool
        If True, the model is refit using only the variables that have
        non-zero coefficients in the regularized fit.  The refitted
        model is not regularized.
    check_step : bool
        If True, confirm that the first step is an improvement and search
        further if it is not.
    loglike_kwds : dict-like or None
        Keyword arguments for the log-likelihood function.
    score_kwds : dict-like or None
        Keyword arguments for the score function.
    hess_kwds : dict-like or None
        Keyword arguments for the Hessian function.

    Returns
    -------
    A results object.

    Notes
    -----
    The ``elastic net`` penalty is a combination of L1 and L2
    penalties.

    The function that is minimized is:

    -loglike/n + alpha*((1-L1_wt)*|params|_2^2/2 + L1_wt*|params|_1)

    where |*|_1 and |*|_2 are the L1 and L2 norms.

    The computational approach used here is to obtain a quadratic
    approximation to the smooth part of the target function:

    -loglike/n + alpha*(1-L1_wt)*|params|_2^2/2

    then repeatedly optimize the L1 penalized version of this function
    along coordinate axes.
    """

    k_exog = model.exog.shape[1]
    n_exog = model.exog.shape[0]

    loglike_kwds = {} if loglike_kwds is None else loglike_kwds
    score_kwds = {} if score_kwds is None else score_kwds
    hess_kwds = {} if hess_kwds is None else hess_kwds

    if np.isscalar(alpha):
        alpha = alpha * np.ones(k_exog)

    # Define starting params
    if start_params is None:
        params = np.zeros(k_exog)
    else:
        params = start_params.copy()

    converged = False
    btol = 1e-4
    params_zero = np.zeros(len(params), dtype=bool)

    init_args = dict([(k, getattr(model, k)) for k in model._init_keys
                      if k != "offset" and hasattr(model, k)])
    init_args['hasconst'] = False

    fgh_list = [_gen_npfuncs(k, L1_wt, alpha, loglike_kwds, score_kwds, hess_kwds)
                for k in range(k_exog)]

    for itr in range(maxiter):

        # Sweep through the parameters
        params_save = params.copy()
        for k in range(k_exog):

            # Under the active set method, if a parameter becomes
            # zero we don't try to change it again.
            # TODO : give the user the option to switch this off
            if params_zero[k]:
                continue

            # Set the offset to account for the variables that are
            # being held fixed in the current coordinate
            # optimization.
            params0 = params.copy()
            params0[k] = 0
            offset = np.dot(model.exog, params0)
            if hasattr(model, "offset") and model.offset is not None:
                offset += model.offset

            # Create a one-variable model for optimization.
            model_1var = model.__class__(model.endog, model.exog[:, k], offset=offset,
                                         **init_args)

            # Do the one-dimensional optimization.
            func, grad, hess = fgh_list[k]
            params[k] = _opt_1d(func, grad, hess, model_1var, params[k], alpha[k]*L1_wt,
                                tol=btol, check_step=check_step)

            # Update the active set
            if itr > 0 and np.abs(params[k]) < zero_tol:
                params_zero[k] = True
                params[k] = 0.

        # Check for convergence
        pchange = np.max(np.abs(params - params_save))
        if pchange < cnvrg_tol:
            converged = True
            break

    # Set approximate zero coefficients to be exactly zero
    params[np.abs(params) < zero_tol] = 0

    if not refit:
        results = RegularizedResults(model, params)
        return RegularizedResultsWrapper(results)

    # Fit the reduced model to get standard errors and other
    # post-estimation results.
    ii = np.flatnonzero(params)
    cov = np.zeros((k_exog, k_exog))
    init_args = dict([(k, getattr(model, k, None)) for k in model._init_keys])
    if len(ii) > 0:
        model1 = model.__class__(model.endog, model.exog[:, ii],
                               **init_args)
        rslt = model1.fit()
        cov[np.ix_(ii, ii)] = rslt.normalized_cov_params
    else:
        # Hack: no variables were selected but we need to run fit in
        # order to get the correct results class.  So just fit a model
        # with one variable.
        model1 = model.__class__(model.endog, model.exog[:, 0], **init_args)
        rslt = model1.fit(maxiter=0)

    # fit may return a results or a results wrapper
    if issubclass(rslt.__class__, wrap.ResultsWrapper):
        klass = rslt._results.__class__
    else:
        klass = rslt.__class__

    # Not all models have a scale
    if hasattr(rslt, 'scale'):
        scale = rslt.scale
    else:
        scale = 1.

    # Assuming a standard signature for creating results classes.
    refit = klass(model, params, cov, scale=scale)
    refit.regularized = True
    refit.method = method
    refit.fit_history = {'iteration' : itr + 1}

    return refit


def fit_elasticnet_path(model, alpha_path=np.arange(0., 1., 100),
                        error_variance=None, **kwargs):
    """
    Return a path of elastic net regularized fits as well as corresponding
    test statistics

    Parameters
    ----------
    model : model object
        A statsmodels object implementing ``loglike``, ``score``, and
        ``hessian``.
    alpha_path : array-like
        Array of penalty weights.  Each element can itself be an array
        of alpha values or a scalar.
    error_variance : None or scalar
        The error variance used when estimating the test statistic,
        if None this corresponds to

        ||y - X beta^{OLS}||_2^2 / (n - p)

    Returns
    -------
    A results object.

    Notes
    -----
    The test statistic here corresponds to the "covariance statistic"
    proposed by Lockhart et al.

    https://arxiv.org/abs/1301.7161

    The implementation here works for GLMs and Cox regression.
    """
    # TODO
    # ----
    #
    # * Add support for unknown error_variance where p > n.
    #
    # * Add support for cases where zeroed alpha values are
    # included (e.g. intercept)
    #
    # * Confirm support for GLMs/Cox
    #
    # * Add support for more than OLS

    defaults = {}
    defaults.update(kwargs)

    # select distribution for covariance statistic
    n, p = model.exog.shape
    if error_variance is None:
        from scipy.stats import f
        dist = lambda x: f.pdf(x, 2, n - p)
    else:
        from scipy.stats import expon
        dist = expon.pdf

    # generate error_variance if None
    if error_variance is None:
        # TODO ensure that this is fully general
        error_fit = model.fit()
        error_variance = error_fit.mse_model

    # iterate over alpha values and build list of params
    param_path = []
    fit_l = []
    for alpha in alpha_path:
        defaults["alpha"] = alpha
        fit = fit_elasticnet(model, **defaults)
        param_path.append(fit.params)
        # NOTE we keep a list of fits calling predict to generate
        # the test stat later
        fit_l.append(fit)
    param_path = np.vstack((np.zeros(p), np.array(param_path)))
    param_path_ind = np.sign(param_path)
    break_points = param_path_ind != np.roll(param_path_ind, 1, axis=0)
    break_points = np.sum(break_points, axis=1)
    break_points = (break_points > 0)[1:]
    knot_count = np.sum(break_points)
    fit_l = [f for f, b in zip(fit_l, break_points) if b]
    # this takes the points where the sign of the params changes (knots)
    param_knots = param_path[1:,:][break_points]
    rolled_param_knots = param_path[np.roll(break_points, 1)][1:]

    # we extract these to build a temporary model for the active fit
    init_args = dict([(k, getattr(model, k, None)) for k in model._init_keys])

    # generate params for active parameter sets
    active_param_knots = []
    active_fit_l = []
    for alpha, params in zip(alpha_path[break_points], rolled_param_knots):
        active_set = params != 0
        active_model = model.__class__(model.endog, model.exog[:,active_set],
                                       **init_args)
        defaults["alpha"] = alpha
        active_fit = fit_elasticnet(active_model, **defaults)
        active_fit_l.append(active_fit)
        active_param_knots.append(active_fit.params)
    active_param_knots = np.array(active_param_knots)

    # generate covariance statistic for each knot
    cov_stat_knots = []
    for knot in range(knot_count):
        cov_stat = np.inner(model.endog, fit_l[knot].predict())
        cov_stat -= np.inner(model.endog, active_fit_l[knot].predict())
        active_set = rolled_param_knots[knot,:] != 0
        diff = (np.sign(param_knots[knot,:][active_set]) !=
                np.sign(active_param_knots[knot,:]))
        cov_stat /= np.max([1, np.sum(diff)])
        cov_stat /= error_variance

        cov_stat_knots.append(cov_stat)
    cov_stat_knots = np.array(cov_stat_l)

#    # generate pval
#    pval = dist(cov_stat_knots)

    return RegularizedPathResults(model, param_knots, cov_stat_knots, dist)

#    # now map the covariance statistics to parameters
#    # TODO map this to the standard statsmodels ContrastResults
#    min_active_ind = np.argmax(param_knots != 0, axis=0)
#    pval = dist(cov_stat_knots)
#    pval = dist(cov_stat_knots[min_active_ind])
#    pval[np.sum(param_path != 0, axis=0) == 0] = np.NAN
#    pval[min_active_ind == 0] = np.NAN

    # TODO add proper results class
#    res = {"cov_stat_knots": cov_stat_knots, "pval": pval,
#           "alpha_knots": alpha_knots, "fit_knots": fit_knots,
#           "active_fit_knots": active_fit_knots, "param_knots": param_knots,
#           "dist": dist, "break_knots": break_knots}
#    return res


class RegularizedPathResults(Results):

    def __init__(self, model, params):
        super(RegularizedPathResults, self).__init__(model, params)

    @cache_readonly
    def fittedvalues(self, alpha=None):

        if alpha is None:




def _opt_1d(func, grad, hess, model, start, L1_wt, tol,
            check_step=True):
    """
    One-dimensional helper for elastic net.

    Parameters:
    -----------
    func : function
        A smooth function of a single variable to be optimized
        with L1 penaty.
    grad : function
        The gradient of `func`.
    hess : function
        The Hessian of `func`.
    model : statsmodels model
        The model being fit.
    start : real
        A starting value for the function argument
    L1_wt : non-negative real
        The weight for the L1 penalty function.
    tol : non-negative real
        A convergence threshold.
    check_step : bool
        If True, check that the first step is an improvement and
        use bisection if it is not.  If False, return after the
        first step regardless.

    Notes
    -----
    ``func``, ``grad``, and ``hess`` have argument signature (x,
    model), where ``x`` is a point in the parameter space and
    ``model`` is the model being fit.

    If the log-likelihood for the model is exactly quadratic, the
    global minimum is returned in one step.  Otherwise numerical
    bisection is used.

    Returns
    -------
    The argmin of the objective function.
    """

    # Overview:
    # We want to minimize L(x) + L1_wt*abs(x), where L() is a smooth
    # loss function that includes the log-likelihood and L2 penalty.
    # This is a 1-dimensional optimization.  If L(x) is exactly
    # quadratic we can solve for the argmin exactly.  Otherwise we
    # approximate L(x) with a quadratic function Q(x) and try to use
    # the minimizer of Q(x) + L1_wt*abs(x).  But if this yields an
    # uphill step for the actual target function L(x) + L1_wt*abs(x),
    # then we fall back to a expensive line search.  The line search
    # is never needed for OLS.

    x = start
    f = func(x, model)
    b = grad(x, model)
    c = hess(x, model)
    d = b - c*x

    # The optimum is achieved by hard thresholding to zero
    if L1_wt > np.abs(d):
        return 0.

    # x + h is the minimizer of the Q(x) + L1_wt*abs(x)
    if d >= 0:
        h = (L1_wt - b) / c
    elif d < 0:
        h = -(L1_wt + b) / c
    else:
        return np.nan

    # If the new point is not uphill for the target function, take it
    # and return.  This check is a bit expensive and un-necessary for
    # OLS
    if not check_step:
        return x + h
    f1 = func(x + h, model) + L1_wt*np.abs(x + h)
    if f1 <= f + L1_wt*np.abs(x) + 1e-10:
        return x + h

    # Fallback for models where the loss is not quadratic
    from scipy.optimize import brent
    x_opt = brent(func, args=(model,), brack=(x-1, x+1), tol=tol)
    return x_opt


class RegularizedResults(Results):

    def __init__(self, model, params):
        super(RegularizedResults, self).__init__(model, params)

    @cache_readonly
    def fittedvalues(self):
        return self.model.predict(self.params)


class RegularizedResultsWrapper(wrap.ResultsWrapper):
    _attrs = {
        'params': 'columns',
        'resid': 'rows',
        'fittedvalues': 'rows',
    }

    _wrap_attrs = _attrs

wrap.populate_wrapper(RegularizedResultsWrapper,
                      RegularizedResults)
