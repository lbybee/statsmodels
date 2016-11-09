import numpy as np
from statsmodels.base import model
from statsmodels.base.optimizer import Optimizer
from statsmodels.duration.hazard_regression import *


class MixedPHSurvivalTime(object):

    def __init__(self, time, status, exog, clusters, strata=None, entry=None,
                 offset=None):
        """
        Represent a collection of survival times with possible
        stratification and left truncation.

        Parameters
        ----------
        time : array_like
            The times at which either the event (failure) occurs or
            the observation is censored.
        status : array_like
            Indicates whether the event (failure) occurs at `time`
            (`status` is 1), or if `time` is a censoring time (`status`
            is 0).
        exog : array_like
            The exogeneous (covariate) data matrix, cases are rows and
            variables are columns.
        clusters : array-like
            array of cluster labels
        strata : array_like
            Grouping variable defining the strata.  If None, all
            observations are in a single stratum.
        entry : array_like
            Entry (left truncation) times.  The observation is not
            part of the risk set for times before the entry time.  If
            None, the entry time is treated as being zero, which
            gives no left truncation.  The entry time must be less
            than or equal to `time`.
        offset : array-like
            An optional array of offsets
        """

        # Default strata
        if strata is None:
            strata = np.zeros(len(time), dtype=np.int32)

        # Default entry times
        if entry is None:
            entry = np.zeros(len(time))

        # Parameter validity checks.
        n1, n2, n3, n4 = len(time), len(status), len(strata),\
            len(entry)
        nv = [n1, n2, n3, n4]
        if max(nv) != min(nv):
            raise ValueError("endog, status, strata, and " +
                             "entry must all have the same length")
        if min(time) < 0:
            raise ValueError("endog must be non-negative")
        if min(entry) < 0:
            raise ValueError("entry time must be non-negative")

        # In Stata, this is entry >= time, in R it is >.
        if np.any(entry > time):
            raise ValueError("entry times may not occur " +
                             "after event or censoring times")

        # Get the row indices for the cases in each stratum
        stu = np.unique(strata)
        #sth = {x: [] for x in stu} # needs >=2.7
        sth = dict([(x, []) for x in stu])
        for i,k in enumerate(strata):
            sth[k].append(i)
        stratum_rows = [np.asarray(sth[k], dtype=np.int32) for k in stu]
        stratum_names = stu

        # Remove strata with no events
        ix = [i for i,ix in enumerate(stratum_rows) if status[ix].sum() > 0]
        self.nstrat_orig = len(stratum_rows)
        stratum_rows = [stratum_rows[i] for i in ix]
        stratum_names = [stratum_names[i] for i in ix]

        # The number of strata
        nstrat = len(stratum_rows)
        self.nstrat = nstrat

        # Remove subjects whose entry time occurs after the last event
        # in their stratum.
        for stx,ix in enumerate(stratum_rows):
            last_failure = max(time[ix][status[ix] == 1])

            # Stata uses < here, R uses <=
            ii = [i for i,t in enumerate(entry[ix]) if
                  t <= last_failure]
            stratum_rows[stx] = stratum_rows[stx][ii]

        # Remove subjects who are censored before the first event in
        # their stratum.
        for stx,ix in enumerate(stratum_rows):
            first_failure = min(time[ix][status[ix] == 1])

            ii = [i for i,t in enumerate(time[ix]) if
                  t >= first_failure]
            stratum_rows[stx] = stratum_rows[stx][ii]

        # Order by time within each stratum
        for stx,ix in enumerate(stratum_rows):
            ii = np.argsort(time[ix])
            stratum_rows[stx] = stratum_rows[stx][ii]

        if offset is not None:
            self.offset_s = []
            for stx in range(nstrat):
                self.offset_s.append(offset[stratum_rows[stx]])
        else:
            self.offset_s = None

        # Number of informative subjects
        self.n_obs = sum([len(ix) for ix in stratum_rows])

        # Split everything by stratum
        self.time_s = []
        self.exog_s = []
        self.clusters_s = []
        self.status_s = []
        self.entry_s = []
        for ix in stratum_rows:
            self.time_s.append(time[ix])
            self.exog_s.append(exog[ix,:])
            self.clusters_s.append(clusters[ix])
            self.status_s.append(status[ix])
            self.entry_s.append(entry[ix])

        # TODO properly handle this
        self.clusters_s = np.asarray(self.clusters_s)

        self.stratum_rows = stratum_rows
        self.stratum_names = stratum_names

        # Precalculate some indices needed to fit Cox models.
        # Distinct failure times within a stratum are always taken to
        # be sorted in ascending order.
        #
        # ufailt_ix[stx][k] is a list of indices for subjects who fail
        # at the k^th sorted unique failure time in stratum stx
        #
        # risk_enter[stx][k] is a list of indices for subjects who
        # enter the risk set at the k^th sorted unique failure time in
        # stratum stx
        #
        # risk_exit[stx][k] is a list of indices for subjects who exit
        # the risk set at the k^th sorted unique failure time in
        # stratum stx
        self.ufailt_ix, self.risk_enter, self.risk_exit, self.ufailt =\
            [], [], [], []

        for stx in range(self.nstrat):

            # All failure times
            ift = np.flatnonzero(self.status_s[stx] == 1)
            ft = self.time_s[stx][ift]

            # Unique failure times
            uft = np.unique(ft)
            nuft = len(uft)

            # Indices of cases that fail at each unique failure time
            #uft_map = {x:i for i,x in enumerate(uft)} # requires >=2.7
            uft_map = dict([(x, i) for i,x in enumerate(uft)]) # 2.6
            uft_ix = [[] for k in range(nuft)]
            for ix,ti in zip(ift,ft):
                uft_ix[uft_map[ti]].append(ix)

            # Indices of cases (failed or censored) that enter the
            # risk set at each unique failure time.
            risk_enter1 = [[] for k in range(nuft)]
            for i,t in enumerate(self.time_s[stx]):
                ix = np.searchsorted(uft, t, "right") - 1
                if ix >= 0:
                    risk_enter1[ix].append(i)

            # Indices of cases (failed or censored) that exit the
            # risk set at each unique failure time.
            risk_exit1 = [[] for k in range(nuft)]
            for i,t in enumerate(self.entry_s[stx]):
                ix = np.searchsorted(uft, t)
                risk_exit1[ix].append(i)

            self.ufailt.append(uft)
            self.ufailt_ix.append([np.asarray(x, dtype=np.int32) for x in uft_ix])
            self.risk_enter.append([np.asarray(x, dtype=np.int32) for x in risk_enter1])
            self.risk_exit.append([np.asarray(x, dtype=np.int32) for x in risk_exit1])



class MixedPHReg(model.LikelihoodModel):
    """
    Fit the Cox proportional hazards regression model for right
    censored data for fast frailty.

    Parameters
    ----------
    endog : array-like
        The observed times (event or censoring)
    exog : 2D array-like
        The covariates or exogeneous variables
    clusters : array-like
        array of cluster labels
    status : array-like
        The censoring status values; status=1 indicates that an
        event occured (e.g. failure or death), status=0 indicates
        that the observation was right censored. If None, defaults
        to status=1 for all cases.
    entry : array-like
        The entry times, if left truncation occurs
    strata : array-like
        Stratum labels.  If None, all observations are taken to be
        in a single stratum.
    ties : string
        The method used to handle tied times, must be either 'breslow'
        or 'efron'.
    missing : string
        The method used to handle missing data

    Notes
    -----
    Proportional hazards regression models should not include an
    explicit or implicit intercept.  The effect of an intercept is
    not identified using the partial likelihood approach.

    `endog`, `clusters`, `event`, `strata`, `entry`, and the first dimension
    of `exog` all must have the same length
    """

    def __init__(self, endog, exog, clusters, status=None, entry=None,
                 strata=None, offset=None, ties='breslow',
                 missing='drop', **kwargs):

        # Default is no censoring
        if status is None:
            status = np.ones(len(endog))

        # TODO Fix this
#        super(MixedPHReg, self).__init__(endog, exog, clusters,
#                                         status=status, entry=entry,
#                                         strata=strata, offset=offset,
#                                         missing=missing, **kwargs)


        self.endog = endog
        self.exog = exog
        self.clusters = clusters
        self.status = status
        self.entry = entry
        self.strata = strata
        self.offset = offset
        self.missing = missing

        # endog and exog are automatically converted, but these are
        # not
        if self.status is not None:
            self.status = np.asarray(self.status)
        if self.entry is not None:
            self.entry = np.asarray(self.entry)
        if self.strata is not None:
            self.strata = np.asarray(self.strata)

        self.surv = MixedPHSurvivalTime(self.endog, self.status,
                                        self.exog, self.clusters,
                                        self.strata, self.entry,
                                        self.offset)

        self.nobs = len(self.endog)
        self.groups = None

        # TODO: not used?
        self.missing = missing

        self.df_resid = (np.float(self.exog.shape[0] -
                                  np_matrix_rank(self.exog)))
        self.df_model = np.float(np_matrix_rank(self.exog))

        ties = ties.lower()
        if ties not in ("efron", "breslow"):
            raise ValueError("`ties` must be either `efron` or " +
                             "`breslow`")

        self.ties = ties


    @classmethod
    def from_formula(cls, formula, data, clusters, status=None, entry=None,
                     strata=None, offset=None, subset=None,
                     ties='breslow', missing='drop', *args, **kwargs):
        """
        Create a proportional hazards regression model from a formula
        and dataframe.

        Parameters
        ----------
        formula : str or generic Formula object
            The formula specifying the model
        data : array-like
            The data for the model. See Notes.
        clusters : string
            variable in data corresponding to cluster index
        status : array-like
            The censoring status values; status=1 indicates that an
            event occured (e.g. failure or death), status=0 indicates
            that the observation was right censored. If None, defaults
            to status=1 for all cases.
        entry : array-like
            The entry times, if left truncation occurs
        strata : array-like
            Stratum labels.  If None, all observations are taken to be
            in a single stratum.
        offset : array-like
            Array of offset values
        subset : array-like
            An array-like object of booleans, integers, or index
            values that indicate the subset of df to use in the
            model. Assumes df is a `pandas.DataFrame`
        ties : string
            The method used to handle tied times, must be either 'breslow'
            or 'efron'.
        missing : string
            The method used to handle missing data
        args : extra arguments
            These are passed to the model
        kwargs : extra keyword arguments
            These are passed to the model with one exception. The
            ``eval_env`` keyword is passed to patsy. It can be either a
            :class:`patsy:patsy.EvalEnvironment` object or an integer
            indicating the depth of the namespace to use. For example, the
            default ``eval_env=0`` uses the calling namespace. If you wish
            to use a "clean" environment set ``eval_env=-1``.

        Returns
        -------
        model : MixedPHReg model instance
        """

        # Allow array arguments to be passed by column name.
        if isinstance(status, str):
            status = data[status]
        if isinstance(entry, str):
            entry = data[entry]
        if isinstance(strata, str):
            strata = data[strata]
        if isinstance(offset, str):
            offset = data[offset]
        if isinstance(clusters, str):
            clusters = data[clusters]

        mod = super(MixedPHReg, cls).from_formula(formula, data,
                    clusters, status=status, entry=entry, strata=strata,
                    offset=offset, subset=subset, ties=ties,
                    missing=missing, *args, **kwargs)

        return mod


    def est_beta_params(self, method="newton", maxiter=100, callback=None,
                        fargs=(), retall=None, full_output=True, disp=True, **kwargs):
        """estimates beta using the standard PHReg method with the
        frailty terms as an offset

        See LikelihoodModel.fit for params.
        TODO update this
        """

        nobs = self.endog.shape[0]
        f = lambda params: -self.loglike_beta(params) / nobs
        score = lambda params: -self.score_beta(params) / nobs
        try:
            hess = lambda params: -self.hessian_beta(params) / nobs
        except:
            hess = None

        if method == 'newton':
            score = lambda params: self.score_beta(params) / nobs
            print score
            hess = lambda params: self.hessian_beta(params) / nobs
            #TODO: why are score and hess positive?

        optimizer = Optimizer()
        xopt, retvals, optim_settings = optimizer._fit(f, score,
                                                       self.beta_params,
                                                       fargs, kwargs,
                                                       hessian=hess,
                                                       method=method,
                                                       disp=disp,
                                                       maxiter=maxiter,
                                                       callback=callback,
                                                       retall=retall,
                                                       full_output=full_output)

        self.beta_params = xopt


    def est_b_params(self, method="newton", maxiter=100, callback=None,
                        fargs=(), retall=None, full_output=True, disp=True, **kwargs):
        """estimates b (frailty terms) using the standard PHReg method

        See LikelihoodModel.fit for params.
        TODO update this
        """

        nobs = self.endog.shape[0]
        f = lambda params: -self.loglike_frailty(params) / nobs
        score = lambda params: -self.score_frailty(params) / nobs
        try:
            hess = lambda params: -self.hessian_frailty(params) / nobs
        except:
            hess = None

        if method == 'newton':
            score = lambda params: self.score_frailty(params) / nobs
            hess = lambda params: self.hessian_frailty(params) / nobs
            #TODO: why are score and hess positive?

        optimizer = Optimizer()
        xopt, retvals, optim_settings = optimizer._fit(f, score,
                                                       self.b_params,
                                                       fargs, kwargs,
                                                       hessian=hess,
                                                       method=method,
                                                       disp=disp,
                                                       maxiter=maxiter,
                                                       callback=callback,
                                                       retall=retall,
                                                       full_output=full_output)

        self.b_params = xopt


    def est_theta(self):
        """
        uses breslow
        TBA
        """

        surv = self.surv

        nclust = self.b_params.shape[0]
        D_inv = np.eye(nclust) * 1 / self.theta
        # Loop over strata
        for stx in range(surv.nstrat):

            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)

            exog_s = surv.exog_s[stx]

            # clusters in stratum
            clust_ix = surv.clusters_s[stx]

            linpred = np.dot(exog_s, self.beta_params)
            linpred += self.b_params[surv.clusters_s[stx]]
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)

            xp0, xp1, xp2 = 0., 0., 0.

            # Iterate backward through the unique failure times.
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                if len(ix) > 0:
                    xp0 += e_linpred[ix].sum()
                    v = np.zeros((len(ix), len(self.b_params)))
                    # TODO fix handling of range(len(clust_ix[ix]))
                    v[range(len(clust_ix[ix])),clust_ix[ix]] = 1
                    # TODO this may not be sufficient for the 2nd update
                    # need to double check the math to make sure we don't
                    # need an additional term besides xp1
                    xp1 += (e_linpred[ix][:,None] * v).sum(0)
                    elx = e_linpred[ix]

                    m = len(uft_ix[i])

                    # TODO should be renamed
                    gamma_i = np.sqrt(1 / xp0**2 * m)
                    alpha_i = 1 / xp0 * m

                    xp1 *= gamma_i

                    # TODO there is likely a better way to handle this
                    ix_ln = len(ix)
                    for k in range(ix_ln):

                        v_k = v[k,:]
                        alpha_ik = np.sqrt(alpha_i * elx[k])
                        v_k *= alpha_ik

                        update0 = D_inv.dot(np.outer(v_k, v_k)).dot(D_inv)
                        update0 /= (1 + v_k.T.dot(D_inv).dot(v_k))

                        D_inv -= update0

                    update1 = D_inv.dot(np.outer(xp1, xp1)).dot(D_inv)
                    update1 /= (1 - xp1.T.dot(D_inv).dot(xp1))

                    D_inv += update1

        theta = (self.b_params.T.dot(self.b_params) + np.trace(D_inv))
        theta /= nclust
        self.theta = theta


    def fit(self, convg_tol=1e-5, beta_params_start=None, b_params_start=None,
            theta_start=None, method="newton", maxiter=100):
        """performs the fast frailty estimation procedure

        Parameters
        ----------

        Returns
        -------
        """

        fit_kwds={"method": method, "maxiter": maxiter, "fargs": (),
                  "callback": None, "retall": None, "full_output": True,
                  "disp": True}

        ll0 = np.inf
        ll1 = 0

        if beta_params_start is None:
            self.beta_params = np.array([0] * self.exog.shape[1])
        else:
            self.beta_params = beta_params_start

        if b_params_start is None:
            self.b_params = np.array([0] * np.unique(self.clusters).shape[0])
        else:
            self.b_params = b_params_start

        if theta_start is None:
            self.theta = 1
        else:
            self.theta = theta_start

        while not np.isclose(ll0, ll1, rtol=convg_tol):

            ll0 = ll1
            self.est_beta_params(**fit_kwds)
            self.est_b_params(**fit_kwds)
            self.est_theta()
            ll1 = self.loglike_frailty(self.b_params)

        return self.beta_params, self.b_params, self.theta


    # loglike, score and hessian for standard params
    def loglike_beta(self, params):
        """
        Returns the log partial likelihood function evaluated at
        `params`.
        """

        if self.ties == "breslow":
            return self.breslow_loglike_beta(params)
        elif self.ties == "efron":
            return self.efron_loglike_beta(params)

    def score_beta(self, params):
        """
        Returns the score function evaluated at `params`.
        """

        if self.ties == "breslow":
            return self.breslow_gradient_beta(params)
        elif self.ties == "efron":
            return self.efron_gradient_beta(params)

    def hessian_beta(self, params):
        """
        Returns the Hessian matrix of the log partial likelihood
        function evaluated at `params`.
        """

        if self.ties == "breslow":
            return self.breslow_hessian_beta(params)
        elif self.ties == "efron":
            return self.efron_hessian_beta(params)

    def breslow_loglike_beta(self, params):
        """
        Returns the value of the log partial likelihood function
        evaluated at `params`, using the Breslow method to handle tied
        times.
        """

        surv = self.surv

        like = 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            uft_ix = surv.ufailt_ix[stx]
            exog_s = surv.exog_s[stx]
            nuft = len(uft_ix)

            linpred = np.dot(exog_s, params)
            linpred += self.b_params[surv.clusters_s[stx]]
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)

            xp0 = 0.

            # Iterate backward through the unique failure times.
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                xp0 += e_linpred[ix].sum()

                # Account for all cases that fail at this point.
                ix = uft_ix[i]
                like += (linpred[ix] - np.log(xp0)).sum()

                # Update for cases leaving the risk set.
                ix = surv.risk_exit[stx][i]
                xp0 -= e_linpred[ix].sum()

        return like

    def efron_loglike_beta(self, params):
        """
        Returns the value of the log partial likelihood function
        evaluated at `params`, using the Efron method to handle tied
        times.
        """

        surv = self.surv

        like = 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            # exog and linear predictor for this stratum
            exog_s = surv.exog_s[stx]
            linpred = np.dot(exog_s, params)
            linpred += self.b_params[surv.clusters_s[stx]]
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)

            xp0 = 0.

            # Iterate backward through the unique failure times.
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                xp0 += e_linpred[ix].sum()
                xp0f = e_linpred[uft_ix[i]].sum()

                # Account for all cases that fail at this point.
                ix = uft_ix[i]
                like += linpred[ix].sum()

                m = len(ix)
                J = np.arange(m, dtype=np.float64) / m
                like -= np.log(xp0 - J*xp0f).sum()

                # Update for cases leaving the risk set.
                ix = surv.risk_exit[stx][i]
                xp0 -= e_linpred[ix].sum()

        return like

    def breslow_gradient_beta(self, params):
        """
        Returns the gradient of the log partial likelihood, using the
        Breslow method to handle tied times.
        """

        surv = self.surv

        grad = 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            # Indices of subjects in the stratum
            strat_ix = surv.stratum_rows[stx]

            # Unique failure times in the stratum
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)

            # exog and linear predictor for the stratum
            exog_s = surv.exog_s[stx]
            linpred = np.dot(exog_s, params)
            linpred += self.b_params[surv.clusters_s[stx]]
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)

            xp0, xp1 = 0., 0.

            # Iterate backward through the unique failure times.
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                if len(ix) > 0:
                    v = exog_s[ix,:]
                    xp0 += e_linpred[ix].sum()
                    xp1 += (e_linpred[ix][:,None] * v).sum(0)

                # Account for all cases that fail at this point.
                ix = uft_ix[i]
                grad += (exog_s[ix,:] - xp1 / xp0).sum(0)

                # Update for cases leaving the risk set.
                ix = surv.risk_exit[stx][i]
                if len(ix) > 0:
                    v = exog_s[ix,:]
                    xp0 -= e_linpred[ix].sum()
                    xp1 -= (e_linpred[ix][:,None] * v).sum(0)

        return grad

    def efron_gradient_beta(self, params):
        """
        Returns the gradient of the log partial likelihood evaluated
        at `params`, using the Efron method to handle tied times.
        """

        surv = self.surv

        grad = 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            # Indices of cases in the stratum
            strat_ix = surv.stratum_rows[stx]

            # exog and linear predictor of the stratum
            exog_s = surv.exog_s[stx]
            linpred = np.dot(exog_s, params)
            linpred += self.b_params[surv.clusters_s[stx]]
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)

            xp0, xp1 = 0., 0.

            # Iterate backward through the unique failure times.
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                if len(ix) > 0:
                    v = exog_s[ix,:]
                    xp0 += e_linpred[ix].sum()
                    xp1 += (e_linpred[ix][:,None] * v).sum(0)
                ixf = uft_ix[i]
                if len(ixf) > 0:
                    v = exog_s[ixf,:]
                    xp0f = e_linpred[ixf].sum()
                    xp1f = (e_linpred[ixf][:,None] * v).sum(0)

                    # Consider all cases that fail at this point.
                    grad += v.sum(0)

                    m = len(ixf)
                    J = np.arange(m, dtype=np.float64) / m
                    numer = xp1 - np.outer(J, xp1f)
                    denom = xp0 - np.outer(J, xp0f)
                    ratio = numer / denom
                    rsum = ratio.sum(0)
                    grad -= rsum

                # Update for cases leaving the risk set.
                ix = surv.risk_exit[stx][i]
                if len(ix) > 0:
                    v = exog_s[ix,:]
                    xp0 -= e_linpred[ix].sum()
                    xp1 -= (e_linpred[ix][:,None] * v).sum(0)

        return grad

    def breslow_hessian_beta(self, params):
        """
        Returns the Hessian of the log partial likelihood evaluated at
        `params`, using the Breslow method to handle tied times.
        """

        surv = self.surv

        hess = 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)

            exog_s = surv.exog_s[stx]

            linpred = np.dot(exog_s, params)
            linpred += self.b_params[surv.clusters_s[stx]]
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)

            xp0, xp1, xp2 = 0., 0., 0.

            # Iterate backward through the unique failure times.
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                if len(ix) > 0:
                    xp0 += e_linpred[ix].sum()
                    v = exog_s[ix,:]
                    xp1 += (e_linpred[ix][:,None] * v).sum(0)
                    mat = v[None,:,:]
                    elx = e_linpred[ix]
                    xp2 += (mat.T * mat * elx[None,:,None]).sum(1)

                # Account for all cases that fail at this point.
                m = len(uft_ix[i])
                hess += m*(xp2 / xp0  - np.outer(xp1, xp1) / xp0**2)

                # Update for new cases entering the risk set.
                ix = surv.risk_exit[stx][i]
                if len(ix) > 0:
                    xp0 -= e_linpred[ix].sum()
                    v = exog_s[ix,:]
                    xp1 -= (e_linpred[ix][:,None] * v).sum(0)
                    mat = v[None,:,:]
                    elx = e_linpred[ix]
                    xp2 -= (mat.T * mat * elx[None,:,None]).sum(1)

        return -hess

    def efron_hessian_beta(self, params):
        """
        Returns the Hessian matrix of the partial log-likelihood
        evaluated at `params`, using the Efron method to handle tied
        times.
        """

        surv = self.surv

        hess = 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            exog_s = surv.exog_s[stx]

            linpred = np.dot(exog_s, params)
            linpred += self.b_params[surv.clusters_s[stx]]
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)

            xp0, xp1, xp2 = 0., 0., 0.

            # Iterate backward through the unique failure times.
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                if len(ix) > 0:
                    xp0 += e_linpred[ix].sum()
                    v = exog_s[ix,:]
                    xp1 += (e_linpred[ix][:,None] * v).sum(0)
                    mat = v[None,:,:]
                    elx = e_linpred[ix]
                    xp2 += (mat.T * mat * elx[None,:,None]).sum(1)

                ixf = uft_ix[i]
                if len(ixf) > 0:
                    v = exog_s[ixf,:]
                    xp0f = e_linpred[ixf].sum()
                    xp1f = (e_linpred[ixf][:,None] * v).sum(0)
                    mat = v[None,:,:]
                    elx = e_linpred[ixf]
                    xp2f = (mat.T * mat * elx[None,:,None]).sum(1)

                # Account for all cases that fail at this point.
                m = len(uft_ix[i])
                J = np.arange(m, dtype=np.float64) / m
                c0 = xp0 - J*xp0f
                mat = (xp2[None,:,:] - J[:,None,None]*xp2f) / c0[:,None,None]
                hess += mat.sum(0)
                mat = (xp1[None, :] - np.outer(J, xp1f)) / c0[:, None]
                mat = mat[:, :, None] * mat[:, None, :]
                hess -= mat.sum(0)

                # Update for new cases entering the risk set.
                ix = surv.risk_exit[stx][i]
                if len(ix) > 0:
                    xp0 -= e_linpred[ix].sum()
                    v = exog_s[ix,:]
                    xp1 -= (e_linpred[ix][:,None] * v).sum(0)
                    mat = v[None,:,:]
                    elx = e_linpred[ix]
                    xp2 -= (mat.T * mat * elx[None,:,None]).sum(1)

        return -hess



    # loglike, score and hessian for frailty
    def loglike_frailty(self, params):
        """
        Returns the log partial likelihood function evaluated at
        `params`.
        """

        if self.ties == "breslow":
            return self.breslow_loglike_frailty(params)
        elif self.ties == "efron":
            return self.efron_loglike_frailty(params)

    def score_frailty(self, params):
        """
        Returns the score function evaluated at `params` for the
        frailty terms.
        """

        if self.ties == "breslow":
            return self.breslow_gradient_frailty(params)
        elif self.ties == "efron":
            return self.efron_gradient_frailty(params)

    def hessian_frailty(self, params):
        """
        Returns the Hessian matrix of the log partial likelihood
        function evaluated at `params` for the frailty terms.
        """

        if self.ties == "breslow":
            return self.breslow_hessian_frailty(params)
        else:
            return self.efron_hessian_frailty(params)

    def breslow_loglike_frailty(self, params):
        """
        Returns the value of the log partial likelihood function
        evaluated at `params`, using the Breslow method to handle tied
        times for the frailty terms.
        """

        surv = self.surv

        like = 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            # clusters in stratum
            clust_ix = surv.clusters_s[stx]

            uft_ix = surv.ufailt_ix[stx]
            exog_s = surv.exog_s[stx]
            nuft = len(uft_ix)

            linpred = np.dot(exog_s, self.beta_params)
            linpred += params[clust_ix]
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)

            xp0 = 0.

            # Iterate backward through the unique failure times.
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                xp0 += e_linpred[ix].sum()

                # Account for all cases that fail at this point.
                ix = uft_ix[i]
                like += (linpred[ix] - np.log(xp0)).sum()

                # Update for cases leaving the risk set.
                ix = surv.risk_exit[stx][i]
                xp0 -= e_linpred[ix].sum()

        like -= params.T.dot(params) / (2 * self.theta)

        return like

    def efron_loglike_frailty(self, params):
        """
        Returns the value of the log partial likelihood function
        evaluated at `params`, using the Efron method to handle tied
        times for the frailty terms.
        """

        surv = self.surv

        like = 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            # clusters in stratum
            clust_ix = surv.clusters_s[stx]

            # exog and linear predictor for this stratum
            exog_s = surv.exog_s[stx]
            linpred = np.dot(exog_s, self.beta_params)
            linpred += params[clust_ix]
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)

            xp0 = 0.

            # Iterate backward through the unique failure times.
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                xp0 += e_linpred[ix].sum()
                xp0f = e_linpred[uft_ix[i]].sum()

                # Account for all cases that fail at this point.
                ix = uft_ix[i]
                like += linpred[ix].sum()

                m = len(ix)
                J = np.arange(m, dtype=np.float64) / m
                like -= np.log(xp0 - J*xp0f).sum()

                # Update for cases leaving the risk set.
                ix = surv.risk_exit[stx][i]
                xp0 -= e_linpred[ix].sum()

        like -= params.T.dot(params) / (2 * self.theta)

        return like

    def breslow_gradient_frailty(self, params):
        """
        Returns the gradient of the log partial likelihood, using the
        Breslow method to handle tied times for the frailty terms.
        """

        surv = self.surv

        grad = 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            # Indices of subjects in the stratum
            strat_ix = surv.stratum_rows[stx]

            # clusters in stratum
            clust_ix = surv.clusters_s[stx]

            # Unique failure times in the stratum
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)

            # exog and linear predictor for the stratum
            exog_s = surv.exog_s[stx]
            linpred = np.dot(exog_s, self.beta_params)
            linpred += params[clust_ix]
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)

            xp0, xp1 = 0., 0.

            # Iterate backward through the unique failure times.
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                if len(ix) > 0:
                    v = np.zeros((len(ix), len(params)))
                    v[range(0, len(clust_ix[ix])),clust_ix[ix]] = 1
                    xp0 += e_linpred[ix].sum()
                    xp1 += (e_linpred[ix][:,None] * v).sum(0)

                # Account for all cases that fail at this point.
                ix = uft_ix[i]
                v = np.zeros((len(ix), len(params)))
                v[range(0, len(clust_ix[ix])),clust_ix[ix]] = 1
                grad += (v - xp1 / xp0).sum(0)

                print xp1
                print xp0
                print grad
                print ix, stx

                # Update for cases leaving the risk set.
                ix = surv.risk_exit[stx][i]
                if len(ix) > 0:
                    v = np.zeros((len(ix), len(params)))
                    v[range(0, len(clust_ix[ix])),clust_ix[ix]] = 1
                    xp0 -= e_linpred[ix].sum()
                    xp1 -= (e_linpred[ix][:,None] * v).sum(0)
#                print grad

        grad -= params / self.theta

        return grad

    def efron_gradient_frailty(self, params):
        """
        Returns the gradient of the log partial likelihood evaluated
        at `params`, using the Efron method to handle tied times for
        the frailty terms.
        """

        surv = self.surv

        grad = 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            # Indices of cases in the stratum
            strat_ix = surv.stratum_rows[stx]

            # clusters in stratum
            clust_ix = surv.clusters_s[stx]

            # exog and linear predictor of the stratum
            exog_s = surv.exog_s[stx]
            linpred = np.dot(exog_s, self.beta_params)
            linpred += params[clust_ix]
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)

            xp0, xp1 = 0., 0.

            # Iterate backward through the unique failure times.
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                if len(ix) > 0:
                    v = np.zeros((len(ix), len(params)))
                    v[range(0, len(clust_ix[ix])),clust_ix[ix]] = 1
                    xp0 += e_linpred[ix].sum()
                    xp1 += (e_linpred[ix][:,None] * v).sum(0)
                ixf = uft_ix[i]
                if len(ixf) > 0:
                    v = np.zeros((len(ix), len(params)))
                    v[range(0, len(clust_ix[ix])),clust_ix[ixf]] = 1
                    xp0f = e_linpred[ixf].sum()
                    xp1f = (e_linpred[ixf][:,None] * v).sum(0)

                    # Consider all cases that fail at this point.
                    grad += v.sum(0)

                    m = len(ixf)
                    J = np.arange(m, dtype=np.float64) / m
                    numer = xp1 - np.outer(J, xp1f)
                    denom = xp0 - np.outer(J, xp0f)
                    ratio = numer / denom
                    rsum = ratio.sum(0)
                    grad -= rsum

                # Update for cases leaving the risk set.
                ix = surv.risk_exit[stx][i]
                if len(ix) > 0:
                    v = np.zeros((len(ix), len(params)))
                    v[range(0, len(clust_ix[ix])),clust_ix[ix]] = 1
                    xp0 -= e_linpred[ix].sum()
                    xp1 -= (e_linpred[ix][:,None] * v).sum(0)
#                print grad

        grad -= params / self.theta

        return grad

    def breslow_hessian_frailty(self, params):
        """
        Returns the Hessian of the log partial likelihood evaluated at
        `params`, using the Breslow method to handle tied times for the
        frailty terms.
        """

        surv = self.surv

        hess = 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)

            # clusters in stratum
            clust_ix = surv.clusters_s[stx]

            exog_s = surv.exog_s[stx]

            linpred = np.dot(exog_s, self.beta_params)
            linpred += params[clust_ix]
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)

            xp0, xp1, xp2 = 0., 0., 0.

            # Iterate backward through the unique failure times.
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                if len(ix) > 0:
                    xp0 += e_linpred[ix].sum()
                    v = np.zeros((len(ix), len(params)))
                    v[range(0, len(clust_ix[ix])),clust_ix[ix]] = 1
                    xp1 += (e_linpred[ix][:,None] * v).sum(0)
                    mat = exog_s[ix,:][None,:,:]
                    mat = v[None,:,:]
                    elx = e_linpred[ix]
                    xp2 += (mat.T * mat * elx[None,:,None]).sum(1)

                # Account for all cases that fail at this point.
                m = len(uft_ix[i])
                hess += m*(xp2 / xp0  - np.outer(xp1, xp1) / xp0**2)

                # Update for new cases entering the risk set.
                ix = surv.risk_exit[stx][i]
                if len(ix) > 0:
                    xp0 -= e_linpred[ix].sum()
                    v = np.zeros((len(ix), len(params)))
                    v[range(0, len(clust_ix[ix])),clust_ix[ix]] = 1
                    xp1 -= (e_linpred[ix][:,None] * v).sum(0)
                    mat = v[None,:,:]
                    elx = e_linpred[ix]
                    xp2 -= (mat.T * mat * elx[None,:,None]).sum(1)
#                print hess

        hess += 1. / self.theta

        return -hess

    def efron_hessian_frailty(self, params):
        """
        Returns the Hessian matrix of the partial log-likelihood
        evaluated at `params`, using the Efron method to handle tied
        times for the frailty terms.
        """

        surv = self.surv

        hess = 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            exog_s = surv.exog_s[stx]

            # clusters in stratum
            clust_ix = surv.clusters_s[stx]

            linpred = np.dot(exog_s, self.beta_params)
            linpred += params[clust_ix]
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)

            xp0, xp1, xp2 = 0., 0., 0.

            # Iterate backward through the unique failure times.
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                if len(ix) > 0:
                    xp0 += e_linpred[ix].sum()
                    v = np.zeros((len(ix), len(params)))
                    v[range(0, len(clust_ix[ix])),clust_ix[ix]] = 1
                    xp1 += (e_linpred[ix][:,None] * v).sum(0)
                    mat = v[None,:,:]
                    elx = e_linpred[ix]
                    xp2 += (mat.T * mat * elx[None,:,None]).sum(1)

                ixf = uft_ix[i]
                if len(ixf) > 0:
                    v = np.zeros((len(ix), len(params)))
                    v[range(0, len(clust_ix[ix])),clust_ix[ixf]] = 1
                    xp0f = e_linpred[ixf].sum()
                    xp1f = (e_linpred[ixf][:,None] * v).sum(0)
                    mat = v[None,:,:]
                    elx = e_linpred[ixf]
                    xp2f = (mat.T * mat * elx[None,:,None]).sum(1)

                # Account for all cases that fail at this point.
                m = len(uft_ix[i])
                J = np.arange(m, dtype=np.float64) / m
                c0 = xp0 - J*xp0f
                mat = (xp2[None,:,:] - J[:,None,None]*xp2f) / c0[:,None,None]
                hess += mat.sum(0)
                mat = (xp1[None, :] - np.outer(J, xp1f)) / c0[:, None]
                mat = mat[:, :, None] * mat[:, None, :]
                hess -= mat.sum(0)

                # Update for new cases entering the risk set.
                ix = surv.risk_exit[stx][i]
                if len(ix) > 0:
                    xp0 -= e_linpred[ix].sum()
                    v = np.zeros((len(ix), len(params)))
                    v[range(0, len(clust_ix[ix])),clust_ix[ix]] = 1
                    xp1 -= (e_linpred[ix][:,None] * v).sum(0)
                    mat = v[None,:,:]
                    elx = e_linpred[ix]
                    xp2 -= (mat.T * mat * elx[None,:,None]).sum(1)
#                print hess

        hess += 1. / self.theta

        return -hess
