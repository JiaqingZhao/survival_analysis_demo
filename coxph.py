"""
Modified from https://github.com/sebp/scikit-survival with

1. Simplified process for demonstration purposes

2. More comments to explain the Cox Regression process aiming to make the process more transparent

3. Minor fixes

Citations:
1. Pölsterl, S., Navab, N., and Katouzian, A., Fast Training of Support Vector Machines for Survival Analysis.
    Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2015, Porto, Portugal,
    Lecture Notes in Computer Science, vol. 9285, pp. 243-259 (2015)
2. Pölsterl, S., Navab, N., and Katouzian, A., An Efficient Training Algorithm for Kernel Survival Support Vector
    Machines. 4th Workshop on Machine Learning in Life Sciences, 23 September 2016, Riva del Garda, Italy
3. Pölsterl, S., Gupta, P., Wang, L., Conjeti, S., Katouzian, A., and Navab, N., Heterogeneous ensembles for
    predicting survival of metastatic, castrate-resistant prostate cancer patients. F1000Research, vol. 5, no. 2676
    (2016).
"""

import warnings
import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import check_is_fitted


class cox_ph_regressor(BaseEstimator):


    def __init__(self, n_iter=200, tol=1e-9):
        self.n_iter = n_iter # iterate rounds
        self.tol = tol # stopping criteria
        self.coef_ = None # final coef_
        self.x = None
        self.event = None
        self.time = None


    def fit(self, X, y):

        # break y into event and time, sort time descending
        event, time =  y[y.columns[0]].values, y[y.columns[1]].values

        self.order = np.argsort(-time, kind="quicksort")
        self.x = X.values[self.order, :]
        self.event = event[self.order]
        self.time = time[self.order]

        # initialize
        w = np.zeros(self.x.shape[1])
        w_prev = w
        i = 0
        loss = float('inf')

        # start iterations
        while True:

            if i >= self.n_iter:
                #print("Reached maximum iteration. Current iteration " + str(i))
                break

            ## get gradient and hessian
            gradient, hessian = self._get_gradient_hess(w)

            ## solve beta step
            delta = solve(hessian,
                          gradient,
                          overwrite_a=False,
                          overwrite_b=False,
                          check_finite=False)

            if not np.all(np.isfinite(delta)):
                raise ValueError("search direction contains NaN or infinite values")

            ## update w and loss
            w_new = w - delta

            loss_new = self._neg_log_partial_likelihood(w_new)

            ## perform step-halving if negative log-likelihood does not decrease
            if loss_new > loss:
                w = (w_prev + w) / 2
                loss = self._neg_log_partial_likelihood(w)
                i += 1
                continue

            w_prev = w
            w = w_new


            res = np.abs(1 - (loss_new / loss))

            if res < self.tol:
                #print("Step size can't exceed tol")
                break

            loss = loss_new
            i += 1

        # store results
        self.coef_ = w

        self.cum_baseline_hazard_ = self._breslow_estimator(event, time)

        self.baseline_survival_ = self._step_func(self.cum_baseline_hazard_[0],
                                               np.exp(- self.cum_baseline_hazard_[1]))

        return self


    def _get_gradient_hess(self,w):

        n_samples, n_features = self.x.shape

        # initialize
        sum_risk_j = 0
        sum_risk_j_xi = 0
        sum_risk_j_xi_xi_t = 0
        gradient = np.zeros((1, n_features), dtype=float)
        hessian = np.zeros((n_features, n_features), dtype=float)

        # calculate log risk core
        exp_xw = np.exp(np.dot(self.x, w))
        k = 0
        # iterate through samples such that the iterated part contains individuals at risk at time time[i]

        for i in range(n_samples):

            ti = self.time[i]
            #while k < n_samples and ti == self.time[k]:
            ## components for calculating gradient/hessian matrix.
            xi = self.x[i:i + 1]
            xi_t_xi = np.dot(xi.T, xi)
            sum_risk_j += exp_xw[i]
            sum_risk_j_xi += exp_xw[i] * xi
            sum_risk_j_xi_xi_t += exp_xw[i] * xi_t_xi
            #k += 1

            ## if event happens, death or censor
            if self.event[i]:

                hessian_left = sum_risk_j_xi_xi_t / sum_risk_j
                hessian_right_b4_sqr = sum_risk_j_xi / sum_risk_j
                hessian_right = np.dot(hessian_right_b4_sqr.T, hessian_right_b4_sqr)

                gradient -= (xi - sum_risk_j_xi / sum_risk_j) / n_samples
                hessian += (hessian_left - hessian_right) / n_samples
        #print(ikset)
        return gradient.ravel(), hessian

    def _neg_log_partial_likelihood(self,w):

        # initialize
        loss = 0
        sum_risk_j = 0
        n_samples = self.x.shape[0]
        xw = np.dot(self.x, w)

        # iterate through samples
        for i in range(n_samples):
            ti = self.time[i]

            #while k < n_samples and ti == self.time[k]:
            sum_risk_j += np.exp(xw[i])
            #k += 1


            if self.event[i]:

                ## the negative log partial likelihood

                loss -= (xw[i] - np.log(sum_risk_j)) / n_samples

        return loss

    def _breslow_estimator(self, event, time):

        risk_score = np.exp(np.dot(self.x, self.coef_))

        #(self.coef_,"end")
        uniq_times, n_events, n_at_risk = self._compute_counts(event, time, self.order)

        divisor = np.empty(n_at_risk.shape, dtype=np.float_)
        value = np.sum(risk_score)
        divisor[0] = value
        k = 0
        for i in range(1, len(n_at_risk)):
            d = n_at_risk[i - 1] - n_at_risk[i]
            value -= risk_score[k:(k + d)].sum()
            k += d
            divisor[i] = value

        assert k == n_at_risk[0] - n_at_risk[-1]

        y = np.cumsum(n_events / divisor)
        return self._step_func(uniq_times, y)

    def _step_func(self, x, y, a = 1., b = 0.):

        o = np.argsort(x, kind="quicksort")
        x = x[o]

        y = y[o]
        return x, a * y + b

    def predict(self, X):

        if self.coef_.any():
            x = X.values
            return np.dot(x, self.coef_)
        else:
            raise ValueError("Invalid coefficients")

    def predict_cumulative_hazard_function(self):

        risk_score = np.exp(self.predict(X))
        n_samples = risk_score.shape[0]
        funcs = np.empty(n_samples, dtype=np.object_)
        for i in range(n_samples):
            funcs[i] = _step_func(x=self.cum_baseline_hazard_.x,
                                    y=self.cum_baseline_hazard_.y,
                                    a=risk_score[i])
        return funcs

    def predict_survival_function(self,X):

        risk_score = np.exp(self.predict(X))
        n_samples = risk_score.shape[0]
        funcs = np.empty(n_samples, dtype=np.object_)
        for i in range(n_samples):
            funcs[i] = self._step_func(x=self.baseline_survival_[0],
                                    y=np.power(self.baseline_survival_[1], risk_score[i]))
        return funcs

    def _compute_counts(self, event, time, order=None):

        n_samples = event.shape[0]

        if order is None:
            order = np.argsort(time, kind="mergesort")

        uniq_times = np.empty(n_samples, dtype=time.dtype)
        uniq_events = np.empty(n_samples, dtype=np.int_)
        uniq_counts = np.empty(n_samples, dtype=np.int_)

        i = 0
        prev_val = time[order[0]]
        j = 0
        while True:
            count_event = 0
            count = 0
            while i < n_samples and prev_val == time[order[i]]:
                if event[order[i]]:
                    count_event += 1

                count += 1
                i += 1

            uniq_times[j] = prev_val
            uniq_events[j] = count_event
            uniq_counts[j] = count
            j += 1

            if i == n_samples:
                break

            prev_val = time[order[i]]

        times = np.resize(uniq_times, j)
        n_events = np.resize(uniq_events, j)
        total_count = np.resize(uniq_counts, j)

        # offset cumulative sum by one
        total_count = np.concatenate(([0], total_count))
        n_at_risk = n_samples - np.cumsum(total_count)

        return times, n_events, n_at_risk[:-1]


if __name__ == '__main__':


    import pandas as pd
    whas = pd.read_csv("whas_new.csv")
    X = whas.iloc[:, 2:15]




    y = whas.iloc[:, [-7, -5]]


    cpr = cox_ph_regressor(n_iter=100)
    cpr.fit(X, y)

    print(pd.Series(cpr.coef_, index=X.columns))



