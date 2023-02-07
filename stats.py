# Copyright 2022 by the author(s) of CHI2023 submission "Short-Form
# Videos Degrade Our Capacity to Retain Intentions: Effect of Context
# Switching On Prospective Memory". All rights reserved.
#
# Use of this source code is governed by a GPLv3 license that
# can be found in the LICENSE file.

from scipy.stats import shapiro, ttest_ind, mannwhitneyu, ranksums, levene, bartlett
from typing import Tuple, List
import itertools

import numpy as np
import pandas as pd
import pingouin as pg
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa


def fit_ex_gaussian(x: np.ndarray, epochs=500, progress=False) -> Tuple[float, float, float]:
    """
    Fit an expotential Gaussian distribution to the data using Tensorflow Probability.

    An exponential Gaussian distribution is a combination of an exponential
    distribution and a Gaussian distribution which can be modeled using numpy:
    np.random.exponential(scale=1/lambda, size=n) + np.random.normal(loc=location, scale=scale, size=n)

    This function returns the parameters of the exponential Gaussian distribution: location, scale, and lambda.
    """
    tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(3),
        tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.ExponentiallyModifiedGaussian(
                loc=t[..., :1],
                scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]),
                rate=tf.math.softplus(t[..., 2:]),
            )
        ),
    ])
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.2, 10, 0.99, staircase=False, name=None
    )
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr_schedule),
                  loss=lambda y, distr: -distr.log_prob(y))
    model.fit(np.zeros(x.shape)[:, np.newaxis], x,
              epochs=epochs, verbose=0, callbacks=None if not progress else [tqdm_callback])
    return model.weights[1][0], tf.math.softplus(0.05 * model.weights[1][1]), tf.math.softplus(model.weights[1][2])

def pairwise_test(df: pd.DataFrame, factor: str, response: str, pairs=None, verbose=False) -> pd.DataFrame:
    """
    Perform pairwise statistical tests on the data.
    If pairs is not specified, this will try to run a full pairwise test of all levels.
    """
    ret = pd.DataFrame()
    if pairs == None:
        # If there are no specified pairs, use all pairs of factors.
        pairs = list(itertools.combinations(df[factor].unique(), 2))

    for pair in pairs:
        ret = pd.concat([ret, two_sample_test(
            df, factor, pair[0], pair[1], response, verbose=verbose)])
    return ret


def two_sample_test(df: pd.DataFrame, factor: str, level1: str, level2: str, response: str, verbose=False):
    """two_population_test run an independent sample test on the given data frame. This function applies a
    decision tree based on the characteristic of the data. If the specific two independent samples are normally
    distributed, then this function apply a t-test (including the consideration of homogeneity of variance).
    If the specific two independent samples are not normally distributed, then this function apply a Mann-Whitney
    U test (as known as Wilcoxon's rank sum test).

    This function can also be used for post-hoc pairwise comparisons between two levels.
    """
    # Test normality using Shapiro–Wilk_test for dependent variables.
    p1 = df[df[factor] == level1][response]
    p2 = df[df[factor] == level2][response]
    shp1 = shapiro(p1)
    shp2 = shapiro(p2)

    normal = (shp1.pvalue >= 0.05) and (shp2.pvalue >= 0.05)
    if verbose:
        if (shp1.pvalue < 0.05):
            print(
                f"The Shapiro-Wilk test shows {response} violates normality on {level1} (W={shp1.statistic:.3f}, p={shp1.pvalue:.3f}).")
        else:
            print(
                f"The Shapiro-Wilk test shows {response} is normally distributed on {level1} (W={shp1.statistic:.3f}, p={shp1.pvalue:.3f}).")
        if (shp2.pvalue < 0.05):
            print(
                f"The Shapiro-Wilk test shows {response} violates normality on {level2} (W={shp2.statistic:.3f}, p={shp2.pvalue:.3f}).")
        else:
            print(
                f"The Shapiro-Wilk test shows {response} is normally distributed on {level2} (W={shp2.statistic:.3f}, p={shp2.pvalue:.3f}).")

    # Test homogeneity of variance using Bartlett's test or Levene's test depending on normality.
    eqvar = False
    if normal:
        # Bartlett’s test for equal variances if normal.
        # null hypothesis that all input samples are from populations with equal variances.
        # hence pvalue < 0.05 => variances are different
        wvar, pvar = bartlett(p1, p2)
        vartest_name = "Bartlett's test"
        eqvar = pvar >= 0.05
        if verbose:
            if (pvar >= 0.05):
                print(
                    f'The Bartlett test shows {response} has equal variances on {level1} and {level2} (p={pvar:.3f}, W={wvar:.3f}).')
            else:
                print(
                    f'The Bartlett test shows {response} has different variances on {level1} and {level2} (p={pvar:.3f}, W={wvar:.3f}).')
    else:
        # Levene's test is more robust non-normal data.
        # null hypothesis that all input samples are from populations with equal variances.
        # hence pval < 0.05 => variances are different.
        wvar, pvar = levene(p1, p2)
        vartest_name = "Levene's test"
        eqvar = pvar >= 0.05
        if verbose:
            if (pvar >= 0.05):
                print(
                    f'The Levene test shows {response} has equal variances on {level1} and {level2} (p={pvar:.3f}, W={wvar:.3f}).')
            else:
                print(
                    f'The Levene test shows {response} has different variances on {level1} and {level2} (p={pvar:.3f}, W={wvar:.3f}).')

    if normal:
        n = 't'
        r = ttest_ind(p1, p2, equal_var=eqvar)
        m1 = f'Mean={p1.mean():.3f}'
        m2 = f'Mean={p2.mean():.3f}'
    else:
        # U test is robust to the violation of homogeneity of variance.
        alternative = 'two-sided'
        if p1.median() < p2.median():
            alternative = 'less'
        elif p1.median() > p2.median():
            alternative = 'greater'

        # check if there are ties between p1 and p2
        all_values = np.concatenate((p1.to_numpy(), p2.to_numpy()))
        if np.unique(all_values).size == all_values.size:  # no ties
            r = ranksums(p1, p2, alternative)
            n = 'ranksum'
        else:  # mannwhiteneyu can handle ties.
            r = mannwhitneyu(p1, p2, alternative=alternative)
            n = 'u'
        m1 = f'Median={p1.median():.3f}'
        m2 = f'Median={p2.median():.3f}'

    # Mann-Whitney test is also called Wilcoxon rank-sum test. Since it is the same test there is no need to explain the difference
    # cf. https://stats.stackexchange.com/questions/113936/what-is-the-difference-between-the-mann-whitney-and-wilcoxon-rank-sumtest
    # However, note that in the implementation of the Mann-Whitney test can handle ties in the data, whereas the rank sum does not.
    test_name = 't-test'
    if n == 'u':
        test_name = 'Mann-Whitney U test'
    elif n == 'ranksum':
        test_name = 'Wilcoxon rank-sum test'

    if verbose:
        if r.pvalue < 0.05:
            if r.pvalue < 0.001:
                print(
                    f"A {test_name} shows a significant difference ({n}={r.statistic:.3f}, p<.001) between {level1} ({m1}) and {level2} ({m2}).")
            else:
                print(f"A {test_name} shows a significant difference ({n}={r.statistic:.3f}, p={r.pvalue:.3f}) between {level1} ({m1}) and {level2} ({m2}).")
        else:
            print(f"A {test_name} cannot found significance between {level1} and {level2} ({n}={r.statistic:.3f}, p={r.pvalue:.3f}) using {test_name}.")

    sig = 'n.s.'
    if r.pvalue < 0.001:
        sig = '***'
    elif r.pvalue < 0.01:
        sig = '**'
    elif r.pvalue < 0.05:
        sig = '*'

    results = np.array([test_name, response, level1, p1.mean() if normal else p1.median(), p1.var(), level2, p2.mean() if normal else p2.median(
    ), p2.var(), sig, r.pvalue, r.statistic, shp1.pvalue, shp1.statistic, shp2.pvalue, shp2.statistic, vartest_name, pvar, wvar])
    return pd.DataFrame([results], columns=[
        'test_name', 'response', 'level1', 'm1', 'var1', 'level2', 'm2', 'var2',
        'sig', 'p', 'W', 'p_shapiro1', 'W_shapiro1', 'p_shapiro_2', 'W_shapiro2', 'var_test_name', 'p_variance', 'W_variance'])
