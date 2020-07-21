import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def ecdf(arr):
    """
    Calculate Empirical Cumulative Distribution Function. Similar function
    as Histogram but no bin-bias.

    Parameters:
    arr: numpy array with continuous variables

    Returns:
    x = numpy array on interval [arr.min, arr.max]
    y = ecdf of arr on interval [0, 1]
    """
    n = len(arr)
    x = np.sort(arr)
    y = np.arange(n)/n
    return x,y


def discrete_pdp(clf, df, col, vals):
    """
    Partial Dependence Plot (PDP) for a discrete feature <col>, where vals
    denote the values this feature can take, for each of which the model
    prediction is measured holding all other features constant.

    Parameters:
    df: pandas dataframe, input df for the ml-model
    col: str, column name of df to be varied in pdp
    vals: numpy_array of the feature values used in the PDP

    Returns:
    output: numpy_array of the model repone for corresponding val
    """

    # copy prior to updating
    X = df.copy()

    # calculate effect for each value
    output = []
    for val in vals:
        X[col] = val
        y_pred = clf.predict(X)
        output.append(y_pred.mean())

    return np.array(output)


def continuous_pdp(clf, df, col, val_start = 0, val_end = 1, n = 10):
    """
    Partial Dependence Plot (PDP) for a continuous feature <col>. This feature
    is varied between <val_start> and <val_end> and model prediction is measured
    holding all other features constant.

    Parameters:
    df: pandas dataframe, input df for the ml-model
    col: str, column name of df to be varied in pdp
    val_start: int, start or range for the feature <col> (often X[col].min())
    val_end: int, end or range for the feature <col> (often X[col].max())
    n: int, number of points to measure

    Returns:
    vals: numpy_array of the feature values used in the PDP
    output: numpy_array of the model repone for corresponding val
    """
    # generate our value_range
    vals = np.linspace(val_start, val_end, n)

    # calculate the pdp
    output = discrete_pdp(clf, df, col, vals)

    return vals, output


def plot_continuous_pdp(clf, df, col, n = 10, val_range = None, figsize = (9,7)):
    """
    Partial Dependence Plot (PDP) and ECDF for a continuous feature <col>.
    This feature is varied between n values in the <val_range>  and model
    prediction is measuredholding all other features constant.

    Parameters:
    df: pandas dataframe, input df for the ml-model
    col: str, column name of df to be varied in pdp
    val_range: list, range for the feature <col> [val_start, val_end]
    n: int, number of points to measure
    figsize: tuple, matplotlib figsize, default (9,7)
    """
    # plot setup
    fig, ax = plt.subplots(2,1, figsize = figsize)

    # set the range to vary our value in pdp
    if val_range == None:
        val_start = df[col].min()
        val_end = df[col].max()
    else:
        val_start = val_range[0]
        val_end = val_range[1]

    #calculate the pdp
    vals, output = continuous_pdp(clf, df, col, val_start = val_start,
                                  val_end = val_end, n = n)
    ax[0].plot(vals, output)
    ax[0].scatter(vals, output)
    # get axis range to match in ecdf
    ymin, ymax = ax[0].get_xlim()
    ax[0].set_ylabel('no-show rate')


    # ECDF
    x,y = ecdf(df[col])
    ax[1].scatter(x,y, s= 2, alpha = .1)

    # crop ecdf axis to match pdp-val_range
    # capture population frac we keep with this
    covered_frac = len(df[df[col] <= val_end])/len(df)
    ax[1].set_xlim(ymin, ymax)

    # format
    ax[1].set_xlabel(col)
    ax[1].set_ylabel('ecdf (pop. frac: {})'.format(round(covered_frac,2)))
    plt.show()


def plot_discrete_pdp(clf, df, col, figsize = (9,7)):
    """
    Partial Dependence Plot (PDP) and ECDF for the values of discrete
    feature <col>. Model prediction is measured holding all other features
    constant.

    Parameters:
    df: pandas dataframe, input df for the ml-model
    col: str, column name of df to be varied in pdp
    """
    # setup
    fig, ax = plt.subplots(2,1, figsize = figsize)
    base_color = sns.color_palette()[0]

    # plot population distribution
    data = df[col].value_counts(normalize=True).reset_index()
    sns.barplot(ax = ax[0], data = data, x = 'index', y = col, color=base_color)
    ax[0].set_ylabel('population %')
    ax[0].set_xlabel('') #empty

    # plot PDP
    vals = np.sort(df[col].unique())
    output = discrete_pdp(clf, df, col, vals)

    sns.barplot(ax = ax[1], x = vals, y = output, color=base_color)

    ax[1].set_ylabel('no-show rate')
    ax[1].set_xlabel(col)

    plt.show()


##### NEW FUNCTION - only for this dataset
# note: would have been easier to encode 'dow' as 0-6

def daily_pdp(clf, df, col_basis, figsize = (9,7)):
    """
    Calculate daily PDP (days are dummy vars)
    """
    # get the relevant columns
    dm_cols = [col for col in df.columns if col[:len(col_basis)] == col_basis]

    # output lists
    dow = ['monday']
    output = []
    population = []

    # handle monday (baseline = droppped dummy)
    X = df.copy()
    X[dm_cols] = 0
    y_pred = clf.predict(X)

    output.append(y_pred.mean())
    population.append(len(df) - df[dm_cols].sum().sum())

    # other days
    for sel_day in dm_cols:
        X = df.copy()

        # all to zero
        X[dm_cols] = 0

        # relevant day to 1
        X[sel_day] = 1
        y_pred = clf.predict(X)

        # update output
        output.append(y_pred.mean())
        dow.append(sel_day[len(col_basis)+1:])
        population.append(df[sel_day].sum())

    #### MAKE THE PDP PLOTS
    fig, ax = plt.subplots(2,1, figsize = figsize)
    base_color = sns.color_palette()[0]

    # plot population distribution
    sns.barplot(ax = ax[0], x = dow, y = population, color=base_color)
    ax[0].set_ylabel('appointments')
    ax[0].set_xlabel('')

    # plot PDP
    sns.barplot(ax = ax[1], x = dow, y = output, color=base_color)
    ax[1].set_ylabel('no-show rate')
    ax[1].set_xlabel('dow')

    plt.show()
