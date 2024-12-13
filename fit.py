"""

Author: Jean Paul
Email: jean.louys-sanso@uibk.ac.at

Creation Date: 2020
 Last Modification Date: 2024-12-13 10:18:33

This script contains useful functions to implement quick general fitting routines. It was initially created during my Bachelor and I have been adding features since. This version is the most recent one and contains comments in English. It is mainly a wrapper for scipy.optimize.curve_fit and scipy.odr.

"""

import pandas as pd
import numpy as np
from sympy import *  # noqa: F403
from scipy import optimize, odr, stats
import matplotlib.pyplot as plt
import math
import matplotlib
from colorama import Fore


def readcsv(dataset):
    """Loads a csv file.

    Args:
        dataset (str): dataset name

    Returns:
        list: list of columns in the dataset
    """
    data = pd.read_csv(dataset)
    columnas = []
    for columna in range(len(data.columns)):
        x = list(data.iloc[:, columna])
        for i in range(len(x)):
            x[i] = float(x[i])
        columnas.append(x)
    return columnas


def decimales(
    x,
):
    """Returns the number of decimal places of a number. The goal is to get this information to later adjust the error to a significant figure and approximate the representative value to as many decimal places as the error has.

    Args:
        x (float): number to extract the decimal places from

    Returns:
        int: number of decimal places
    """
    # esta es medio rara. la idea es que devuelve las cifras decimales (en realidad no) para despues ajustar el error a una cifra significativa y aproximar el valor representativo a tantos decimales como tenga el error.
    try:
        if abs(x) < 1 and x != 0:
            s = 1 - int(math.log10(abs(x)))
        elif x != 0:
            s = -int(math.log10(abs(x)))
        else:
            s = -1
    except OverflowError:
        print(
            Fore.YELLOW + "Rounding error in decimales function. OverflowError in x=",
            x,
            Fore.RESET,
        )
    return s


def plottearunafuncion(
    funcion,
    x,
    xtitle="x",
    ytitle="f(x)",
    title="grafico",
    grids=True,
    color="red",
    linew=4,
):
    """Quickly plots a function with some default settings.

    Args:
        funcion (func): function
        x (array-like): x values to evaluate the function at
        xtitle (str, optional): x axis label. Defaults to "x".
        ytitle (str, optional): y axis label. Defaults to "f(x)".
        title (str, optional): name for the png output. Defaults to "grafico".
        grids (bool, optional): whether or not to include grids. Defaults to True.
        color (str, optional): color of the (x,y) pairs. Defaults to "red".
        linew (int, optional): linewidth. Defaults to 4.

    Returns:
        _type_: _description_
    """
    matplotlib.rc("font", size=20)  # controls default text sizes
    matplotlib.rc("axes", titlesize=30)  # fontsize of the axes title
    matplotlib.rc("axes", labelsize=30)  # fontsize of the x and y labels
    matplotlib.rc("xtick", labelsize=24)  # fontsize of the tick labels
    matplotlib.rc("ytick", labelsize=24)  # fontsize of the tick labels
    matplotlib.rc("legend", fontsize=24)  # legend fontsize
    matplotlib.rc("figure", titlesize=18)  # fontsize of the figure title
    plt.figure(figsize=(16, 12))
    y = []
    for i in x:
        y.append(funcion(i))
    plt.plot(x, y, color=color, linewidth=linew)
    plt.grid(grids)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.savefig(title + ".png", bbox_inches="tight", dpi=200)
    return 0


def R_squared(observed, predicted, uncertainty=1):
    """Returns R square measure of goodness of fit for predicted model."""
    weight = 1.0 / uncertainty
    return 1.0 - (np.var((observed - predicted) * weight) / np.var(observed * weight))


def lsq(
    function,
    x,
    y,
    guess,
    yerr=1,
    xerr=0,
    ajustar_errores=False,
    bounds=(-np.inf, +np.inf),
    pesos=None,
    paramhipothesis=None,
    taufac=1,
    maxit=50,
    fix=None,
    sigfig=2,
):
    """Computes a least squares fit of a function to data. Wrapper for curve_fit. Extracts some statistical tests. #~All must be checked, except for the chi squared goodness of fit, which was tested and works.

    Args:
        function (function): function that describes the fit. IMPORTANT: it must be defined in terms of the dependent variable first and then the parameters, in that order.
        x (list): list of values of the independent variable
        y (list): list of values of the other variable
        guess (list): values to start "testing" the parameters. In general, they can be initialized to any value, but you may encounter underflow/overflow or strange behavior.
        yerr (list): absolute errors in y.
        ajustar_errores (bool): if True, adjusts considering the errors in y. If False, ignores them.
        bounds (tuple): tuple with two lists or numbers. The first value of the tuple is for the lower bound of the parameters, the second for the upper bound.
        sigfig (int): number of significant figures to round the error to.
        #? maxit and taufac are placeholders for the ODR method. They are not used in the lsq method.

    Returns:
        list: returns a list with the parameters, another with the errors, the R^2, and the reduced chi^2.
    """
    if paramhipothesis is None:
        paramhipothesis = np.zeros(len(guess))
    x = np.array(x)
    y = np.array(y)
    if ajustar_errores:
        param, error = optimize.curve_fit(
            function, x, y, guess, sigma=yerr, absolute_sigma=True, bounds=bounds
        )
    elif type(pesos) is not type(None):
        param, error = optimize.curve_fit(
            function, x, y, guess, sigma=yerr, absolute_sigma=False, bounds=bounds
        )
    else:
        param, error = optimize.curve_fit(function, x, y, guess, bounds=bounds)
    y_predict = function(x, *param)
    if ajustar_errores:
        r_squared = R_squared(y, y_predict, uncertainty=yerr)
    elif type(pesos) is not type(None):
        r_squared = R_squared(y, y_predict, uncertainty=pesos)
    else:
        r_squared = R_squared(y, y_predict)

    std = np.sqrt(np.diag(error))
    yerr = np.array(yerr)
    if yerr.all() != 0:
        chi2 = np.sum((y - y_predict) ** 2 / yerr**2)
        chi2r = chi2 / (len(y) - len(guess))
    else:
        chi2r = (
            Fore.YELLOW
            + "Chi squared was not estimated because the error in y is 0 or not given."
            + Fore.RESET
        )
    t_stat = (param - np.array(paramhipothesis)) / std
    p_val = stats.t.sf(np.abs(t_stat), len(x) - len(guess)) * 2
    try:
        for i in range(len(std)):
            std[i] = round(
                std[i], -int(math.floor(math.log10(abs(std[i])))) + sigfig - 1
            )  # ? rounds the error to a significant figure
            # param[i] = round(param[i], decimales(std[i]))# round the parameter to the same number of decimal places as the error, #! not working correctly
    except:  # noqa: E722
        pass
    return [param, std, r_squared, chi2r, t_stat, p_val]


def prediccionesyconfianza(x, y, func, param, std, alpha=0.05):
    """Calculates the prediction and confidence intervals for a given model.

    Args:
        x (array-like): The input values.
        y (array-like): The target values.
        func (function): The model function used for prediction.
        param (array-like): The parameters of the model.
        std (array-like): The standard deviations of the parameters.
        alpha (float, optional): The significance level for confidence intervals. Defaults to 0.05.

    Returns:
        list: A list containing the following elements:
            - hires_x (array-like): The high-resolution input values.
            - pred_upper (array-like): The upper bounds of the prediction intervals.
            - pred_lower (array-like): The lower bounds of the prediction intervals.
            - trust_upper (array-like): The upper bounds of the confidence intervals.
            - trust_lower (array-like): The lower bounds of the confidence intervals.
    """
    hires_x = np.linspace(min(x), max(x), 10000)
    param, std = np.array(param), np.array(std)
    ypred = func(x, *param)
    noise = np.std(y - ypred)
    predictions = np.array([np.random.normal(ypred, noise) for j in range(10000)])
    pred_lower, pred_upper = np.quantile(
        predictions, [alpha / 2, 1 - alpha / 2], axis=0
    )
    # confianza
    trust_upper = func(hires_x, *(param + std))
    trust_lower = func(hires_x, *(param - std))
    return [hires_x, pred_upper, pred_lower, trust_upper, trust_lower]


def letras(s=1):
    """Simple way to scale plot fonts.

    Args:
        s (int, optional): scaling factor. Defaults to 1.
    """
    # font = {"family": "DejaVu Sans", "weight": "normal", "size": 22}
    matplotlib.rc("font", size=20 * s)  # controls default text sizes
    matplotlib.rc("axes", titlesize=30 * s)  # fontsize of the axes title
    matplotlib.rc("axes", labelsize=30 * s)  # fontsize of the x and y labels
    matplotlib.rc("xtick", labelsize=24 * s)  # fontsize of the tick labels
    matplotlib.rc("ytick", labelsize=24 * s)  # fontsize of the tick labels
    matplotlib.rc("legend", fontsize=24 * s)  # legend fontsize
    matplotlib.rc("figure", titlesize=18 * s)


def fit(
    func,
    x,
    y,
    xerr=0,
    yerr=0,
    xtitle="x",
    ytitle="y",
    loglog=False,
    grids=True,
    linew=4,
    legend=True,
    title="grafico",
    msize=10,
    xlim=None,
    ylim=None,
    leyendas=[""],
    guess=[],
    ajustar_errores=False,
    bounds=(-np.inf, +np.inf),
    nombre_params=[],
    show=False,
    alpha=0.05,
    trust_predict=False,
    skipsave=False,
    metodo="lsq",
    pesos=None,
    fix=None,
    taufac=1,
    maxit=50,
    paramhipothesis=None,
    silent=False,
    sigfig=2,
    s=1,
):
    """Plots a fit and a scatter plot of the given data. By default, it uses a least square fit, but can be set to ODR. If x and y are lists with multiple groups of data,
    it overlays the fits on a single plot and adjusts the colors accordingly. Returns the fit parameters and some statistical tests.


    Args:
        func (function): fitting function. Can be a list of functions.
        x (list): x values. Can contain lists of x values.
        y (list): y values. Can contain lists of y values.
        xerr (int/list, optional): x error. Defaults to 0.
        yerr (int/list, optional): y error. Defaults to 0.
        xtitle (str, optional): x-axis title. Defaults to 'x'.
        ytitle (str, optional): y-axis title. Defaults to 'y'.
        loglog (bool, optional): if True, plots the axes in log-log scale. Defaults to False.
        grids (bool, optional): grid lines on the plot. Defaults to True.
        legend (bool, optional): legend on the plot. Defaults to True.
        title (str, optional): title on the plot. Defaults to 'grafico'.
        msize (int, optional): marker size. Defaults to 10.
        linew (int, optional): line width. Defaults to 4.
        guess (list, optional): seed for the fitting parameters. Required if spline=False. Defaults to [].
        leyendas (list, optional): list of legends for each fit/spline. Ignored if only one fit/spline is made. Defaults to [''].
        ajustar_errores (bool, optional): considers the parameter errors in the fitting (True). Defaults to False.
        trust_predict (bool, optional): shows confidence and prediction intervals. Defaults to False.
        alpha (float, optional): chooses the 1-alpha confidence interval with alpha/2 on each side. Defaults to 0.05.
        show (bool, optional): shows the interactive plot instead of saving it. Defaults to False.
        skipsave (bool, optional): avoids saving the plot, so that more information can be added after performing the fits. Defaults to False.
        pesos (list, optional): list of (lists of) relative weights for fitting. If not None, automatically sets absigma=False. Defaults to None.
        fix (list, optional): list of 0/1 of the same size as x that fixes some points (0), only works with ODR. Defaults to None.
        taufac (float, optional): float specifying the initial trust region. The initial trust region is equal to taufac times the length of the first computed Gauss-Newton step. Defaults to 1.
        maxit (int, optional): maximum number of iterations allowed for ODR. Defaults to 50.
        metodo (str, optional): fitting method. Valid options: 'lsq', 'odr'. Defaults to 'lsq'.
        nombre_params(list, optional): parameter names. It is for easier identification when printing the values. If not specified, they are numbered. #!Currently only works for single data set fits.
    """
    colors = [
        "green",
        "purple",
        "blue",
        "red",
        "orange",
        "brown",
        "cyan",
        "olive",
        "gold",
        "lime",
    ]
    letras(s=s)  # fontsize of the figure title
    if metodo.lower() == "lsq":
        funcion_de_ajuste = lsq
    elif metodo.lower() == "odr":
        funcion_de_ajuste = Odr
    else:
        print(
            Fore.YELLOW
            + "fit function: The chosen fitting method does not exist. Valid options are: lsq, ODR."
            + Fore.RESET
        )
        return None
    plt.figure(figsize=(16, 12))
    if type(xerr) is type(0.0) or type(xerr) is type(0):
        xerr = np.repeat(xerr, len(x))
    else:
        pass
    if type(yerr) is type(0.0) or type(yerr) is type(0):
        yerr = np.repeat(yerr, len(y))
    else:
        pass
    if nombre_params == []:
        nombre_params = np.arange(1, len(guess) + 1)
    else:
        pass
    if not loglog:
        if type(x[0]) is type([]) or type(x[0]) is type(
            np.array([])
        ):  # checks whether it is a single dataset or not
            [params, stds, r_squareds, chi2rs, tstats, pvals] = [[], [], [], [], [], []]
            for i in range(len(x)):
                xx = x[i]
                yy = y[i]
                yyerr = yerr[i]
                xxerr = xerr[i]
                if type(pesos) is type([]) or type(pesos) is type(np.array([])):
                    pesosi = pesos[i]
                else:
                    pesosi = pesos
                if type(fix) is type([]) or type(fix) is type(np.array([])):
                    fixi = fix[i]
                else:
                    fixi = fix
                intervalo = np.linspace(np.min(xx), np.max(xx), len(xx) * 20)
                if type(func) is type([]):
                    funcion = func[i]
                else:
                    funcion = func
                [param, std, r_squared, chi2r, tstat, pval] = funcion_de_ajuste(
                    funcion,
                    xx,
                    yy,
                    guess[i],
                    yerr=yyerr,
                    ajustar_errores=ajustar_errores,
                    bounds=bounds,
                    pesos=pesosi,
                    fix=fixi,
                    taufac=taufac,
                    maxit=maxit,
                    paramhipothesis=paramhipothesis,
                    sigfig=sigfig,
                )
                if trust_predict is True:
                    [hires_x, pred_upper, pred_lower, trust_upper, trust_lower] = (
                        prediccionesyconfianza(xx, yy, func, param, std, alpha=alpha)
                    )
                    plt.fill_between(
                        x,
                        pred_lower,
                        pred_upper,
                        alpha=0.15,
                        label="Prediction interval",
                    )
                    plt.plot(
                        hires_x,
                        trust_upper,
                        linestyle="--",
                        linewidth=2,
                        label="Confidence bands",
                    )
                    plt.plot(hires_x, trust_lower, linestyle="--", linewidth=2)
                plt.errorbar(
                    xx,
                    yy,
                    xerr=xxerr,
                    yerr=yyerr,
                    ecolor="black",
                    fmt="o",
                    markersize=msize,
                    color=colors[i],
                )
                plt.plot(
                    intervalo,
                    funcion(intervalo, *param),
                    label=leyendas[i],
                    color=colors[i],
                    linewidth=linew,
                )
                params.append(param)
                stds.append(std)
                r_squareds.append(r_squared)
                chi2rs.append(chi2r)
                tstats.append(tstat)
                pvals.append(pval)
        else:
            [params, stds, r_squareds, chi2rs, tstats, pvals] = funcion_de_ajuste(
                func,
                x,
                y,
                guess,
                yerr=yerr,
                ajustar_errores=ajustar_errores,
                bounds=bounds,
                pesos=pesos,
                fix=fix,
                taufac=taufac,
                maxit=maxit,
                paramhipothesis=paramhipothesis,
                sigfig=sigfig,
            )
            if trust_predict is True:
                [hires_x, pred_upper, pred_lower, trust_upper, trust_lower] = (
                    prediccionesyconfianza(x, y, func, params, stds, alpha=alpha)
                )
                plt.fill_between(
                    x,
                    pred_lower,
                    pred_upper,
                    color="green",
                    alpha=0.15,
                    label="Prediction interval",
                )
                plt.plot(
                    hires_x,
                    trust_upper,
                    "g--",
                    linewidth=2,
                    label="Confidence bands",
                )
                plt.plot(hires_x, trust_lower, "g--", linewidth=2)
            intervalo = np.linspace(np.min(x), np.max(x), len(x) * 20)
            plt.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                ecolor="black",
                fmt="o",
                color="red",
                markersize=msize,
                label=leyendas[0],
            )
            plt.plot(
                intervalo,
                func(intervalo, *params),
                linestyle="-",
                color="green",
                linewidth=linew,
            )
    else:
        if type(x[0]) is type([]):  # chequea si es solo un set de datos o si no
            plt.xscale("log")
            plt.yscale("log")
            [params, stds, r_squareds, chi2rs, tstats, pvals] = [[], [], [], [], [], []]
            for i in range(len(x)):
                xx = x[i]
                yy = y[i]
                yyerr = yerr[i]
                xxerr = xerr[i]
                intervalo = np.linspace(np.min(xx), np.max(xx), len(xx) * 20)
                if type(pesos) is type([]) or type(pesos) is type(np.array([])):
                    pesosi = pesos[i]
                else:
                    pesosi = pesos
                if type(fix) is type([]) or type(fix) is type(np.array([])):
                    fixi = fix[i]
                else:
                    fixi = fix
                if type(func) is type([]):
                    funcion = func[i]
                else:
                    funcion = func
                [param, std, r_squared, chi2r, tstat, pval] = funcion_de_ajuste(
                    funcion,
                    xx,
                    yy,
                    guess[i],
                    yerr=yyerr,
                    ajustar_errores=ajustar_errores,
                    bounds=bounds,
                    pesos=pesosi,
                    fix=fixi,
                    taufac=taufac,
                    maxit=maxit,
                    paramhipothesis=paramhipothesis,
                    sigfig=sigfig,
                )
                if trust_predict:
                    [hires_x, pred_upper, pred_lower, trust_upper, trust_lower] = (
                        prediccionesyconfianza(xx, yy, func, param, std, alpha=alpha)
                    )
                    plt.fill_between(
                        x,
                        pred_lower,
                        pred_upper,
                        alpha=0.15,
                        label="Prediction interval",
                    )
                    plt.plot(
                        hires_x,
                        trust_upper,
                        linestyle="--",
                        linewidth=2,
                        label="Confidence bands",
                    )
                    plt.plot(hires_x, trust_lower, linestyle="--", linewidth=2)
                plt.errorbar(
                    xx,
                    yy,
                    xerr=xxerr,
                    yerr=yyerr,
                    ecolor="black",
                    fmt="o",
                    markersize=msize,
                    color=colors[i],
                )
                plt.plot(
                    intervalo,
                    funcion(intervalo, *param),
                    label=leyendas[i],
                    color=colors[i],
                    linewidth=linew,
                )
                params.append(param)
                stds.append(std)
                r_squareds.append(r_squared)
                chi2rs.append(chi2r)
                tstats.append(tstat)
                pvals.append(pval)
        else:
            plt.xscale("log")
            plt.yscale("log")
            [params, stds, r_squareds, chi2rs, tstats, pvals] = funcion_de_ajuste(
                func,
                x,
                y,
                guess,
                yerr=yerr,
                ajustar_errores=ajustar_errores,
                bounds=bounds,
                pesos=pesos,
                fix=fix,
                taufac=taufac,
                maxit=maxit,
                paramhipothesis=paramhipothesis,
                sigfig=sigfig,
            )
            if trust_predict:
                [hires_x, pred_upper, pred_lower, trust_upper, trust_lower] = (
                    prediccionesyconfianza(x, y, func, params, stds, alpha=alpha)
                )
                plt.fill_between(
                    x,
                    pred_lower,
                    pred_upper,
                    color="green",
                    alpha=0.15,
                    label="Prediction interval",
                )
                plt.plot(
                    hires_x,
                    trust_upper,
                    "g--",
                    linewidth=2,
                    label="Confidence bands",
                )
                plt.plot(hires_x, trust_lower, "g--", linewidth=2)
            intervalo = np.linspace(np.min(x), np.max(x), len(x) * 20)
            plt.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                ecolor="black",
                fmt="o",
                color="red",
                markersize=msize,
            )
            plt.plot(
                intervalo,
                func(intervalo, *params),
                linestyle="-",
                color="green",
                linewidth=linew,
            )
    plt.grid(grids, which="both")
    plt.xlabel(xtitle)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.ylabel(ytitle)
    if legend:
        plt.legend()
    if show:
        plt.show()
    if not skipsave:
        plt.savefig(title + ".png", bbox_inches="tight", dpi=200)
        plt.close()
    if type(params) is not type([]) and not silent:
        for i in range(len(params)):
            print(f"Parameter {nombre_params[i]}: {params[i]}", "\u00b1", stds[i])
        print(
            f"R^2: {r_squareds}\nReduced chi squared: {chi2rs}\nt-stats: {tstats}\np-vals: {pvals}\n"
        )
    return [params, stds, r_squareds, chi2rs, tstats, pvals]


def adaptaraodr(func):
    """Reorders a function's parameters to be suitable for use with ODR method"""

    def func2(B, x):
        return func(x, *B)

    return func2


def adjusted_R(x, y, model, popt, unc=1):
    """
    Returns adjusted R squared test for optimal parameters popt calculated
    according to W-MN formula, other forms have different coefficients:
    Wherry/McNemar : (n - 1)/(n - p - 1)
    Wherry : (n - 1)/(n - p)
    Lord : (n + p - 1)/(n - p - 1)
    Stein : (n - 1)/(n - p - 1) * (n - 2)/(n - p - 2) * (n + 1)/n

    """
    # ? Assuming you have a model with ODR argument order f(beta, x). Otherwise if model is of the form f(x, a, b, c..) you could use R = R_squared(y, model(x, *popt), uncertainty=unc)
    R = R_squared(y, model(popt, x), uncertainty=unc)
    n, p = len(y), len(popt)
    coefficient = (n - 1) / (n - p - 1)
    adj = 1 - (1 - R) * coefficient
    return adj, R


def Odr(
    funcion,
    x,
    y,
    guess,
    xerr=1,
    yerr=1,
    fix=None,
    taufac=1,
    maxit=50,
    ajustar_errores=False,
    bounds=(-np.inf, +np.inf),
    pesos=None,
    paramhipothesis=None,
    sigfig=2,
):
    """Computes an orthogonal distance regression fit of a function to data. Extracts some statistical tests. #~All must be checked, except for the chi squared goodness of fit, which was tested and works.

    Args:
        funcion (function): The function to fit the data.
        x (array-like): The independent variable data.
        y (array-like): The dependent variable data.
        guess (array-like): Initial guess for the parameters of the function.
        xerr (array-like, optional): The uncertainties in the independent variable data. Defaults to 1.
        yerr (array-like, optional): The uncertainties in the dependent variable data. Defaults to 1.
        fix (array-like, optional): Fixed values for some parameters of the function. Defaults to None.
        taufac (float, optional): The factor used to calculate the initial step size for the ODR iterations. Defaults to 1.
        maxit (int, optional): The maximum number of iterations for the ODR algorithm. Defaults to 50.
        ajustar_errores (bool, optional): Whether to adjust the errors in the data. Defaults to False.
        bounds (tuple, optional): The lower and upper bounds for the parameters of the function. Defaults to (-np.inf, +np.inf).
        pesos (array-like, optional): Weights for the data points. Defaults to None.
        paramhipothesis (array-like, optional): Hypothesized values for the parameters of the function. Defaults to None.
        sigfig (int, optional): The number of significant figures to round the errors to. Defaults to 2.

    Returns:
        list: A list containing the fitted parameters, their standard deviations, the R-squared value, the reduced chi-squared value, the t-statistics, and the p-values.
    """
    x, y, xerr, yerr = np.array(x), np.array(y), np.array(xerr), np.array(yerr)
    func = adaptaraodr(funcion)
    model = odr.Model(func)
    data = odr.RealData(x, y=y, sx=xerr, sy=yerr, fix=fix)
    myodr = odr.ODR(data, model, beta0=guess, taufac=taufac, maxit=maxit)
    myoutput = myodr.run()
    # myoutput.pprint()
    param = myoutput.beta
    std = myoutput.sd_beta
    if paramhipothesis is None:
        paramhipothesis = np.zeros(len(guess))
    try:
        for i in range(len(std)):
            std[i] = round(
                std[i], -int(math.floor(math.log10(abs(std[i])))) + sigfig - 1
            )  # ? rounds error to sigfig significant figures
            param[i] = round(
                param[i], decimales(std[i])
            )  # rounds parameter to the same number of decimal places as the error. #! might not be working correctly
    except:  # noqa: E722
        pass
    r_squared = R_squared(y, funcion(x, *param), uncertainty=yerr)
    if yerr.all() != 0:  #! must be checked
        chi2 = np.sum((y - funcion(x, *param)) ** 2 / yerr**2)
        chi2r = chi2 / (len(y) - len(guess))
    else:
        chi2r = (
            Fore.YELLOW
            + "odr function: reduced chi squared was not computed because error in y data is zero or not given"
            + Fore.RESET
        )
    t_stat = (
        param - np.array(paramhipothesis)
    ) / std  # t statistic for the slope parameter
    p_val = stats.t.sf(np.abs(t_stat), len(x) - len(guess)) * 2
    return [param, std, r_squared, chi2r, t_stat, p_val]


def propagador(func, variables, errores, printt=False, sigfig=2, norm2=True):
    """Calculates the absolute error of a function of the form z=f(*variables), where each variable is associated with an absolute error in errors. It does not work with numpy functions. (Instead of using np.sqrt, you can use sqrt, the sympy function)

    Args:
        func (function): The function for which the error is to be propagated.
        variables (list): The point at which the function is to be evaluated.
        errors (list): The absolute errors of the point.
        printt (bool, optional): Prints the propagated value with its error. Defaults to False.
        norm2 (bool, optional): If True, calculates the error as the L2 norm of the gradient of the function. If False, uses the L1 norm. Defaults to True.

    Returns:
        list: A list containing the mean value and its absolute error of the function at the given point.
    """
    valor_representativo = func(*variables)
    gradiente_abs = []
    for i in range(len(variables)):

        def parcial(x):
            variabless = variables.copy()
            x = Symbol("x")  # noqa: F405
            variabless[i] = x

            def f(x):
                return func(*variabless)

            derivada = diff(f(x), x)  # noqa: F405
            return derivada.evalf(subs={x: variables[i]})

        gradiente_abs.append(parcial(variables[i]))
    gradiente_abs = np.abs(np.array(gradiente_abs))
    errores = np.array(errores)
    if norm2:
        error = float(
            np.sqrt(np.sum(np.dot(gradiente_abs, np.transpose(errores)) ** 2))
        )
    else:
        error = float(np.sum(np.dot(gradiente_abs, np.transpose(errores))))
    error = round(error, -int(math.floor(math.log10(abs(error)))) + sigfig - 1)
    # valor_representativo=round(valor_representativo, decimales(error)) #! not working
    if printt:
        print(
            Fore.GREEN + "\nValue of the function:",
            float(valor_representativo),
            "\u00b1",
            float(error),
            "\n" + Fore.RESET,
        )
    return [float(valor_representativo), float(error)]


def ordenar(lista_principal, lista_adicional):
    """
    Sorts the numbers in the main list, lista_principal, in ascending order, while maintaining the relationship between indices in the secondary list, lista_adicional.

    Args:
        lista_principal (list): The main list to be sorted.
        lista_adicional (list): The additional list that should be sorted along with the main list.

    Returns:
        list: A list containing the sorted main list and the sorted additional list.
    """
    lista_principal, lista_adicional = (
        list(t) for t in zip(*sorted(zip(lista_principal, lista_adicional)))
    )
    return [lista_principal, lista_adicional]


def propagar(func, x, ex, otrasvariables, otroserrores, sigfig=2):
    """Calculate x' = f(x) with its error (a change of variable)

    Args:
        func (function): Function to propagate
        x (list): Initial values
        ex (list): Errors of the initial values
        otrasvariables (list): Other parameters that may be in the function
        otroserrores (list): Errors of the other parameters

    Returns:
        list: List with the two lists of x' and its error.
    """
    newx = []
    newex = []
    for i in range(len(x)):
        p = propagador(
            func, [x[i], *otrasvariables], [ex[i], *otroserrores], sigfig=sigfig
        )
        newx.append(p[0])
        newex.append(p[1])
    return [newx, newex]


# ? Useful functions for fitting


def lineal(x, a, b):
    return a * x + b


def lineal2(x, a):
    return a * x


def cuadratica(x, a, b, c):
    return a * x**2 + b * x + c


def exponencial(x, a, b, c):
    return a * np.e ** (b * x + c)


def seno(x, a, b, c):
    return a * sin(b * x + c)  # noqa: F405


def seno_absoluto(x, a, b, c):
    return a * abs(sin(b * x + c))  # noqa: F405


def gaussiana(x, a, b, c):
    return a * np.e ** (-((x - b) ** 2) / (2 * c**2))


def logaritmo(x, a, b):
    return a * log(b * x)  # noqa: F405
