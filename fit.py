import pandas as pd
import numpy as np
from sympy import *
from scipy import optimize, odr,stats
import matplotlib.pyplot as plt
import math
import matplotlib

#? funcionara esto??

def readcsv(dataset): 
    data = pd.read_csv(dataset)
    columnas=[]
    for columna in range(len(data.columns)):
        x = list(data.iloc[:, columna])
        for i in range(len(x)):
            x[i] = float(x[i])
        columnas.append(x)
    return columnas

def decimales(x):  # esta es medio rara. la idea es que devuelve las cifras decimales (en realidad no) para despues ajustar el error a una cifra significativa y aproximar el valor representativo a tantos decimales como tenga el error.
    try:
        if abs(x) < 1 and x != 0:
            s = 1-int(math.log10(abs(x)))
        elif x != 0:
            s = -int(math.log10(abs(x)))
        else:
            s = -1
    except OverflowError:
        print('Error al redondear. OverflowError en x=', x)
    return s

def plottearunafuncion(funcion, x, xtitle='x', ytitle='f(x)',title='grafico', grids=True,color='red', linew=4):
    matplotlib.rc('font', size=20)          # controls default text sizes
    matplotlib.rc('axes', titlesize=30)     # fontsize of the axes title
    matplotlib.rc('axes', labelsize=30)    # fontsize of the x and y labels
    matplotlib.rc('xtick', labelsize=24)    # fontsize of the tick labels
    matplotlib.rc('ytick', labelsize=24)    # fontsize of the tick labels
    matplotlib.rc('legend', fontsize=24)       # legend fontsize
    matplotlib.rc('figure', titlesize=18)  # fontsize of the figure title
    plt.figure(figsize=(16,12))
    y=[]
    for i in x:
        y.append(funcion(i))
    plt.plot(x, y, color=color, linewidth=linew)
    plt.grid(grids)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.savefig(title+'.png',bbox_inches='tight', dpi=200)
    return 0
def R_squared(observed, predicted, uncertainty=1):
    """ Returns R square measure of goodness of fit for predicted model. """
    weight = 1./uncertainty
    return 1. - (np.var((observed - predicted)*weight) / np.var(observed*weight))

def lsq(function, x, y, guess, yerr=1, xerr=0, ajustar_errores=False, bounds=(-np.inf, +np.inf), pesos=None,paramhipothesis=None, taufac=1,maxit=50, fix=None, sigfig=2):
    """Calcula los parametros de un ajuste con su error

    Args:
        function (function): funcion que describe el ajuste. IMPORTANTE: se debe definir en funcion de la variable y luego de los parametros, en ese orden.
        x (list): lista de valores de la variable independiente
        y (list): lista de valores de la otra variable
        guess (list): valores en los que empezar a "probar" los parametros. En general se pueden inicializar en cualquier valor, pero podes estar consiguiendo underflow/overflow o cosas raras
        yerr (list): errores absolutos en y. Es decir que la variable en y se mueve entre y-yerr e y+yerr.
        ajustar_errores (bool): si es True, ajusta considerando los errores en y. Si es False, los ignora.
        bounds (tuple): tupla con dos listas o numeros. El primer valor de la tupla es para la cota minima de los parametros, el segundo para la cota superior.
        
    Returns:
        list: devuelve una lista con los parametros, otra con los errores, el R^2 y el chi^2 reducido. 
    """
    #if type(yerr)!=type([]):
    #    yerr=np.repeat(yerr, len(y))
    if paramhipothesis==None:
        paramhipothesis=np.zeros(len(guess))
    x=np.array(x)
    y=np.array(y)
    if ajustar_errores==True:
        param, error = optimize.curve_fit(function, x, y, guess, sigma=yerr, absolute_sigma=True, bounds=bounds)
    elif type(pesos)!=type(None):
        param, error = optimize.curve_fit(function, x, y, guess, sigma=yerr, absolute_sigma=False, bounds=bounds)
    else:
        param, error = optimize.curve_fit(function, x, y, guess, bounds=bounds)
    y_predict=function(x, *param)
    if ajustar_errores==True:
        r_squared=R_squared(y, y_predict, uncertainty=yerr)
    elif type(pesos)!=type(None):
        r_squared=R_squared(y, y_predict, uncertainty=pesos)
    else:
        r_squared=R_squared(y, y_predict)

    std = np.sqrt(np.diag(error))
    yerr=np.array(yerr)
    if yerr.all()!=0:
        chi2=np.sum( (y - y_predict)**2 / yerr**2 )
        chi2r=chi2/(len(y) - len(guess))
    else:
        chi2r='No se pudo estimar porque el error en y es 0.'
    t_stat = (param - np.array(paramhipothesis)) / std  
    p_val = stats.t.sf(np.abs(t_stat), len(x)-len(guess)) * 2  
    try:
        for i in range(len(std)):
            std[i] = round(std[i],  - int(math.floor(math.log10(abs(std[i]))))+sigfig-1)# aca redondea el error a una cifra significativa
            #param[i] = round(param[i], decimales(std[i]))# redondea el parametro a tantos decimales como tenga el error, !# no esta andando bien
    except:
        pass
    return [param, std, r_squared, chi2r, t_stat, p_val]
def prediccionesyconfianza(x, y, func, param, std, alpha=0.05):
    #Prediccion
    hires_x = np.linspace(min(x), max(x), 10000)
    param,std=np.array(param), np.array(std)
    ypred = func(x,*param)
    noise = np.std(y - ypred)
    predictions = np.array([np.random.normal(ypred,noise) for j in range(10000)])
    pred_lower, pred_upper = np.quantile(predictions, [alpha/2, 1-alpha/2], axis = 0)
    #confianza
    trust_upper = func(hires_x, *(param + std))
    trust_lower = func(hires_x, *(param - std))
    return [hires_x, pred_upper, pred_lower, trust_upper, trust_lower]

def letras(s=1):
    font = {'family' : 'DejaVu Sans',
            'weight' : 'normal',
            'size'   : 22}
    matplotlib.rc('font', size=20*s)          # controls default text sizes
    matplotlib.rc('axes', titlesize=30*s)     # fontsize of the axes title
    matplotlib.rc('axes', labelsize=30*s)    # fontsize of the x and y labels
    matplotlib.rc('xtick', labelsize=24*s)    # fontsize of the tick labels
    matplotlib.rc('ytick', labelsize=24*s)    # fontsize of the tick labels
    matplotlib.rc('legend', fontsize=24*s)       # legend fontsize
    matplotlib.rc('figure', titlesize=18*s)
    
def fit(func, x,y,xerr=0,yerr=0, xtitle='x', ytitle='y', loglog=False, grids=True, linew=4,legend=True,title='grafico', msize=10, xlim=None, ylim=None, 
        leyendas=[''], guess=[],  ajustar_errores=False, bounds=(-np.inf, +np.inf), nombre_params=[], show=False, alpha=0.05, trust_predict=False, skipsave=False, 
        metodo='lsq', pesos=None, fix=None, taufac=1, maxit=50,paramhipothesis=None, silent=False, sigfig=2,s=1):
    """hace un plot de un ajuste por cuadrados minimos y un scatter de los datos dados. Si x e y son listas con varios grupos de datos, 
    superpone los ajustes en un solo grafico y acomoda los colores por su cuenta. devuelve los parametros de los ajustes y algunos tests estadisticos.


    Args:
        func (function): funcion de ajuste. puede ser una lista de funciones.
        x (list): valores de x. puede contener listas de valores de x.
        y (list): valores de y. puede contener listas de valores de y.
        xerr (int/list, optional): error en x. Defaults to 0.
        yerr (int/list, optional): error en y. Defaults to 0.
        xtitle (str, optional): titulo eje x. Defaults to 'x'.
        ytitle (str, optional): titulo eje y. Defaults to 'y'.
        loglog (bool, optional): si es True pone ejes en loglog. Defaults to False.
        grids (bool, optional): cuadricula en el grafico. Defaults to True.
        legend (bool, optional): leyenda en el grafico. Defaults to True.
        title (str, optional): titulo en el grafico. Defaults to 'grafico'.
        msize (int, optional): tamano del marcador. Defaults to 10.
        linew (int, optional): grosor de la linea. Defaults to 4.
        guess (list, optional): semilla de los parametros de ajuste. obligatorio si spline=False. Defaults to []. 
        leyendas (list, optional): lista de leyendas para cada ajuste/spline. Es ignorado si solo se hace un ajuste/spline. Defaults to [''].
        ajustar_errores (bool, optional): considera el error de los parametros en el fitteo (True). Defaults to False.
        trust_predict (bool, optional): muestra intervalos de confianza y prediccion. Defaults to False.
        alpha (float, optional): elige el intervalo de confianza 1-alpha con alpha/2 a cada lado. Defaults to 0.05.
        show (bool, optional): muestra el grafico interactivo en lugar de guardarlo. Defaults to False.
        skipsave (bool, optional): evita guardar el grafico, para que se le pueda agregar mas informacion despues de realizar los ajustes. Defaults to False.
        pesos (list, optional): lista de (listas de) pesos relativos para ajustar. Si no es None setea automaticamente absigma=False. Defaults to None.
        fix (list, optional): lista de 0/1 de igual tamaño que x que fija algunos puntos (0), solo funciona con ODR. Defaults to None.
        taufac (float, optional): float specifying the initial trust region. The initial trust region is equal to taufac times the length of the first computed Gauss-Newton step. Defaults to 1.
        maxit (int, optional): número máximo de iteraciones permitidas para ODR. Defaults to 50.
        metodo (str, optional): metodo de ajuste. Opciones validas: 'lsq', 'odr'. Defaults to 'lsq'.
        #!nombre_params(list, optional): nombre de los parametros. es para que se identifiquen mas facilmente al imprimir los valores. si no se especifica, se los numera. por ahora solo funciona para ajustes de un solo set de datos
    """
    colors=['green', 'purple', 'blue', 'red', 'orange', 'brown', 'cyan', 'olive', 'gold', 'lime']
    letras(s=s)  # fontsize of the figure title
    if metodo.lower()=='lsq':
        funcion_de_ajuste=lsq
    elif metodo.lower()=='odr':
        funcion_de_ajuste=Odr
    else:
        print('EL METODO DE AJUSTE ELEGIDO NO EXISTE. OPCIONES VALIDAS: LSQ, ODR.')
        return None
    #matplotlib.rc('font', **font)
    plt.figure(figsize=(16,12))
    if type(xerr)==type(0.) or type(xerr)==type(0):
        xerr=np.repeat(xerr, len(x))
    else:
        pass
    if type(yerr)==type(0.) or type(yerr)==type(0):
        yerr=np.repeat(yerr, len(y))
    else:
        pass
    if nombre_params==[]:
        nombre_params=np.arange(1,len(guess)+1)
    else:
        pass
    if loglog==False:
        if type(x[0])==type([]) or type(x[0])==type(np.array([])): #chequea si es solo un set de datos o si no
            [params, stds, r_squareds,chi2rs,tstats,pvals]=[[],[],[],[],[],[]]
            for i in range(len(x)):
                xx=x[i]
                yy=y[i]
                yyerr=yerr[i]
                xxerr=xerr[i]
                if type(pesos)==type([]) or type(pesos)==type(np.array([])):
                    pesosi=pesos[i]
                else:
                    pesosi=pesos      
                if type(fix)==type([]) or type(fix)==type(np.array([])):
                    fixi=fix[i]
                else:
                    fixi=fix                
                intervalo=np.linspace(np.min(xx), np.max(xx), len(xx)*20)
                if type(func)==type([]):
                    funcion=func[i]
                else:
                    funcion=func
                [param, std, r_squared,chi2r,tstat,pval]=funcion_de_ajuste(funcion, xx, yy, guess[i], yerr=yyerr, ajustar_errores=ajustar_errores, bounds=bounds, pesos=pesosi,fix=fixi,taufac=taufac,maxit=maxit,paramhipothesis=paramhipothesis, sigfig=sigfig) 
                if trust_predict==True:
                    [hires_x,pred_upper, pred_lower, trust_upper, trust_lower]=prediccionesyconfianza(xx, yy, func, param, std, alpha=alpha)
                    plt.fill_between(x, pred_lower, pred_upper, alpha = 0.15, label='Intervalo de predicción')
                    plt.plot(hires_x,trust_upper, linestyle='--', linewidth=2, label='Bandas de confianza')
                    plt.plot(hires_x,trust_lower, linestyle='--', linewidth=2)
                plt.errorbar(xx,yy,xerr=xxerr,yerr=yyerr, ecolor='black', fmt='o', markersize=msize, color=colors[i])
                plt.plot(intervalo, funcion(intervalo, *param), label=leyendas[i], color=colors[i], linewidth=linew)
                params.append(param)
                stds.append(std)
                r_squareds.append(r_squared)
                chi2rs.append(chi2r)
                tstats.append(tstat)
                pvals.append(pval)
        else:
            [params, stds, r_squareds,chi2rs,tstats,pvals]=funcion_de_ajuste(func, x, y, guess, yerr=yerr, ajustar_errores=ajustar_errores, bounds=bounds, pesos=pesos,fix=fix,taufac=taufac,maxit=maxit,paramhipothesis=paramhipothesis, sigfig=sigfig)
            if trust_predict==True:
                [hires_x,pred_upper, pred_lower, trust_upper, trust_lower]=prediccionesyconfianza(x, y, func, params, stds, alpha=alpha)
                plt.fill_between(x, pred_lower, pred_upper, color = 'green', alpha = 0.15, label='Intervalo de predicción')
                plt.plot(hires_x,trust_upper, 'g--', linewidth=2, label='Bandas de confianza')
                plt.plot(hires_x,trust_lower, 'g--', linewidth=2)
            intervalo=np.linspace(np.min(x), np.max(x), len(x)*20)
            plt.errorbar(x,y,xerr=xerr,yerr=yerr, ecolor='black', fmt='o', color='red', markersize=msize, label=leyendas[0])
            plt.plot(intervalo, func(intervalo, *params), linestyle='-', color='green',  linewidth=linew)
    else:
        if type(x[0])==type([]): #chequea si es solo un set de datos o si no
            plt.xscale('log')
            plt.yscale('log')
            [params, stds, r_squareds,chi2rs,tstats,pvals]=[[],[],[],[],[],[]]
            for i in range(len(x)):
                xx=x[i]
                yy=y[i]
                yyerr=yerr[i]
                xxerr=xerr[i]
                intervalo=np.linspace(np.min(xx), np.max(xx), len(xx)*20)
                if type(pesos)==type([]) or type(pesos)==type(np.array([])):
                    pesosi=pesos[i]
                else:
                    pesosi=pesos   
                if type(fix)==type([]) or type(fix)==type(np.array([])):
                    fixi=fix[i]
                else:
                    fixi=fix  
                if type(func)==type([]):
                    funcion=func[i]
                else:
                    funcion=func
                [param, std, r_squared,chi2r,tstat,pval]=funcion_de_ajuste(funcion, xx, yy, guess[i], yerr=yyerr, ajustar_errores=ajustar_errores, bounds=bounds, pesos=pesosi,fix=fixi,taufac=taufac,maxit=maxit,paramhipothesis=paramhipothesis, sigfig=sigfig) 
                if trust_predict==True:
                    [hires_x,pred_upper, pred_lower, trust_upper, trust_lower]=prediccionesyconfianza(xx, yy, func, param, std, alpha=alpha)
                    plt.fill_between(x, pred_lower, pred_upper, alpha = 0.15, label='Intervalo de predicción')
                    plt.plot(hires_x,trust_upper, linestyle='--', linewidth=2, label='Bandas de confianza')
                    plt.plot(hires_x,trust_lower, linestyle='--', linewidth=2)
                plt.errorbar(xx,yy,xerr=xxerr,yerr=yyerr, ecolor='black', fmt='o', markersize=msize, color=colors[i])
                plt.plot(intervalo, funcion(intervalo, *param), label=leyendas[i], color=colors[i], linewidth=linew)
                params.append(param)
                stds.append(std)
                r_squareds.append(r_squared)
                chi2rs.append(chi2r)
                tstats.append(tstat)
                pvals.append(pval)
        else:
            plt.xscale('log')
            plt.yscale('log')
            [params, stds, r_squareds,chi2rs,tstats,pvals]=funcion_de_ajuste(func, x, y, guess, yerr=yerr, ajustar_errores=ajustar_errores, bounds=bounds, pesos=pesos,fix=fix,taufac=taufac,maxit=maxit,paramhipothesis=paramhipothesis, sigfig=sigfig)
            if trust_predict==True:
                [hires_x,pred_upper, pred_lower, trust_upper, trust_lower]=prediccionesyconfianza(x, y, func, params, stds, alpha=alpha)
                plt.fill_between(x, pred_lower, pred_upper, color = 'green', alpha = 0.15, label='Intervalo de predicción')
                plt.plot(hires_x,trust_upper, 'g--', linewidth=2, label='Bandas de confianza')
                plt.plot(hires_x,trust_lower, 'g--', linewidth=2)
            intervalo=np.linspace(np.min(x), np.max(x), len(x)*20)
            plt.errorbar(x,y,xerr=xerr,yerr=yerr, ecolor='black', fmt='o', color='red', markersize=msize)
            plt.plot(intervalo, func(intervalo, *params), linestyle='-', color='green',  linewidth=linew)
    plt.grid(grids, which="both")
    plt.xlabel(xtitle)
    if xlim!=None:
        plt.xlim(xlim)
    if ylim!=None:
        plt.ylim(ylim)
    plt.ylabel(ytitle)
    if legend==True:
        plt.legend()
    if show==True:
        plt.show()
    if skipsave==False:
        plt.savefig(title+'.png',bbox_inches='tight', dpi=200)
        plt.close()
    if type(params)!=type([]) and silent==False:
        for i in range(len(params)):
            print(f"Parámetro {nombre_params[i]}: {params[i]}", u"\u00B1", stds[i])
        print(f"R^2: {r_squareds}\nChi cuadrado reducido: {chi2rs}\nt-stats: {tstats}\np-vals: {pvals}\n")
    return [params, stds, r_squareds,chi2rs,tstats,pvals]

def adaptaraodr(func):
    def func2(B,x):
        return func(x,*B)
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
    # Assuming you have a model with ODR argument order f(beta, x)
    # otherwise if model is of the form f(x, a, b, c..) you could use
    # R = R_squared(y, model(x, *popt), uncertainty=unc)
    R = R_squared(y, model(popt, x), uncertainty=unc)
    n, p = len(y), len(popt)
    coefficient = (n - 1)/(n - p - 1)
    adj = 1 - (1 - R) * coefficient
    return adj, R
def Odr(funcion,x,y,guess,xerr=1,yerr=1, fix=None,taufac=1,maxit=50, ajustar_errores=False, bounds=(-np.inf, +np.inf), pesos=None,paramhipothesis=None, sigfig=2):
    x,y,xerr,yerr=np.array(x),np.array(y), np.array(xerr), np.array(yerr)
    func=adaptaraodr(funcion)  
    model=odr.Model(func)
    data=odr.RealData(x,y=y,sx=xerr,sy=yerr, fix=fix)
    myodr=odr.ODR(data,model,beta0=guess,taufac=taufac,maxit=maxit)
    myoutput = myodr.run()
    #myoutput.pprint()
    param=myoutput.beta
    std=myoutput.sd_beta
    if paramhipothesis==None:
        paramhipothesis=np.zeros(len(guess))
    try:
        for i in range(len(std)):
            std[i] = round(std[i],  - int(math.floor(math.log10(abs(std[i]))))+sigfig-1)# aca redondea el error a una cifra significativa
            param[i] = round(param[i], decimales(std[i]))# redondea el parametro a tantos decimales como tenga el error
    except:
        pass
    r_squared=R_squared(y,funcion(x,*param), uncertainty=yerr)
    if yerr.all()!=0: #! REVISAR ESTO
        chi2=np.sum( (y - funcion(x,*param))**2 / yerr**2 )
        chi2r=chi2/(len(y) - len(guess))
    else:
        chi2r='No se pudo estimar porque el error en y es 0.'
    t_stat = (param - np.array(paramhipothesis)) / std  # t statistic for the slope parameter
    p_val = stats.t.sf(np.abs(t_stat), len(x)-len(guess)) * 2
    return [param, std, r_squared, chi2r,t_stat,p_val]

def propagador(func, variables, errores, printt=False,sigfig=2, norm2=True):
    """calcula el error absoluto de una funcion del tipo z=f(*variables), cada variable asociada a un error absoluto en errores. No funciona con funciones
    de numpy. (en lugar de usar np.sqrt se puede usar sqrt, la funcion de sympy)

    Args:
        func (function): funcion cuyo error se quiere propagar
        variables (list): punto en el que se quiere evaluar la funcion
        errores (list): errores absolutos del punto
        printt (bool, optional): imprime el valor de la propagacion con su error. Defaults to False.
        norm2 (bool, optional): si es True, calcula el error como la norma 2 del gradiente de la funcion. Si es False, usa la norma 1. Defaults to True.

    Returns:
        list: lista con el valor representativo y su error absoluto de la funcion en el punto dado.
    """
    valor_representativo=func(*variables)
    gradiente_abs=[]
    for i in range(len(variables)):
        def parcial(x):
            variabless=variables.copy()
            x=Symbol('x')
            variabless[i]=x
            def f(x):
                return func(*variabless)
            derivada=diff(f(x), x)
            return derivada.evalf(subs={x:variables[i]})
        gradiente_abs.append(parcial(variables[i]))
    gradiente_abs=np.abs(np.array(gradiente_abs))
    errores=np.array(errores)
    if norm2==True:
        error=float(np.sqrt(np.sum(np.dot(gradiente_abs, np.transpose(errores))**2)))
    else:
        error=float(np.sum(np.dot(gradiente_abs, np.transpose(errores))))
    error=round(error, - int(math.floor(math.log10(abs(error))))+sigfig-1) 
    #valor_representativo=round(valor_representativo, decimales(error)) #! no funciona ok
    if printt==True:
        print('\nValor de la función:',float(valor_representativo), u"\u00B1",float(error), '\n')
    return [float(valor_representativo), float(error)]

def ordenar(lista_principal, lista_adicional):
    #ordena de menor a mayor los numeros en lista_principal, ordenando a la vez una lista secundaria de manera de respetar la relacion entre los indices.
    lista_principal, lista_adicional = (list(t) for t in zip(*sorted(zip(lista_principal, lista_adicional))))
    return [lista_principal, lista_adicional]

def propagar(func, x, ex, otrasvariables, otroserrores,sigfig=2):
    """calcula x'=f(x) con su error (un cambio de variable)

    Args:
        func (function): funcion a propagar
        x (list): valores iniciales
        ex (list): errores de los valores iniciales
        otrasvariables (list): otros parametros que puedan estar en la funcion
        otroserrores (list): errores de los otros parametros

    Returns:
        list: lista con las dos listas de x' y su error.
    """
    newx=[]
    newex=[]
    for i in range (len(x)):
        p=propagador(func, [x[i],*otrasvariables], [ex[i],*otroserrores],sigfig=sigfig)
        newx.append(p[0])
        newex.append(p[1])
    return [newx, newex]
#!FUNCIONES UTILES PARA AJUSTAR:

def lineal(x,a,b):
    return a*x+b
def lineal2(x,a):
    return a*x
def cuadratica(x,a,b,c):
    return a*x**2+b*x+c
def exponencial(x,a,b,c):
    return a*np.e**(b*x+c)
def seno(x,a,b,c):
    return a*sin(b*x+c)
def seno_absoluto(x,a,b,c):
    return a*abs(sin(b*x+c))
def gaussiana(x,a,b,c):
    return a*np.e**(-(x-b)**2/(2*c**2))
def logaritmo(x,a,b):
    return a*log(b*x)


