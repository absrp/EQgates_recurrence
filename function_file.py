import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import linear_model
from scipy.optimize import curve_fit
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, f1_score, confusion_matrix
from scipy import stats
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats import lognorm
from scipy.stats import bootstrap
from sklearn.utils import resample


def bootstrap_errors(xfeature, BUbin, classweightb, ax, minx, maxx, length, n_iterations=10000):

    """
    This function calculates errors for the logistic fits based on bootstrapping

    """    

    logistic_reg = []
    np.random.seed(42) # for reproducibility

    # Bootstrap resampling and fitting logistic regression
    for i in range(n_iterations):
        # Resample with replacement
        resampled_xfeature, resampled_BUbin = resample(xfeature, BUbin, replace=True)

    # dealing with small number of unbreached gaps
        if len(resampled_BUbin.unique()) < 2:
            i = i-1 # update i to re-run this iteration
            continue  

        probname = sklearn.linear_model.LogisticRegression(penalty='none', class_weight=classweightb).fit(
            np.atleast_2d(resampled_xfeature).T, resampled_BUbin
        )

        x = np.atleast_2d(np.linspace(minx, maxx, 10000)).T
        logistic_reg.append(probname.predict_proba(x)[:, 1])

    
    percentiles_2_5 = np.percentile(logistic_reg, 2.5, axis=0) 
    percentiles_97_5 = np.percentile(logistic_reg, 97.5, axis=0) 
    
    xi = np.linspace(minx, maxx, 10000)

    if length==True:
        ax.fill_between(10**xi, percentiles_2_5, percentiles_97_5, color='slategray', alpha=0.2)
    else:
        ax.fill_between(xi, percentiles_2_5, percentiles_97_5, color='slategray', alpha=0.2)
    return logistic_reg


def build_logistic_regression(
        grouped,
        groupid, 
        type, 
        length_or_angle, 
        class_weightb, 
        axesid,
        minx,
        maxx,
        colorline,
        xlabel,
        ptsize
        ):
    
    """
    This function builds logistic regressions for earthquake gates, based on the groups mapped as breached and unbreached.

    """   
    
    EQgate = grouped.get_group(groupid)

    if type  == 'restraining':
        grouped_type = EQgate.groupby(EQgate["Type (releasing or restraining)"])
        group = grouped_type.get_group('restraining')

    elif type  == 'releasing':
        grouped_type = EQgate.groupby(EQgate["Type (releasing or restraining)"])
        group = grouped_type.get_group('releasing')

    elif type == 'single':
        grouped_type = EQgate.groupby(EQgate["Type (single or double)"])
        group = grouped_type.get_group('single')


    elif type == 'double':
        grouped_type = EQgate.groupby(EQgate["Type (single or double)"])
        group = grouped_type.get_group('double')

    else:
        group = EQgate

    BUbin = pd.get_dummies(group['Breached or unbreached'])
    BUbin = BUbin['unbreached']
    
    if length_or_angle == 'length':
        group['logfeature'] = np.log10(group['Length (m) or angle (deg)'].astype('float')) 
        xfeature = group['logfeature']
        minx = np.log10(minx)
        maxx = np.log10(maxx)

    elif length_or_angle == 'angle':
        group['logfeature'] = np.log10(group['Length (m) or angle (deg)'])
        xfeature = group['Length (m) or angle (deg)']

    else: 
        raise Exception("Feature must include a length or an angle")

    palette = {'breached': 'teal', 'unbreached': 'darkorange'}
    
    if max(group['Length (m) or angle (deg)'])>90:
            sns.swarmplot(
            data=group,
            x='Length (m) or angle (deg)',
            y='Breached or unbreached',
            ax=axesid,size=ptsize,
            hue="Breached or unbreached",
            palette=palette,alpha=0.7
        )
    else:  
        sns.swarmplot(
            data=group,
            x='Length (m) or angle (deg)',
            y='Breached or unbreached',
            ax=axesid,size=ptsize,
            hue="Breached or unbreached",
            palette=palette,alpha=0.7
        )

    if max(group['Length (m) or angle (deg)'])>90:
        bootstrap_errors(xfeature, BUbin, class_weightb, axesid, minx, maxx, True)
    else: 
        bootstrap_errors(xfeature, BUbin, class_weightb, axesid, minx, maxx, False)

    probname = sklearn.linear_model.LogisticRegression(penalty='none',class_weight=class_weightb).fit(np.atleast_2d(xfeature).T,BUbin)

    # tests
    acc = accuracy_score(BUbin, probname.predict(np.atleast_2d(xfeature).T))
    pre = precision_score(BUbin, probname.predict(np.atleast_2d(xfeature).T))
    f1 = f1_score(BUbin, probname.predict(np.atleast_2d(xfeature).T))
    roc =  roc_auc_score(BUbin, probname.predict_proba(np.atleast_2d(xfeature).T)[:,1])
    confusion_matrixi = confusion_matrix(BUbin, probname.predict(np.atleast_2d(xfeature).T))

    x = np.atleast_2d(np.linspace(minx, maxx, 10000)).T

    if max(group['Length (m) or angle (deg)'])>90:
        axesid.plot(10**x,probname.predict_proba(x)[:,1],color = colorline)
        axesid.text(10**x[-10], -0.1, f'ROC={roc:.2f}', ha='right', va='top',fontsize=14)
        axesid.set_xscale('log')
    else:
        axesid.text(x[-10], -0.1, f'ROC={roc:.2f}', ha='right', va='top',fontsize=14)
        axesid.plot(x,probname.predict_proba(x)[:,1],color = colorline)

    axesid.set_ylabel('Passing probability')
    axesid.set_xlabel(xlabel)
    axesid.set_yticklabels(["Breached", "Unbreached"],rotation=90,va='center')
    axesid.get_legend().remove()

    return probname, acc, pre, f1, roc, confusion_matrixi, BUbin, xfeature 

def power_law(x, a, b):
     return b*np.log10(x)+a
