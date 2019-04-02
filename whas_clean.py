import numpy as np
from sksurv.nonparametric import kaplan_meier_estimator
import matplotlib.pyplot as plt
import seaborn as sns

def groups(row):
    if row['dstat'] == 0 and row['fstat'] == 0:
        val = 3
    elif row['dstat'] == 0 and row['fstat'] == 1:
        val = 2
    else:
        val = 1
    return val

def penalize(row):
    if row['group'] == 2:
        if row['after_dis'] > 1164 and row['after_dis'] <= 1333:
            return round(np.random.normal(1164,100)) + row['los']
        elif row['after_dis'] > 1921 and row['after_dis'] <= 2116:
            return round(np.random.normal(1921,100)) + row['los']
        elif row['after_dis'] > 2116:
            return round(np.random.normal(2011,100)) + row['los']
        else:
            return row['lenfol']
    else:
        return row['lenfol']

def display_km(dic,field,df):
    for k in dic.keys():
        mask = df[field] == k
        ti, surv_prob =  kaplan_meier_estimator(
            df["fstat"][mask].values.astype("bool"),
            df["time"][mask])
        plt.step(ti, surv_prob, where="post",
             label="%s = %s (n = %d)" % (field, dic[k], mask.sum()))
    plt.ylabel("est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$")
    plt.legend(loc="best")


def age_group(row):
    if row['age'] > 30 and row['age'] <= 50:
        return  1
    elif row['age'] > 50 and row['age'] <= 70:
        return  2
    elif row['age'] > 70 and row['age'] <= 90:
        return 3
    else:
        return 4

def hr_level(row):
    if row['hr'] <= 50:
        return  1
    elif row['hr'] > 50 and row['hr'] <= 60:
        return  2
    elif row['hr'] > 61 and row['age'] <= 85:
        return 3
    else:
        return 4

def bmi_level(row):
    if row['bmi'] <= 18.5:
        return  1
    elif row['bmi'] > 18.5 and row['bmi'] <= 25.:
        return  2
    elif row['bmi'] > 25. and row['bmi'] <= 30:
        return 3
    else:
        return 4
