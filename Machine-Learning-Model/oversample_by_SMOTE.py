from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd


def oversample_smote(X, y):
    print('Original dataset shape {}'.format(Counter(y)))
    sm = SMOTE()

    # Should apply SMOTE twice by OVA(one-versus-all).

    y_bad_set = y[y==0]
    y_warning_set = y[y==1]
    y_safe_set = y[y==2]

    X_bad_set = X.loc[y_bad_set.index]
    X_warning_set = X.loc[y_warning_set.index]
    X_safe_set = X.loc[y_safe_set.index]

    # warning_set versus all
    y_wva = y_safe_set.replace(2,0)
    y_wva = pd.concat([y_bad_set, y_warning_set, y_wva], axis=0)
    X_wva = pd.concat([X_bad_set, X_warning_set, X_safe_set], axis=0)
    X_wva_res, y_wva_res = sm.fit_sample(X_wva, y_wva)
    print('Resampled dataset shape {}'.format(Counter(y_wva_res)))

    y_warning = pd.Series(y_wva_res[y_wva_res==1]).sample(n=167769)
    X_warning = pd.DataFrame(X_wva_res[y_wva_res==1], columns=X_warning_set.columns).loc[y_warning.index]

    y_bva = y_safe_set.replace(2,1)
    y_bva = pd.concat([y_bad_set, y_warning, y_bva], axis=0)
    X_bva = pd.concat([X_bad_set, X_warning, X_safe_set], axis=0)
    X_bva_res, y_bva_res = sm.fit_sample(X_bva, y_bva)
    print('Resampled dataset shape {}'.format(Counter(y_bva_res)))

    y_bad = pd.Series(y_bva_res[y_bva_res==0]).sample(n=167769)
    X_bad = pd.DataFrame(X_bva_res[y_bva_res==0], columns=X_bad_set.columns).loc[y_bad.index]

    X_return = pd.concat([X_bad, X_warning, X_safe_set], axis=0)
    y_return = pd.concat([y_bad, y_warning, y_safe_set], axis=0)
    print('Resampled dataset shape {}'.format(Counter(y_return)))
    return X_return, y_return
