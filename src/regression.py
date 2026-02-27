import numpy as np

def run_ols(X_vars, y, df):

    X = df[X_vars].values
    X = np.hstack([np.ones((X.shape[0],1)), X])

    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    y_hat = X @ beta_hat

    ss_total = np.sum((y - np.mean(y))**2)
    ss_resid = np.sum((y - y_hat)**2)

    r2 = 1 - ss_resid/ss_total

    return beta_hat, r2, y_hat
