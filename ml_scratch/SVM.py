import numpy as np


class SVM:
    def __init__(self, C=1):
        self.alpha = None
        self.C = C
        self.X_s = None

    def fit(self, X, y):
        n, m = X.shape
        tol = 1e-3
        num_iterations = int(1e5)
        alpha = np.zeros(n)

        for _ in range(num_iterations):
            i, j = np.random.choice(np.arange(n), 2, replace = False)

            x1 = X[i, :]
            x2 = X[j, :]
            y1 = y[i]
            y2 = y[j]
            a1_old = alpha[i]
            a2_old = alpha[j]

            if y1 == y2:
                L = max(0, a2_old + a1_old - self.C)
                H = min(self.C, a2_old + a1_old)
            else:
                L = max(0, a2_old - a1_old)
                H = min(self.C, self.C + a2_old - a1_old)

            nu = 2 * kernel(x1, x2)- kernel(x1, x1) - kernel(x2, x2)
            nu = nu[0,0]

            E1 = f(alpha, self.C, X, y, x1) - y1
            E2 = f(alpha, self.C, X, y, x2) - y2

            a2_new = a2_old - y2*(E1-E2)/nu

            if a2_new >= H:
                a2_new = H
            if a2_new <= L:
                a2_new = L

            s = y1*y2

            a1_new = a1_old + s*(a2_old - a2_new)

            alpha[i] = a1_new
            alpha[j] = a2_new

    def predict(self, X):
        pass


def kernel(x,z):
    if len(x.shape)==1:
        x = x[None, :]
    if len(z.shape)==1:
        z = z[None, :]
    sigma = 0.5
    dist = np.linalg.norm(x[:, None, :] - z[None, :, :], axis=-1)
    return np.exp(-dist**2 /(2*sigma**2))


def find_b(alpha_s, C, X_s, y_s):
    k_s = kernel(X_s, X_s)
    ayk_term = alpha_s[None,:] * y_s[None,:] * k_s
    diff = y_s - np.sum(ayk_term, axis=1)

    if diff.size > 0:
        b = np.mean(diff)
    else:
        b = 0
    return b

def f(alpha, C, X, y, z, tol=1e-3):
    support_mask = (alpha > tol)*(alpha < (C-tol))

    if sum(support_mask) == 0:
        return 0

    X_s = X[support_mask, :]
    K_s = kernel(X_s, z) # n x 1 vector
    alpha_s = alpha[support_mask]
    y_s = y[support_mask]

    ayk_term = alpha_s * y_s * K_s[:,0]
    b = find_b(alpha_s, C, X_s, y_s)

    return sum(ayk_term) + b
