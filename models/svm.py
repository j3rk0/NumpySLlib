import numpy as np
from multiprocessing import Pool


class _BinaryLinearSVM:
    def __init__(self, max_epochs=1000, learn_rate=1.):
        self.W = None
        self.max_epochs = max_epochs
        self.learn_rate = learn_rate
        return

    def fit(self, data, labels):
        n_sample, n_features = data.shape
        self.W = np.zeros(n_features + 1)
        # embed bias
        data = np.pad(data, ((0, 0), (0, 1)), constant_values=1)
        # hyperplane encoding
        new_lab = np.ones(labels.shape[0])
        new_lab[labels == 0] = -1

        ret = []  # summary of training
        sum_err = 0
        for i in range(self.max_epochs):
            mean_err = 0
            reg = 1 / (i + 1)  # regularization decrease with epochs
            for j in range(n_sample):
                delta = 0
                predicted = data[j, :] @ self.W
                if predicted * new_lab[j] < 1:  # misclassified
                    delta = (new_lab[j] * data[j, :])
                    mean_err += 1
                self.W += self.learn_rate * (delta - 2 * reg * self.W)
            sum_err += (mean_err / n_sample)
            ret.append([i, (sum_err / (i + 1))])
        return np.array(ret)  # return summary

    def query_svm(self, x):
        return int(x @ self.W > 0)

    def predict(self, data):
        ret = []
        data = np.pad(data, ((0, 0), (0, 1)), constant_values=1)  # embed bias
        for x in data:
            ret.append(self.query_svm(x))
        return np.array(ret)


class _BinarySVM:

    def __init__(self, epochs=2000, learn_rate=.1, C=1, kernel="rbf", pow=4, gamma=None, deg=2):
        self.curr = None
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.C = C

        k_map = {
            "rbf": self._rbf,
            "poly": self._poly,
            "gauss": self._gauss,
            "exp": self._exp,
            "laplace": self._laplace,
            "anova": self._anova,
            "sig": self._sig
        }

        self.k_name = kernel
        self.kernel = k_map[kernel]
        self.pow = pow
        self.gamma = gamma
        self.deg = deg
        self.svecs = None
        self.alpha = None
        self.svecs_labels = None
        self.n_svecs = 0
        self.intercept = 0
        return

    def fit(self, X, y):
        n_samples, n_features = X.shape

        if self.gamma is None:
            self.gamma = 1 / n_features

        labels = np.ones(n_samples)
        labels[y == 0] = -1

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        P = np.outer(labels, labels) * K
        q = np.ones(n_samples)
        A = np.diag(labels)
        b = np.zeros(n_samples)

        if self.C is None:
            A = np.hstack((A, np.diag(np.ones(n_samples))))
            b = np.hstack((b, np.zeros(n_samples)))
        else:
            tmp1 = np.diag(np.ones(n_samples))
            tmp2 = np.identity(n_samples)
            A = np.hstack((A, tmp1, tmp2))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            b = np.hstack((b, tmp1, tmp2))

        # Lagrange multipliers
        lagr = np.ravel(self._solve_qp(P, q, A, b, n_samples))

        # Support vectors have non zero lagrange multipliers
        sv = lagr > 1e-8
        ind = np.arange(len(lagr))[sv]
        self.alpha = lagr[sv]
        self.svecs = X[sv]
        self.svecs_labels = labels[sv]
        self.n_svecs = self.svecs.shape[0]

        # bias
        self.intercept = 0
        for n in range(len(self.alpha)):
            self.intercept += self.svecs_labels[n]
            self.intercept -= np.sum(self.alpha * self.svecs_labels * K[ind[n], sv])
        self.intercept /= len(self.alpha)

    def predict(self, X):
        ret = []
        for x in X:
            s = 0
            for a, sv_y, sv in zip(self.alpha, self.svecs_labels, self.svecs):
                s += a * sv_y * self.kernel(x, sv)
            ret.append(int(s + self.intercept > 0))
        return np.array(ret)

    def _rbf(self, x1, x2):
        return np.exp(-self.gamma * (np.linalg.norm(x1 - x2) ** 2))

    def _poly(self, x1, x2):
        return (x1 @ x2 + self.gamma) ** self.deg

    def _gauss(self, x1, x2):
        return np.exp(- np.square(np.linalg.norm(x1 - x2)) / (2 * self.gamma ** 2))

    def _exp(self, x1, x2):
        return np.exp(- np.linalg.norm(x1 - x2) / (2 * self.gamma ** 2))

    def _laplace(self, x1, x2):
        return np.exp(- np.linalg.norm(x1 - x2) / self.gamma ** 2)

    def _sig(self, x1, x2):

        return np.tanh(self.pow * np.inner(x1, x2) + self.gamma)

    def _anova(self, x1, x2):
        n = x1.shape[0]
        ret = 0
        for k in range(n):
            ret += np.exp(-self.gamma * np.square(x1[k] - x2[k])) ** self.deg
        return ret

    def _solve_qp(self, P, q, A, b, num_eq):

        num_constraints = b.size
        x = np.zeros(q.size)
        gamma = np.zeros(num_constraints)
        grad_sum = 1
        for t in range(self.epochs):
            violations = A.T.dot(x) - b
            clipped_violations = violations.copy()
            clipped_violations[num_eq:] = np.clip(violations[num_eq:], a_min=None, a_max=0)
            violations = clipped_violations != 0

            # update x
            grad_x = P.dot(x) - q + gamma.dot(A.T) + \
                     self.learn_rate * np.linalg.multi_dot((A[:, violations], A[:, violations].T, x)) - \
                     self.learn_rate * b[violations].dot(A[:, violations].T)
            # grad_x = P.dot(x) - q + gamma.dot(A.T) \
            #          + self.learn_rate * A[:, violations].dot(A[:, violations].T).dot(x) \
            #          - self.learn_rate * b[violations].dot(A[:, violations].T)

            grad_sum += grad_x ** 2
            x -= grad_x / np.sqrt(grad_sum)

            # update gamma
            violations = A.T.dot(x) - b
            gamma += self.learn_rate * violations
            gamma[num_eq:] = np.clip(gamma[num_eq:], a_min=None, a_max=0)

        return x


class SVM:

    def __init__(self, epochs=2000, learn_rate=.1, C=1, kernel="rbf", pow=4, gamma=None, deg=2):
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.C = C
        self.kernel = kernel
        self.pow = pow
        self.gamma = gamma
        self.deg = deg

        self.SVMs = None
        self.n_pred = 0
        self.data = None
        self.labels = None

    def fit(self, data, labels):
        n_class=np.max(labels) + 1
        self.n_pred = n_class
        self.data = data
        self.labels = labels

        # MULTICLASS
        if n_class > 2:
            self.SVMs = Pool(processes=self.n_pred).map(self.fit_a_svm, range(self.n_pred))
        else:  # BINARY
            if self.kernel == "linear":  # BINARY LINEAR
                self.SVMs = [_BinaryLinearSVM(self.epochs, self.learn_rate)]
            else:  # BINARY KERNEL
                self.SVMs = [
                    _BinarySVM(self.epochs, self.learn_rate, self.C, self.kernel, self.pow, self.gamma, self.deg)]
            self.n_pred = 1
            self.SVMs[0].fit(data, labels)

        self.data = None
        self.labels = None

    def fit_a_svm(self, i):
        currlabels = np.array(self.labels == i, dtype=int)
        if self.kernel == "linear":  # MULTICLASS LINEAR
            svm = _BinaryLinearSVM(self.epochs, self.learn_rate)
        else:  # MULTICLASS KERNEL
            svm = _BinarySVM(self.epochs, self.learn_rate, self.C, self.kernel, self.pow, self.gamma, self.deg)
        svm.fit(self.data, currlabels)
        return svm

    def predict(self, data):

        if self.n_pred == 1:
            return self.SVMs[0].predict(data)

        pred = []

        for i in range(self.n_pred):
            pred.append(self.SVMs[i].predict(data))
        pred = np.array(pred)
        ret = []
        for i in range(data.shape[0]):
            ret.append(np.argmax(pred[:, i]))
        return np.array(ret)
