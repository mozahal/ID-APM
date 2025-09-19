import numpy as np
import numbers
from warnings import warn
import matplotlib.pyplot as plt


def initialize_sigma2(X, Y):
    (N, D) = X.shape
    (M, _) = Y.shape
    diff = X[None, :, :] - Y[:, None, :]
    err = diff ** 2
    return np.sum(err) / (D * M * N)


def gaussian_kernel(X, beta, Y=None):
    if Y is None:
        Y = X
    dist = (X[:, np.newaxis] - Y[np.newaxis, :]) ** 2
    return np.exp(-dist.sum(axis=-1) / (2 * beta ** 2))


class EMRegistration(object):

    def __init__(self, X, Y, sigma2=None, max_iterations=None, tolerance=None, w=None, *args, **kwargs):
        if type(X) is not np.ndarray or X.ndim != 2:
            raise ValueError(
                "The target point cloud (X) must be at a 2D numpy array.")

        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError(
                "The source point cloud (Y) must be a 2D numpy array.")

        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                "Both point clouds need to have the same number of dimensions.")

        if sigma2 is not None and (not isinstance(sigma2, numbers.Number) or sigma2 <= 0):
            raise ValueError(
                "Expected a positive value for sigma2 instead got: {}".format(sigma2))

        if max_iterations is not None and (not isinstance(max_iterations, numbers.Number) or max_iterations < 0):
            raise ValueError(
                "Expected a positive integer for max_iterations instead got: {}".format(max_iterations))
        elif isinstance(max_iterations, numbers.Number) and not isinstance(max_iterations, int):
            warn("Received a non-integer value for max_iterations: {}. Casting to integer.".format(max_iterations))
            max_iterations = int(max_iterations)

        if tolerance is not None and (not isinstance(tolerance, numbers.Number) or tolerance < 0):
            raise ValueError(
                "Expected a positive float for tolerance instead got: {}".format(tolerance))

        if w is not None and (not isinstance(w, numbers.Number) or w < 0 or w >= 1):
            raise ValueError(
                "Expected a value between 0 (inclusive) and 1 (exclusive) for w instead got: {}".format(w))

        self.X = X
        self.Y = Y
        self.TY = Y
        self.sigma2 = initialize_sigma2(X, Y) if sigma2 is None else sigma2
        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape
        self.tolerance = 0.001 if tolerance is None else tolerance
        self.w = 0.0 if w is None else w
        self.max_iterations = 100 if max_iterations is None else max_iterations
        self.iteration = 0
        self.diff = np.inf
        self.q = np.inf
        self.P = np.zeros((self.M, self.N))
        self.Pt1 = np.zeros((self.N,))
        self.P1 = np.zeros((self.M,))
        self.PX = np.zeros((self.M, self.D))
        self.Np = 0

    def register(self, callback=lambda **kwargs: None):
        self.transform_point_cloud()
        while self.iteration < self.max_iterations and self.diff > self.tolerance:
            self.iterate()
            if callable(callback):
                kwargs = {'iteration': self.iteration,
                          'error': self.q, 'X': self.X, 'Y': self.TY}
                callback(**kwargs)

        return self.TY, np.sum(np.dot(self.P, (self.X[:, np.newaxis, :] - self.TY) ** 2)) / (self.M * self.N)

    def get_registration_parameters(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Registration parameters should be defined in child classes.")

    def update_transform(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating transform parameters should be defined in child classes.")

    def transform_point_cloud(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the source point cloud should be defined in child classes.")

    def update_variance(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the Gaussian variance for the mixture model should be defined in child classes.")

    def iterate(self):
        """
        Perform one iteration of the EM algorithm.
        """
        self.expectation()
        self.maximization()
        self.iteration += 1

    def expectation(self):
        """
        Compute the expectation step of the EM algorithm.
        """
        P = np.sum((self.X[None, :, :] - self.TY[:, None, :]) ** 2, axis=2)  # (M, N)
        P = np.exp(-P / (2 * self.sigma2))
        c = (2 * np.pi * self.sigma2) ** (self.D / 2) * self.w / (1. - self.w) * self.M / self.N

        den = np.sum(P, axis=0, keepdims=True)  # (1, N)
        den = np.clip(den, np.finfo(self.X.dtype).eps, None) + c

        t = self.iteration + 1  # 从第1次迭代开始计数（避免log(0)）
        current_drop_prob = self.initial_drop_prob ** (np.log(t + np.e))  # p^(ln(t+e))
        current_save=1-current_drop_prob
        self.P = np.divide(P, den)

        n_retain = int(current_save * self.N)
        n_indices = np.random.choice(self.N, size=n_retain, replace=False)
        n_mask = np.zeros(self.N, dtype=bool)
        n_mask[n_indices] = True

        m_retain = int(current_save * self.M)
        m_indices = np.random.choice(self.M, size=m_retain, replace=False)
        m_mask = np.zeros(self.M, dtype=bool)
        m_mask[m_indices] = True

        self.P[~m_mask, :] = 0  # 未被选中的Y行置零
        self.P[:, ~n_mask] = 0  # 未被选中的X列置零

        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P1)
        self.PX = np.matmul(self.P, self.X)

    def maximization(self):
        """
        Compute the maximization step of the EM algorithm.
        """
        self.update_transform()
        self.transform_point_cloud()
        self.update_variance()


class DeformableRegistration(EMRegistration):
    def __init__(self, X, Y, alpha=None, beta=None, low_rank=False,initial_drop_prob=0.05, num_eig=100, *args, **kwargs):
        super().__init__(X, Y, *args, **kwargs)

        if alpha is not None and (not isinstance(alpha, numbers.Number) or alpha <= 0):
            raise ValueError("Expected a positive value for alpha.")
        if beta is not None and (not isinstance(beta, numbers.Number) or beta <= 0):
            raise ValueError("Expected a positive value for beta.")
        self.initial_drop_prob=initial_drop_prob
        self.alpha = 2 if alpha is None else alpha
        self.beta = 2 if beta is None else beta
        self.W = np.zeros((self.M, self.D))
        self.G = gaussian_kernel(self.Y, self.beta)
        self.low_rank = low_rank
        self.num_eig = num_eig
        self.sigma2_history = []

    def update_transform(self):
        if not self.low_rank:
            A = np.dot(np.diag(self.P1), self.G) + self.alpha * self.sigma2 * np.eye(self.M)
            B = self.PX - np.dot(np.diag(self.P1), self.Y)
            self.W = np.linalg.solve(A, B)

    def transform_point_cloud(self, Y=None):
        if Y is not None:
            G = gaussian_kernel(X=Y, beta=self.beta, Y=self.Y)
            return Y + np.dot(G, self.W)
        else:
            self.TY = self.Y + np.dot(self.G, self.W)

    def update_variance(self):
        qprev = self.sigma2
        self.q = np.inf

        xPx = np.dot(self.Pt1, np.sum(self.X ** 2, axis=1))
        yPy = np.dot(self.P1, np.sum(self.TY ** 2, axis=1))
        trPXY = np.sum(self.TY * self.PX)
        print('NP',self.Np)
        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)
        if self.sigma2 <= 0:
            self.sigma2 = 1e-10  # 避免方差为零

        self.diff = np.abs(self.sigma2 - qprev)
        # print('diff:',self.sigma2 - qprev)
        # if self.sigma2 - qprev>0:
        print('diff:', self.sigma2 - qprev,qprev/(qprev-self.sigma2))
        self.sigma2_history.append(self.sigma2)
        if self.sigma2 < qprev:
            self.diff = np.abs(self.sigma2 - qprev)
        else:
            if np.random.rand() < self.phla:
                self.diff = np.abs(self.sigma2 - qprev)
            else:
                self.sigma2 = qprev
                self.diff = np.abs(self.sigma2 - qprev)

    def get_registration_parameters(self):
        return self.G, self.W
def match_points(X,Y):
    custom_registration = DeformableRegistration(X=X, Y=Y)
    transformed_Y_custom, _ = custom_registration.register()

    # 创建所有可能的点对及其距离
    all_distances = []
    for i, x in enumerate(X):
        for j, y in enumerate(transformed_Y_custom):
            distance = np.linalg.norm(x - y)
            all_distances.append((distance, i, j))

    # 按照距离升序排序
    all_distances.sort()

    # 创建结果匹配数组
    matches = []
    matched_targets = set()  # 已匹配的目标点索引
    matched_sources = set()  # 已匹配的源点索引

    # 选择满足一对一的匹配
    for distance, target_index, source_index in all_distances:
        if target_index not in matched_targets and source_index not in matched_sources:
            matches.append((target_index, source_index))
            matched_targets.add(target_index)
            matched_sources.add(source_index)

    # 输出匹配结果
    matches_array = np.array(matches)
    return matches_array

if __name__=='__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    # from custom_cpd import DeformableRegistration  # 确保 custom_cpd.py 在同一目录

    # 创建示例点云
    np.random.seed(42)
    X = np.random.rand(10, 2)  # 目标点云
    Y = np.random.rand(10, 2) + np.array([0.5, 0])  # 源点云
    print(X)
    # 使用自定义 CPD 进行配准
    custom_registration = DeformableRegistration(X=X, Y=Y, alpha=2, beta=2)
    transformed_Y_custom, _ = custom_registration.register()

    # 创建所有可能的点对及其距离
    all_distances = []
    for i, x in enumerate(X):
        for j, y in enumerate(transformed_Y_custom):
            distance = np.linalg.norm(x - y)
            all_distances.append((distance, i, j))

    # 按照距离升序排序
    all_distances.sort()

    # 创建结果匹配数组
    matches = []
    matched_targets = set()  # 已匹配的目标点索引
    matched_sources = set()  # 已匹配的源点索引

    # 选择满足一对一的匹配
    for distance, target_index, source_index in all_distances:
        if target_index not in matched_targets and source_index not in matched_sources:
            matches.append((target_index, source_index))
            matched_targets.add(target_index)
            matched_sources.add(source_index)

    # 输出匹配结果
    matches_array = np.array(matches)
    print("Matched pairs (Target Index, Source Index):")
    print(matches_array,matches_array[1],matches_array[0,1])

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c='red', label='Target (Custom CPD)', alpha=0.5)
    plt.scatter(transformed_Y_custom[:, 0], transformed_Y_custom[:, 1], c='blue', label='Transformed (Custom CPD)',
                alpha=0.5)

    # 绘制连接线
    for target_index, source_index in matches:
        plt.plot([X[target_index, 0], transformed_Y_custom[source_index, 0]],
                 [X[target_index, 1], transformed_Y_custom[source_index, 1]], 'k--', lw=0.7)

    plt.title('Point Cloud Registration with Unique, Sorted Matches')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid()
    plt.show()
