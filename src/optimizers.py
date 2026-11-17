"""
optimizers.py - Các thuật toán tối ưu cho bài toán chọn feature.

Quy ước chung:
  * Nghiệm = mask nhị phân (ndarray bool) độ dài n_features, True = giữ feature đó.
  * Fitness cần TỐI THIỂU HÓA (nhỏ hơn = tốt hơn).
  * Mỗi thuật toán chỉ cần viết hàm `search(evaluate, n_features, rng)`:
      - gọi `evaluate(mask)` để lấy fitness của một mask (mask có thể là bool hoặc 0/1)
      - cứ việc lặp; khi hết ngân sách hoặc hội tụ, `evaluate` sẽ tự ngắt thuật toán
  * Đếm số lần gọi, cache, ghi lịch sử hội tụ, đo thời gian fitness, dừng theo ngân sách / dừng sớm
    đều do lớp Tracker + BaseOptimizer.run lo -> mọi thuật toán được so sánh công bằng.

Hiện có: SA, GA, PSO, ACO (và RS - random search, dùng làm mốc nếu muốn).
Thêm thuật toán mới (ví dụ GWO, WOA): kế thừa BaseOptimizer, viết search(), đăng ký vào OPTIMIZERS,
rồi thêm tên và tham số vào CONFIG["optimizers"]. Thuật toán dùng biến liên tục tự chuyển sang mask
nhị phân (sigmoid, ngưỡng...) trước khi gọi evaluate.
"""
import time

import numpy as np

# Khi budget_on="unique", nếu thuật toán kẹt (liên tục sinh lại mask đã có) thì dừng sau
# STALL_FACTOR * max_evals lần gọi để tránh vòng lặp vô hạn.
STALL_FACTOR = 50


class StopSearch(Exception):
    """Ném ra bởi Tracker để ngắt thuật toán (hết ngân sách / hội tụ / kẹt)."""


class Tracker:
    """Bọc hàm fitness: cache, đếm, nhớ nghiệm tốt nhất, ghi lịch sử, đo thời gian, kiểm tra điều kiện dừng.

    Cache an toàn vì fitness là xác định (cùng mask -> cùng giá trị), nên cache KHÔNG làm
    thay đổi quỹ đạo tìm kiếm, chỉ làm giảm thời gian. Mỗi lần chạy có một cache riêng.

    Thời gian fitness được ghi hai cách:
      fit_time_actual  : tổng thời gian thực sự tính fitness (chỉ tính các lần cache miss)
      fit_time_nocache : tổng thời gian nếu KHÔNG có cache. Mỗi lần gọi (kể cả cache hit) được cộng
                         thời gian đo được ở lần tính đầu tiên của chính mask đó.
    """

    def __init__(self, fitness_fn, max_evals, patience=None, min_delta=0.0,
                 budget_on="calls", use_cache=True):
        self.fitness_fn = fitness_fn
        self.max_evals = max_evals
        self.patience = patience
        self.min_delta = min_delta
        self.budget_on = budget_on
        self.use_cache = use_cache

        self.cache = {}                # mask (bytes) -> (fitness, giây tính lần đầu)
        self.n_calls = 0               # số lần thuật toán gọi fitness
        self.n_fits = 0                # số lần thực sự tính fitness (cache miss)
        self.fit_time_actual = 0.0
        self.fit_time_nocache = 0.0

        self.best_fitness = np.inf
        self.best_mask = None
        self.history = []              # best-so-far sau mỗi lần gọi
        self.times = []                # giây kể từ lúc bắt đầu, sau mỗi lần gọi
        self.since_improve = 0         # số lần gọi liên tiếp không cải thiện đáng kể
        self.stop_reason = "budget"
        self.t_start = time.perf_counter()

    def _check_stop(self):
        used = self.n_calls if self.budget_on == "calls" else self.n_fits
        if used >= self.max_evals:
            self.stop_reason = "budget"
            raise StopSearch
        if self.budget_on == "unique" and self.n_calls >= STALL_FACTOR * self.max_evals:
            self.stop_reason = "stalled"
            raise StopSearch
        if self.patience is not None and self.since_improve >= self.patience:
            self.stop_reason = "converged"
            raise StopSearch

    def __call__(self, mask):
        self._check_stop()
        mask = np.asarray(mask, dtype=bool)
        key = mask.tobytes()

        if self.use_cache and key in self.cache:
            f, secs = self.cache[key]                       # cache hit: không tính lại
        else:
            t = time.perf_counter()
            f = self.fitness_fn(mask)
            secs = time.perf_counter() - t
            self.n_fits += 1
            self.fit_time_actual += secs
            if self.use_cache:
                self.cache[key] = (f, secs)
        self.n_calls += 1
        self.fit_time_nocache += secs

        if f < self.best_fitness - self.min_delta:
            self.since_improve = 0
        else:
            self.since_improve += 1
        if f < self.best_fitness:
            self.best_fitness = f
            self.best_mask = mask.copy()
        self.history.append(self.best_fitness)
        self.times.append(time.perf_counter() - self.t_start)
        return f


class BaseOptimizer:
    """Lớp cơ sở. Lớp con chỉ cần cài đặt search()."""

    def search(self, evaluate, n_features, rng):
        raise NotImplementedError

    def run(self, fitness_fn, n_features, seed, max_evals, patience=None, min_delta=0.0,
            budget_on="calls", use_cache=True):
        rng = np.random.default_rng(seed)
        tracker = Tracker(fitness_fn, max_evals, patience, min_delta, budget_on, use_cache)
        try:
            self.search(tracker, n_features, rng)
        except StopSearch:
            pass
        return {
            "best_mask": tracker.best_mask,
            "best_fitness": tracker.best_fitness,
            "history": tracker.history,
            "times": tracker.times,
            "n_calls": tracker.n_calls,
            "n_fits": tracker.n_fits,
            "fit_time_actual": tracker.fit_time_actual,
            "fit_time_nocache": tracker.fit_time_nocache,
            "stop_reason": tracker.stop_reason,
        }


class RandomSearch(BaseOptimizer):
    """Mỗi lần sinh một mask ngẫu nhiên (mỗi bit = 1 với xác suất p_select). Dùng làm mốc so sánh."""

    def __init__(self, p_select=0.5):
        self.p_select = p_select

    def search(self, evaluate, n_features, rng):
        while True:                                   # Tracker sẽ ngắt khi cần
            evaluate(rng.random(n_features) < self.p_select)


class GeneticAlgorithm(BaseOptimizer):
    """GA nhị phân: chọn lọc tournament, lai ghép đồng nhất, đột biến lật bit, giữ elite."""

    def __init__(self, pop_size=30, crossover_rate=0.9, mutation_rate=None,
                 tournament_size=3, elite_size=2, init_prob=0.5):
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate            # None -> 1 / n_features
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.init_prob = init_prob

    def _tournament(self, pop, fit, rng):
        idx = rng.choice(len(pop), self.tournament_size, replace=False)
        return pop[idx[np.argmin(fit[idx])]]

    def search(self, evaluate, n_features, rng):
        mut = self.mutation_rate if self.mutation_rate is not None else 1.0 / n_features

        # Khởi tạo quần thể ngẫu nhiên và đánh giá
        pop = rng.random((self.pop_size, n_features)) < self.init_prob
        fit = np.array([evaluate(ind) for ind in pop])

        while True:
            # 1) Giữ lại elite (giữ nguyên fitness, không gọi lại evaluate)
            order = np.argsort(fit)[:self.elite_size]
            elites, elite_fit = pop[order], fit[order]

            # 2) Sinh con bằng chọn lọc + lai ghép + đột biến
            children = []
            while len(children) < self.pop_size - self.elite_size:
                p1 = self._tournament(pop, fit, rng)
                p2 = self._tournament(pop, fit, rng)
                if rng.random() < self.crossover_rate:
                    take_p1 = rng.random(n_features) < 0.5
                    c1, c2 = np.where(take_p1, p1, p2), np.where(take_p1, p2, p1)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                for c in (c1, c2):
                    children.append(c ^ (rng.random(n_features) < mut))   # lật bit
            children = np.array(children[:self.pop_size - self.elite_size])

            # 3) Đánh giá con, hợp thành quần thể mới
            child_fit = np.array([evaluate(c) for c in children])
            pop = np.vstack([elites, children])
            fit = np.concatenate([elite_fit, child_fit])


class SimulatedAnnealing(BaseOptimizer):
    """SA nhị phân: một nghiệm hiện tại, mỗi bước lật n_flip bit ngẫu nhiên để tạo nghiệm lân cận.

    Nghiệm tốt hơn luôn được nhận; nghiệm xấu hơn được nhận với xác suất exp(-chênh_lệch / T).
    Nhiệt độ T giảm dần theo T <- T * cooling sau mỗi bước (làm lạnh hình học).
    T0 cần cùng cỡ với chênh lệch fitness giữa các nghiệm lân cận (khoảng 0.001 - 0.01).
    """

    def __init__(self, t0=0.01, cooling=0.995, n_flip=2, init_prob=0.5):
        self.t0 = t0
        self.cooling = cooling
        self.n_flip = n_flip
        self.init_prob = init_prob

    def search(self, evaluate, n_features, rng):
        current = rng.random(n_features) < self.init_prob
        f_current = evaluate(current)
        t = self.t0

        while True:
            neighbor = current.copy()
            flip = rng.choice(n_features, self.n_flip, replace=False)
            neighbor[flip] = ~neighbor[flip]                     # lật n_flip bit
            f_neighbor = evaluate(neighbor)

            delta = f_neighbor - f_current                       # > 0 nghĩa là xấu hơn
            if delta <= 0 or rng.random() < np.exp(-delta / t):
                current, f_current = neighbor, f_neighbor
            t = max(t * self.cooling, 1e-12)                     # làm lạnh (chặn dưới để không chia cho 0)


class ParticleSwarm(BaseOptimizer):
    """PSO nhị phân (Kennedy & Eberhart): vận tốc thực, vị trí nhị phân.

    v <- w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x), bị cắt trong [-v_max, v_max]
    Mỗi bit của vị trí mới = 1 với xác suất sigmoid(v).
    """

    def __init__(self, n_particles=30, w=0.7, c1=1.5, c2=1.5, v_max=4.0, init_prob=0.5):
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.init_prob = init_prob

    def search(self, evaluate, n_features, rng):
        shape = (self.n_particles, n_features)
        x = rng.random(shape) < self.init_prob                   # vị trí (mask) của từng hạt
        v = rng.uniform(-1.0, 1.0, shape)                        # vận tốc
        fit = np.array([evaluate(p) for p in x])

        pbest, pbest_fit = x.copy(), fit.copy()                  # tốt nhất của từng hạt
        gbest = pbest[np.argmin(pbest_fit)].copy()               # tốt nhất của cả đàn

        while True:
            r1, r2 = rng.random(shape), rng.random(shape)
            v = (self.w * v
                 + self.c1 * r1 * (pbest.astype(float) - x)
                 + self.c2 * r2 * (gbest.astype(float) - x))
            v = np.clip(v, -self.v_max, self.v_max)
            x = rng.random(shape) < 1.0 / (1.0 + np.exp(-v))     # sigmoid -> xác suất chọn feature
            fit = np.array([evaluate(p) for p in x])

            better = fit < pbest_fit
            pbest[better], pbest_fit[better] = x[better], fit[better]
            gbest = pbest[np.argmin(pbest_fit)].copy()


class AntColony(BaseOptimizer):
    """ACO cho chọn feature: mỗi feature là một quyết định 'chọn' / 'bỏ', mỗi lựa chọn có một lượng pheromone.

    Mỗi vòng, n_ants con kiến lần lượt quyết định từng feature với
        P(chọn feature i) = tau[i, chọn] / (tau[i, chọn] + tau[i, bỏ]).
    Sau đó: bay hơi tau <- (1 - rho) * tau, con kiến tốt nhất vòng đó để thêm pheromone
    (lượng = 1 - fitness) lên các lựa chọn nó đã dùng, rồi cắt tau trong [tau_min, tau_max] (kiểu MAX-MIN Ant System)
    để xác suất luôn còn khả năng khám phá.
    """

    def __init__(self, n_ants=30, rho=0.1, tau_min=0.05, tau_max=5.0, tau_init=1.0):
        self.n_ants = n_ants
        self.rho = rho
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_init = tau_init

    def search(self, evaluate, n_features, rng):
        tau = np.full((n_features, 2), self.tau_init)            # cột 0 = bỏ, cột 1 = chọn
        rows = np.arange(n_features)

        while True:
            p_select = tau[:, 1] / (tau[:, 0] + tau[:, 1])
            ants = rng.random((self.n_ants, n_features)) < p_select
            fit = np.array([evaluate(a) for a in ants])

            best = np.argmin(fit)
            tau *= (1.0 - self.rho)                                          # bay hơi
            tau[rows, ants[best].astype(int)] += max(0.0, 1.0 - fit[best])   # thêm pheromone từ kiến tốt nhất vòng
            np.clip(tau, self.tau_min, self.tau_max, out=tau)


# Đăng ký: tên (dùng trong config) -> lớp thuật toán
OPTIMIZERS = {
    "SA": SimulatedAnnealing,
    "GA": GeneticAlgorithm,
    "PSO": ParticleSwarm,
    "ACO": AntColony,
    "RS": RandomSearch,
}


def get_optimizer(name, params=None):
    return OPTIMIZERS[name](**(params or {}))
