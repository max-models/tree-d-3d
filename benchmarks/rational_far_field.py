"""Idea 07: rational (AAA) far fields and optimised equivalent charges vs the Taylor multipole series.

Sources: n signed unit charges uniform in the square [-1,1]^2 (half-diagonal R = sqrt 2).
Targets on the circle |z| = d with rho = R/d the separation ratio (the MAC's r/d).
Far field (complex), analytic and branch-safe for |z| > R:
    W(z) = a0 log z + g(1/z),  g(zeta) = sum_j q_j log1p(-z_j zeta)   (so W' = a0/z - g'(zeta)/z^2)
We compare the FIELD error  max|W'_approx - W'| / max|W'|  on the circle, at equal cost per target:
    Taylor   : p terms            (cost p per target, p complex coefficients)
    AAA      : type (m,m) rational approximant of g on |zeta| = 1/d, m = p/2  (cost ~2m per target)
    AAA-chg  : the AAA approximant of h(zeta) = sum q_j/(1 - z_j zeta) in partial fractions = m
               equivalent complex charges c_k at rho_k, W' ~ sum c_k/(z - rho_k)  (cost m per target)
    Prony    : m charges matching moments M_0..M_{2m-1} (Padé at infinity; ill-conditioned)
"""

import numpy as np
import scipy.linalg as sla


def aaa(F, Z, m, tol=1e-13):
    """AAA (Nakatsukasa, Sete, Trefethen 2018): support points z, values f, weights w."""
    M = len(Z)
    J = np.arange(M)
    z, f = [], []
    R = np.full(M, F.mean())
    w = np.ones(1)
    for _ in range(m + 1):
        j = int(np.argmax(np.abs(F - R)))
        z.append(Z[j])
        f.append(F[j])
        J = J[J != j]
        zz, ff = np.array(z), np.array(f)
        C = 1.0 / (Z[J][:, None] - zz[None, :])
        A = (F[J][:, None] - ff[None, :]) * C
        _, _, Vh = np.linalg.svd(A, full_matrices=False)
        w = Vh[-1].conj()
        R = F.copy()
        R[J] = (C @ (w * ff)) / (C @ w)
        if np.abs(F - R).max() <= tol * np.abs(F).max():
            break
    return np.array(z), np.array(f), w


def aaa_eval(zz, ff, w, Zt):
    C = 1.0 / (Zt[:, None] - zz[None, :])
    return (C @ (w * ff)) / (C @ w)


def aaa_poles_residues(zz, ff, w):
    m = len(zz)
    E = np.zeros((m + 1, m + 1), complex)
    E[0, 1:] = w
    E[1:, 0] = 1.0
    E[1:, 1:] = np.diag(zz)
    B = np.eye(m + 1, dtype=complex)
    B[0, 0] = 0.0
    ev = sla.eigvals(E, B)
    poles = ev[np.isfinite(ev)]
    poles = poles[np.abs(poles) < 1e12]
    # residues: N(p)/D'(p)
    N = ((w * ff)[None, :] / (poles[:, None] - zz[None, :])).sum(axis=1)
    Dp = -(w[None, :] / (poles[:, None] - zz[None, :]) ** 2).sum(axis=1)
    return poles, N / Dp


def prony(moments, m):
    H = np.array([[moments[i + j] for j in range(m)] for i in range(m)])
    coef = np.linalg.solve(H, -np.array([moments[m + i] for i in range(m)]))
    rho = np.roots(np.r_[1.0, coef[::-1]])
    V = np.vander(rho, m, increasing=True).T
    return np.linalg.solve(V, np.array(moments[:m])), rho


def study(n, ratios, orders, seeds):
    R = np.sqrt(2.0)
    print(
        f"== n = {n} signed charges in [-1,1]^2 (half-diagonal {R:.3f}); field error on |z| = d, max over circle, mean over {len(seeds)} clouds"
    )
    print(
        f"{'rho=R/d':>8} {'p':>3} {'m':>3} | {'Taylor(p)':>10} {'AAA(m,m)':>10} {'AAA-chg(m)':>11} {'Prony(m)':>10}"
    )
    for rho in ratios:
        d = R / rho
        for p in orders:
            m = max(p // 2, 1)
            acc = np.zeros(4)
            cnt = np.zeros(4)
            for seed in seeds:
                r = np.random.default_rng(seed)
                zs = r.uniform(-1, 1, n) + 1j * r.uniform(-1, 1, n)
                q = r.choice([-1.0, 1.0], n)
                phi = np.linspace(0, 2 * np.pi, 1024, endpoint=False)
                Z = d * np.exp(1j * phi)
                Zf = d * np.exp(1j * (phi + np.pi / 1024))
                zeta, zetaf = 1 / Z, 1 / Zf
                a0 = q.sum()
                Wp = (q[None, :] / (Z[:, None] - zs[None, :])).sum(axis=1)  # exact W'
                scale = np.abs(Wp).max()
                # Taylor
                a = np.array([np.sum(q * zs**k) for k in range(p + 1)])
                Wp_t = a0 / Z + sum(a[k] / Z ** (k + 1) for k in range(1, p + 1))
                acc[0] += np.abs(Wp_t - Wp).max() / scale
                cnt[0] += 1
                # AAA of g(zeta): W' = a0/z - g'(zeta) zeta^2; derivative of the barycentric form by FD in zeta
                g = lambda zt: np.log1p(-zs[None, :] * zt[:, None]) @ q
                zz, ff, w = aaa(g(zetaf), zetaf, m)
                h = 1e-6 * (1 / d)
                dg = (aaa_eval(zz, ff, w, zeta + h) - aaa_eval(zz, ff, w, zeta - h)) / (
                    2 * h
                )
                Wp_a = a0 / Z - dg * zeta**2
                acc[1] += np.abs(Wp_a - Wp).max() / scale
                cnt[1] += 1
                # AAA of h(zeta) = sum q/(1 - z_j zeta) -> partial fractions -> equivalent charges
                hfun = lambda zt: (q[None, :] / (1 - zs[None, :] * zt[:, None])).sum(
                    axis=1
                )
                zz2, ff2, w2 = aaa(hfun(zetaf), zetaf, m)
                poles, res = aaa_poles_residues(zz2, ff2, w2)
                # h(zeta) ~ sum res_k/(zeta - p_k) + const ; 1/(zeta - p) = -(1/p) * 1/(1 - zeta/p) ; W' = h(1/z)/z
                # equivalent charge c_k at rho_k = 1/p_k: q/(z - rho) = (q/z) /(1 - rho/z) -> compare: res/(zeta-p) = -(res/p)/(1 - zeta/p)
                rho_k = 1 / poles
                c_k = -res / poles
                Wp_c = (c_k[None, :] / (Z[:, None] - rho_k[None, :])).sum(axis=1) + (
                    a0 - c_k.sum()
                ) / Z  # fix the monopole
                acc[2] += np.abs(Wp_c - Wp).max() / scale
                cnt[2] += 1
                # Prony
                try:
                    mom = [np.sum(q * zs**k) for k in range(2 * m)]
                    c, rr = prony(mom, m)
                    Wp_p = (c[None, :] / (Z[:, None] - rr[None, :])).sum(axis=1)
                    e = np.abs(Wp_p - Wp).max() / scale
                    if np.isfinite(e):
                        acc[3] += e
                        cnt[3] += 1
                except np.linalg.LinAlgError:
                    pass
            acc = acc / np.maximum(cnt, 1)
            print(
                f"{rho:8.2f} {p:3d} {m:3d} | {acc[0]:10.1e} {acc[1]:10.1e} {acc[2]:11.1e} {acc[3]:10.1e}"
            )


study(32, ratios=[0.3, 0.5, 0.7, 0.9], orders=[4, 8, 12, 16, 24], seeds=[1, 2, 3])
study(256, ratios=[0.5, 0.7, 0.9], orders=[8, 16, 24], seeds=[5, 6])
