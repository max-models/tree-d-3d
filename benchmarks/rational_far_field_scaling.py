import time

import numpy as np
from rational_far_field import aaa, aaa_poles_residues

R = np.sqrt(2.0)


def errors(n, rho, p, seed, eval_factors=(1.0, 1.5, 2.0, 4.0), samples=1024):
    r = np.random.default_rng(seed)
    zs = r.uniform(-1, 1, n) + 1j * r.uniform(-1, 1, n)
    q = r.choice([-1.0, 1.0], n)
    m = p // 2
    d = R / rho
    phi = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    Zf = d * np.exp(1j * (phi + np.pi / samples))
    a0 = q.sum()
    a = np.array([np.sum(q * zs**k) for k in range(p + 1)])
    t0 = time.time()
    hfun = lambda zt: (q[None, :] / (1 - zs[None, :] * zt[:, None])).sum(axis=1)
    zz, ff, w = aaa(hfun(1 / Zf), 1 / Zf, m)
    poles, res = aaa_poles_residues(zz, ff, w)
    rho_k, c_k = 1 / poles, -res / poles
    fit_time = time.time() - t0
    out = []
    for fac in eval_factors:
        Z = fac * d * np.exp(1j * phi)
        Wp = (q[None, :] / (Z[:, None] - zs[None, :])).sum(axis=1)
        scale = np.abs(Wp).max()
        Wp_t = a0 / Z + sum(a[k] / Z ** (k + 1) for k in range(1, p + 1))
        Wp_c = (c_k[None, :] / (Z[:, None] - rho_k[None, :])).sum(axis=1) + (
            a0 - c_k.sum()
        ) / Z
        out.append((np.abs(Wp_t - Wp).max() / scale, np.abs(Wp_c - Wp).max() / scale))
    return out, fit_time, np.abs(rho_k).max()


print(
    "== scaling with the number of sources per cell (p = 24, m = 12), fit and evaluation on |z| = d"
)
print(
    f"{'n':>6} {'rho':>5} | {'Taylor':>9} {'AAA-chg':>9} {'gain':>7} | {'fit s':>6} {'max|rho_k|/R':>12}"
)
for rho in (0.5, 0.7, 0.9):
    for n in (8, 32, 128, 512, 2048):
        e, ft, rmax = errors(n, rho, 24, seed=11)
        print(
            f"{n:6d} {rho:5.2f} | {e[0][0]:9.1e} {e[0][1]:9.1e} {e[0][0] / e[0][1]:7.1f} | {ft:6.3f} {rmax / R:12.2f}"
        )

print(
    "\n== validity away from the fitting circle (n = 32, p = 24, m = 12): error at |z| = f d for a fit on |z| = d"
)
print(f"{'rho':>5} {'f':>4} | {'Taylor':>9} {'AAA-chg':>9}")
for rho in (0.5, 0.9):
    e, _, _ = errors(32, rho, 24, seed=3)
    for fac, (et, ec) in zip((1.0, 1.5, 2.0, 4.0), e):
        print(f"{rho:5.2f} {fac:4.1f} | {et:9.1e} {ec:9.1e}")

print("\n== neutral cloud vs all-positive cloud (n = 128, rho = 0.7, p = 16)")
for label, signed in (("signed", True), ("all +1", False)):
    r = np.random.default_rng(4)
    n = 128
    p = 16
    m = 8
    rho = 0.7
    d = R / rho
    zs = r.uniform(-1, 1, n) + 1j * r.uniform(-1, 1, n)
    q = r.choice([-1.0, 1.0], n) if signed else np.ones(n)
    phi = np.linspace(0, 2 * np.pi, 1024, endpoint=False)
    Z = d * np.exp(1j * phi)
    Zf = d * np.exp(1j * (phi + np.pi / 1024))
    a0 = q.sum()
    a = np.array([np.sum(q * zs**k) for k in range(p + 1)])
    Wp = (q[None, :] / (Z[:, None] - zs[None, :])).sum(axis=1)
    scale = np.abs(Wp).max()
    Wp_t = a0 / Z + sum(a[k] / Z ** (k + 1) for k in range(1, p + 1))
    hfun = lambda zt: (q[None, :] / (1 - zs[None, :] * zt[:, None])).sum(axis=1)
    zz, ff, w = aaa(hfun(1 / Zf), 1 / Zf, m)
    poles, res = aaa_poles_residues(zz, ff, w)
    rho_k, c_k = 1 / poles, -res / poles
    Wp_c = (c_k[None, :] / (Z[:, None] - rho_k[None, :])).sum(axis=1) + (
        a0 - c_k.sum()
    ) / Z
    print(
        f"{label:7s}: Taylor {np.abs(Wp_t - Wp).max() / scale:9.1e}  AAA-chg {np.abs(Wp_c - Wp).max() / scale:9.1e}"
    )
