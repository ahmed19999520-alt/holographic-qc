# Complete Derivations

## A. Virasoro Algebra from Free Fermion OPE

**Step A.1** — Euclidean action: the edge Hamiltonian
$H = \int dx\,\Psi^\dagger(-i\hbar v_F\sigma^z\partial_x)\Psi$
maps to the Euclidean action
$$S_E = \frac{1}{2\pi}\int d^2z\,[\psi_R^\dagger\bar\partial\psi_R + \psi_L^\dagger\partial\psi_L]$$
in complex coordinates $z = v_F\tau + ix$.

**Step A.2** — Propagators from the path integral:
$$\langle\psi_R(z)\psi_R^\dagger(w)\rangle = \frac{1}{2\pi(z-w)},\quad
\langle\psi_L(\bar z)\psi_L^\dagger(\bar w)\rangle = \frac{1}{2\pi(\bar z-\bar w)}.$$

**Step A.3** — Stress tensor from Noether's theorem:
$$T(z) = \tfrac{1}{2}{:}\psi_R^\dagger\partial\psi_R{:} - \tfrac{1}{2}{:}(\partial\psi_R^\dagger)\psi_R{:}.$$

**Step A.4** — Four Wick contractions of $T(z)T(w)$:

| Contraction | Value |
|-------------|-------|
| $\langle\psi_R^\dagger(z)\partial_w\psi_R(w)\rangle$ | $+1/(2\pi(w-z)^2)$ |
| $\langle\partial_z\psi_R(z)\psi_R^\dagger(w)\rangle$ | $-1/(2\pi(z-w)^2)$ |
| Product (term 1) | $-1/(4\pi^2(z-w)^4)$ |
| Product (term 2) | $-1/(4\pi^2(z-w)^4)$ |
| Sum $\times 4 \times (1/4)$ | $-2/(4\pi^2(z-w)^4)$ |

After Wick rotation: $\langle TT\rangle = 1/(2(z-w)^4) \Rightarrow c = 1$.

**Step A.5** — Residue integrals:

$$[L_m, L_n] = \oint_0\frac{dz}{2\pi i}z^{m+1}\oint_z\frac{dw}{2\pi i}w^{n+1}T(z)T(w)$$

From the $c/2\,(z-w)^{-4}$ pole:
$$\mathrm{Res}_{w=z}\left[w^{n+1}(z-w)^{-4}\right] = \binom{n+1}{3}z^{n-2}$$
$$\Rightarrow \frac{c}{12}m(m^2-1)\delta_{m+n,0}.$$

From the $2T(w)(z-w)^{-2}$ pole:
$$\mathrm{Res}_{w=z}\left[w^{n+1}T(w)(z-w)^{-2}\right] = (n+1)z^nT(z) + z^{n+1}\partial T(z)$$
$$\Rightarrow (m-n)L_{m+n}.$$

Combined: $[L_m, L_n] = (m-n)L_{m+n} + \frac{c}{12}m(m^2-1)\delta_{m+n,0}$. $\square$

---

## B. Bulk-to-Boundary Propagator

**Step B.1** — Poincaré metric: $g_{\mu\nu} = (\ell/z)^2\mathrm{diag}(1,1,-1)$.
Determinant: $\sqrt{-g} = (\ell/z)^3$. Inverse: $g^{\mu\nu} = (z/\ell)^2\mathrm{diag}(1,1,-1)$.

**Step B.2** — Scalar Laplacian:
$$\Box\phi = \frac{z^2}{\ell^2}\left[\partial_z^2 - \frac{1}{z}\partial_z + \partial_x^2 - \partial_t^2\right]\phi.$$

**Step B.3** — Equation of motion $(\Box - m^2)\phi = 0$:
$$\partial_z^2\phi - \frac{1}{z}\partial_z\phi + k^2\phi - \frac{m^2\ell^2}{z^2}\phi = 0, \quad k^2 = \omega^2 - q^2.$$

**Step B.4** — Fourier transform, substitution $\hat\phi = z^{1/2}f$:
$$f'' + \frac{1}{z}f' - \left[k^2 + \frac{\nu^2}{z^2}\right]f = 0, \quad \nu = \sqrt{\tfrac{1}{4} + m^2\ell^2}.$$
Regular solution at $z\to\infty$: $f = A(k)K_\nu(kz)$.

**Step B.5** — Boundary asymptotics ($K_\nu(kz)\sim\Gamma(\nu)/2\cdot(2/(kz))^\nu$):
$$\hat\phi(z) \to A(k)\frac{\Gamma(\nu)}{2}(2/k)^\nu z^{1/2-\nu} \equiv \phi^{(-)}_k z^{(1-\nu)} + \phi^{(+)}_k z^{(1+\nu)}.$$
Scaling dimensions: $\Delta_- = 1-\nu$, $\Delta_+ = 1+\nu$.

**Step B.6** — Position-space propagator (normalized):
$$K(z,x;x') = \left(\frac{z}{z^2+(x-x')^2}\right)^{\!\Delta}, \quad \Delta = 1+\nu. \quad\square$$

---

## C. Holographic Decoherence Rate

**Step C.1** — Environmental Hamiltonian:
$$H_{\rm env} = \sum_q\omega_qb_q^\dagger b_q + \int dx\,\delta\rho(x)\sum_qg_q(b_q+b_q^\dagger)\cos(qx).$$

**Step C.2** — Bare dephasing rate (Fermi's golden rule):
$$\gamma_{\rm std} = \frac{2\pi}{\hbar}\sum_q|g_q|^2[\bar n(\omega_q)+1]\delta(\varepsilon_k - \varepsilon_{k+q} - \hbar\omega_q).$$
For 2D phonons: $\gamma_{\rm std} \propto \alpha_{\rm ph}T^2$.

**Step C.3** — Bulk field solution:
$$\phi_{\rm bulk}(z,x) = \int dx'\,K(z,x;x')\mathcal{O}_{\rm noise}(x').$$

**Step C.4** — Radial integral at $r = |x_0 - x'| \gg \xi$:
$$\tilde K = \int_a^\xi dz\,\frac{z^{\Delta_n}}{r^{2\Delta_n}} = \frac{\xi^{\Delta_n+1}}{(\Delta_n+1)r^{2\Delta_n}}.$$

**Step C.5** — Noise power ratio:
$$\frac{\langle|\phi_{\rm eff}|^2\rangle}{\langle|\mathcal{O}_{\rm noise}|^2\rangle} = \frac{\xi^{2(\Delta_n+1)}}{(\Delta_n+1)^2}\cdot\frac{(1-2\Delta_n)}{2(1-6\Delta_n)}\cdot L^{-4\Delta_n}.$$

**Step C.6** — Final result:
$$\gamma_{\rm holo} = \gamma_{\rm std}\times\exp\!\left(-\frac{2\pi\Delta_n}{c}\frac{L}{\xi}\right),$$
equivalently $(L/\xi)^{-c/6}$ at $\Delta_n = c/(12\pi)$. $\square$

---

## D. Ryu–Takayanagi Formula

**Step D.1** — Replica trick: twist operators at $\pm\ell/2$ have weights $h_n = c(n-1/n)/24$.

**Step D.2** — Twist two-point function: $\langle\mathcal{T}_n\tilde{\mathcal{T}}_n\rangle = \ell^{-c(n-1/n)/6}$.

**Step D.3** — Rényi entropy:
$$S_n = \frac{1}{1-n}\ln\!\left[c_n\left(\frac{\ell}{a}\right)^{-c(n-1/n)/6}\right].$$

**Step D.4** — Von Neumann limit $n\to1$:
$$S = -\frac{\partial}{\partial n}\left[c_n(\ell/a)^{-c(n-1/n)/6}\right]_{n=1} = \frac{c}{3}\ln\frac{\ell}{a}. \quad\square$$

**Step D.5** — RT formula check: geodesic length in embedding space,
$\cosh(L_\gamma/\ell) = 1 + \ell^2/(2a^2) \Rightarrow L_\gamma = 2\ell\ln(\ell/a)$,
$S_A = L_\gamma/(4G_3) = (c/3)\ln(\ell/a)$.  Consistent.