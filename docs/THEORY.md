# Theory notes

These notes derive every result the experiments test. Each section ends with the experiment that checks it, so you can go from a formula to the simulation that confirms (or breaks) it.

Notation: $y \in \mathbb{R}^n$, $X \in \mathbb{R}^{n \times p}$ with full column rank, $\varepsilon = y - X\beta$.

---

## 1. Least squares as an optimisation problem

$$\hat\beta = \arg\min_b \; S(b), \qquad S(b) = \|y - Xb\|_2^2 = y'y - 2b'X'y + b'X'Xb.$$

The gradient is $\nabla S(b) = -2X'y + 2X'Xb$. Setting it to zero gives the **normal equations**

$$X'X\hat\beta = X'y \quad\Longrightarrow\quad \hat\beta = (X'X)^{-1}X'y.$$

The Hessian is $2X'X$, which is positive definite when $X$ has full column rank, so this is the unique global minimum.

**Geometric reading.** Rewrite the normal equations as $X'(y - X\hat\beta) = 0$, i.e. $X'e = 0$. The residual is orthogonal to every column of $X$. So $\hat y = X\hat\beta$ is the point in the column space of $X$ closest to $y$: the **orthogonal projection** of $y$ onto $\text{col}(X)$.

Consequences you can check numerically:
- $\sum_i e_i = 0$ when $X$ contains an intercept (the column of ones is one of the columns $e$ is orthogonal to).
- $\hat y'e = 0$, so Pythagoras holds: $\|y\|^2 = \|\hat y\|^2 + \|e\|^2$.

*Checked in:* `tests/test_ols.py::test_projection_identities`, figure `01a`, table `01a_projection_identities.csv` (all identities hold to about $10^{-16}$).

## 2. The hat matrix

$$H = X(X'X)^{-1}X', \qquad \hat y = Hy, \qquad e = (I - H)y.$$

| Property | Why |
|---|---|
| $H' = H$ | symmetric, because $(X'X)^{-1}$ is symmetric |
| $H^2 = H$ | idempotent: projecting twice changes nothing |
| $\text{tr}(H) = p$ | $\text{tr}(X(X'X)^{-1}X') = \text{tr}((X'X)^{-1}X'X) = \text{tr}(I_p)$ |
| $HX = X$ | vectors already in $\text{col}(X)$ are left alone |
| $(I-H)X = 0$ | residual maker kills anything in $\text{col}(X)$ |

**Leverage.** $h_{ii} = H_{ii}$ measures how much $y_i$ pulls its own fitted value ($\partial \hat y_i / \partial y_i = h_{ii}$). Since $\text{tr}(H) = p$, the average leverage is $p/n$, and $2p/n$ is the usual "high leverage" flag.

With the thin QR factorisation $X = QR$, we get $H = QQ'$, so $h_{ii} = \|Q_{i\cdot}\|^2$. That is how the code computes leverage without ever forming the $n \times n$ matrix.

**Why residuals are not errors.** $e = (I-H)y = (I-H)\varepsilon$ because $(I-H)X\beta = 0$. If $\text{Var}(\varepsilon) = \sigma^2 I$, then

$$\text{Var}(e) = \sigma^2(I-H) \quad\Longrightarrow\quad \text{Var}(e_i) = \sigma^2(1 - h_{ii}).$$

High-leverage points have *small* residuals, because the line bends towards them. This single fact explains three things later on: why $s^2$ divides by $n - p$, why studentised residuals divide by $\sqrt{1-h_{ii}}$, and why HC2/HC3 beat HC0.

**Why $s^2 = e'e/(n-p)$.** $E[e'e] = E[\varepsilon'(I-H)\varepsilon] = \sigma^2\,\text{tr}(I-H) = \sigma^2(n-p)$. Fitting $p$ parameters uses up $p$ dimensions of the noise.

## 3. Computing $\hat\beta$: why QR and not the normal equations

The **condition number** $\kappa(X) = \sigma_{\max}/\sigma_{\min}$ (ratio of singular values) bounds how much relative errors get amplified. Backward-stable algorithms produce relative errors of order $\epsilon_{\text{mach}}\,\kappa$ where $\epsilon_{\text{mach}} \approx 2.2 \times 10^{-16}$.

- **Normal equations** solve a system in $X'X$, and $\kappa(X'X) = \kappa(X)^2$. Error grows like $\epsilon\,\kappa^2$.
- **QR**: $X = QR$ with $Q'Q = I$. Then $\|y - Xb\|^2 = \|Q'y - Rb\|^2 + \text{const}$, solved by the triangular system $Rb = Q'y$. Because $Q$ preserves lengths, $\kappa(R) = \kappa(X)$. Error grows like $\epsilon\,\kappa$.
- **SVD**: $X = USV'$, $\hat\beta = VS^{-1}U'y$. Same $\epsilon\,\kappa$ behaviour, and it degrades gracefully when $X$ is rank deficient.

With polynomial regressors ($1, x, x^2, \dots$) the columns become nearly parallel and $\kappa$ explodes. At degree 11, $\kappa \approx 10^8$: QR still recovers $\beta$ to about $10^{-9}$, while the normal equations are off by 25% of $\|\beta\|$.

*Checked in:* figure `01b`, table `01b_solver_accuracy.csv`.

## 4. Unbiasedness and the sampling variance

Substitute $y = X\beta + \varepsilon$:

$$\hat\beta = \beta + (X'X)^{-1}X'\varepsilon.$$

This one line drives the entire project. Everything that can go wrong is a statement about the term $(X'X)^{-1}X'\varepsilon$.

**Unbiasedness (needs A3, exogeneity).** If $E[\varepsilon \mid X] = 0$ then $E[\hat\beta \mid X] = \beta$.

**Variance (needs A4, spherical errors).** In general, with $\Omega = \text{Var}(\varepsilon \mid X)$:

$$\text{Var}(\hat\beta \mid X) = (X'X)^{-1}X'\Omega X(X'X)^{-1}. \tag{sandwich}$$

Only if $\Omega = \sigma^2 I$ does this collapse to the textbook formula $\sigma^2(X'X)^{-1}$. **The classical standard error is the sandwich formula evaluated under an assumption.** When the assumption fails, the formula is simply the wrong number.

**An exact check.** In the baseline simulation, $x_i \sim N(0,1)$ and $\sigma = 1$. The slope variance given $X$ is $\sigma^2 / \sum (x_i - \bar x)^2$, and $\sum(x_i - \bar x)^2 \sim \chi^2_{n-1}$. Using $E[1/\chi^2_k] = 1/(k-2)$:

$$\text{Var}(\hat\beta_1) = \frac{1}{n - 3}.$$

At $n = 50$ this predicts sd $0.1459$; the simulation gives $0.1454$.

*Checked in:* table `02a_baseline_consistency.csv`, figure `02a`.

**Gauss-Markov.** Under A1 to A4, any other estimator that is linear in $y$ and unbiased has variance at least as large as OLS. Proof sketch: write the competitor as $\tilde\beta = ((X'X)^{-1}X' + D)y$. Unbiasedness forces $DX = 0$, and then $\text{Var}(\tilde\beta) = \text{Var}(\hat\beta) + \sigma^2 DD'$, and $DD'$ is positive semi-definite.

The experiment pits OLS against Wald's (1940) grouping estimator (slope between the means of the top and bottom halves of $x$). Both are unbiased; OLS has sd $0.165$ versus $0.207$.

*Checked in:* table `02d_gauss_markov.csv`.

## 5. Exact inference (needs A5, normality)

If additionally $\varepsilon \mid X \sim N(0, \sigma^2 I)$:

1. $\hat\beta \mid X \sim N(\beta, \sigma^2(X'X)^{-1})$, since it is a linear function of a Gaussian vector.
2. $(n-p)s^2/\sigma^2 = \varepsilon'(I-H)\varepsilon/\sigma^2 \sim \chi^2_{n-p}$, because $I - H$ is a projector of rank $n-p$.
3. $\hat\beta$ and $s^2$ are independent, because $\hat\beta$ depends on $H\varepsilon$ and $e$ on $(I-H)\varepsilon$, and $H(I-H) = 0$ makes these Gaussian vectors uncorrelated.

So

$$t_j = \frac{\hat\beta_j - \beta_j}{s\sqrt{[(X'X)^{-1}]_{jj}}} \sim t_{n-p}.$$

The Student $t$ has fatter tails than $N(0,1)$ because $s$ is itself noisy. Using 1.96 instead of the $t$ critical value at $n = 5$ gives 86% coverage instead of 95%.

**F-test.** For $q$ linear restrictions $H_0: R\beta = r$:

$$F = \frac{(R\hat\beta - r)'\left[R\,\hat V\,R'\right]^{-1}(R\hat\beta - r)}{q} \sim F_{q,\,n-p}$$

with $\hat V = s^2(X'X)^{-1}$. Swapping in a robust $\hat V$ gives a robust Wald test (Section 6).

**What normality is *not* needed for.** Unbiasedness, the variance formula, and Gauss-Markov need no normality at all. And as $n$ grows the central limit theorem makes $t_j$ approximately normal anyway. Normality only buys *exact* small-sample inference.

*Checked in:* figure `02b`, `02c`; table `02c_f_test_size.csv` (F-test rejects a true null 4.9% of the time).

## 6. Heteroskedasticity and the sandwich estimators

When $\Omega = \text{diag}(\sigma_1^2, \dots, \sigma_n^2)$, the middle of the sandwich is $X'\Omega X = \sum_i \sigma_i^2 x_i x_i'$. White's idea: we cannot estimate each $\sigma_i^2$, but we only need this $p \times p$ *sum*, and $\sum_i e_i^2 x_i x_i'$ estimates it consistently.

| Estimator | Weight on $x_ix_i'$ | Idea |
|---|---|---|
| HC0 | $e_i^2$ | White (1980) |
| HC1 | $e_i^2 \cdot \frac{n}{n-p}$ | degrees-of-freedom scaling, as in $s^2$ |
| HC2 | $e_i^2/(1-h_{ii})$ | undo $\text{Var}(e_i) = \sigma_i^2(1-h_{ii})$ |
| HC3 | $e_i^2/(1-h_{ii})^2$ | approximately the jackknife; over-corrects slightly, which helps in small samples |

**How wrong is the classical SE in our DGP?** For the slope with $x \sim N(0,1)$ and $\text{Var}(\varepsilon \mid x) = 0.1 + x^2$, the large-$n$ ratio of true to classical variance is

$$\frac{E[x^2\sigma^2(x)]}{E[x^2]\,E[\sigma^2(x)]} = \frac{0.1 + E[x^4]}{0.1 + 1} = \frac{3.1}{1.1} \approx 2.82.$$

So the classical SE is about $1/\sqrt{2.82} = 0.60$ of the truth, and a nominal 95% interval covers $P(|Z| < 1.96 \times 0.60) \approx 76\%$. The simulation: SE ratio $0.59$ to $0.61$, coverage $76\%$.

Note what makes heteroskedasticity *harmful*: noise that is largest where $x$ is far from its mean, exactly the points that determine the slope. Heteroskedasticity unrelated to $x^2$ barely matters for the slope SE.

**Efficiency.** Under heteroskedasticity OLS is still unbiased but no longer best. Weighted least squares, dividing each row by $\sigma_i$, restores spherical errors, and Gauss-Markov then applies to the transformed problem (Aitken's theorem). With the true weights, WLS cuts the slope sd from 0.172 to 0.121. In practice the weights must be estimated, which is why robust SEs on OLS are the default.

*Checked in:* figures `04a`, table `04a_hc_comparison.csv`, `04b_wls_efficiency.csv`, `04d_joint_test_size.csv`.

## 7. Autocorrelation and Newey-West

With time-series data, $\Omega$ has off-diagonal terms. Let $u_t = x_t\varepsilon_t$ (the "score"). Then

$$X'\Omega X = \text{Var}\Big(\sum_t u_t\Big) = \sum_t \text{Var}(u_t) + \sum_{t \ne s}\text{Cov}(u_t, u_s).$$

**Inflation factor.** If $x_t$ and $\varepsilon_t$ are independent AR(1) processes with coefficients $\rho_x, \rho_\varepsilon$ and unit variance, then $\text{Cov}(u_t, u_{t-k}) = (\rho_x\rho_\varepsilon)^{|k|}$, and

$$\frac{\text{Var}(\hat\beta_1)}{\text{classical}} \approx \sum_{k=-\infty}^{\infty}(\rho_x\rho_\varepsilon)^{|k|} = \frac{1 + \rho_x\rho_\varepsilon}{1 - \rho_x\rho_\varepsilon}.$$

With $\rho = 0.8$ for both, that is $4.56$: the true SE is more than twice the classical one. Note the factor needs *both* the regressor and the error to be persistent. A random-walk-like error alone, with i.i.d. $x$, is mostly harmless for the slope SE.

**HC3 cannot help**, because it only reweights the diagonal of $\Omega$.

**Newey-West (HAC)** estimates the autocovariances up to lag $L$:

$$\hat S = \hat\Gamma_0 + \sum_{l=1}^{L}\Big(1 - \frac{l}{L+1}\Big)(\hat\Gamma_l + \hat\Gamma_l'), \qquad \hat\Gamma_l = \sum_t \hat u_t\hat u_{t-l}'.$$

The Bartlett weights $1 - l/(L+1)$ guarantee a positive semi-definite matrix. The trade-off is unavoidable: a small $L$ misses long-range correlation (bias), a large $L$ adds estimation noise. With the rule-of-thumb $L = \lfloor 4(n/100)^{2/9}\rfloor$ and $\rho = 0.95$, HAC coverage is still only 59%. HAC is a large-sample fix that is known to under-cover with persistent data.

*Checked in:* figure `04b`, table `04c_hac_persistence.csv`.

## 8. Endogeneity: when the estimate itself is wrong

From $\hat\beta = \beta + (X'X)^{-1}X'\varepsilon$, for the simple regression slope

$$\text{plim}\,\hat\beta_1 = \beta_1 + \frac{\text{Cov}(x, \varepsilon)}{\text{Var}(x)}.$$

If $x$ is correlated with the error, OLS converges to the wrong number. Three mechanisms, each with a closed-form bias the simulation reproduces:

| Mechanism | Setup | Bias formula | Predicted | Simulated |
|---|---|---|---|---|
| Omitted variable | true $y = 2 + 3x + 2w + u$, $\text{corr}(x,w) = 0.5$ | $\gamma\,\text{Cov}(x,w)/\text{Var}(x)$ | $+1.000$ | $+1.000$ |
| Measurement error | observe $x + m$, $\text{sd}(m) = 0.7$ | $-\beta_1\frac{\sigma_m^2}{\sigma_x^2 + \sigma_m^2}$ | $-0.987$ | $-0.987$ |
| Simultaneity | $x = z + 0.5\varepsilon$ | $\frac{0.5}{1 + 0.25}$ | $+0.400$ | $+0.401$ |

**Why coverage goes to zero.** The $t$-statistic for the true value is

$$t = \frac{\hat\beta_1 - \beta_1}{\text{se}} \approx \frac{\text{bias}}{c/\sqrt{n}} + O_p(1) \longrightarrow \infty.$$

The SE shrinks like $1/\sqrt n$ while the bias stays put. More data does not help; it makes you more confident in the wrong answer. No standard-error correction can fix this, because the problem is in the numerator. The cures are design-level: include the omitted variable, use instruments, or use a natural experiment.

**A subtle case.** Under measurement error, a true slope of zero stays zero ($0 \times$ attenuation $= 0$). So measurement error biases the size of an effect but does not manufacture false positives. Omitted variables and simultaneity do.

*Checked in:* figure `03b`, table `03_violations.csv`.

## 9. Non-normal errors and the limits of the CLT

With finite variance, $\sqrt n(\hat\beta - \beta)$ is asymptotically normal whatever the error distribution, so $t$-tests become approximately valid. Our skewed and heavy-tailed scenarios show this: coverage is already 93.7% to 95% from $n = 10$.

**Cauchy errors** (no finite mean or variance) are instructive. The estimator never settles down: the interquartile range of $\hat\beta_1$ only falls from 1.13 at $n = 10$ to 0.55 at $n = 1000$, while with finite-variance errors it would shrink 10 times. Yet classical CI coverage is about 95%, because the $t$-statistic is *self-normalising*: a huge error inflates both the numerator and $s$. The intervals are honest only in the sense of being enormous. Good coverage does not mean useful precision.

*Checked in:* table `03d_cauchy.csv`.

## 10. Multicollinearity

For regressor $j$, with $R_j^2$ from regressing $x_j$ on the other regressors:

$$\text{Var}(\hat\beta_j \mid X) = \frac{\sigma^2}{\sum_i (x_{ij} - \bar x_j)^2}\cdot\underbrace{\frac{1}{1 - R_j^2}}_{\text{VIF}}.$$

With two standard normal regressors of correlation $\rho$, the residual sum of squares of $x_1$ on $(1, x_2)$ is $(1-\rho^2)\chi^2_{n-2}$, so $\text{Var}(\hat\beta_1) = 1/[(1-\rho^2)(n-4)]$ exactly. At $\rho = 0.99$, that is 50 times the variance, matched by the simulation (sd 0.718 vs 0.724 predicted), yet bias is zero and coverage stays at 95%. Collinearity is not a violation; it is the data honestly telling you it cannot separate two effects.

*Checked in:* figure `01c`, table `01c_multicollinearity.csv`.

## 11. Influence

**Studentised residual:** $r_i = e_i / (s\sqrt{1-h_{ii}})$, so every residual is on the same scale.

**Cook's distance** measures how far the whole coefficient vector moves when point $i$ is deleted:

$$D_i = \frac{(\hat\beta - \hat\beta_{(i)})'X'X(\hat\beta - \hat\beta_{(i)})}{p\,s^2} = \frac{r_i^2}{p}\cdot\frac{h_{ii}}{1 - h_{ii}}.$$

Influence is leverage times surprise. A point with high leverage but on the line does no harm; an outlier in $y$ at average $x$ mostly shifts the intercept.

**Masking.** In experiment 5b the bad point sits 16 units below the true line, but its residual is only about 4 standard deviations, because the fit bends towards it ($h_{ii} = 0.73$). Raw residuals hide exactly the points that matter most. Leverage and Cook's distance ($D = 22.5$ versus at most 0.14 for every other point) reveal it.

*Checked in:* figure `05b`, table `05_influence.csv`.

## References

- Davidson, R. and MacKinnon, J. (2004). *Econometric Theory and Methods*. Oxford University Press.
- Golub, G. and Van Loan, C. (2013). *Matrix Computations*, 4th ed. Johns Hopkins.
- Long, J. S. and Ervin, L. (2000). Using heteroscedasticity consistent standard errors in the linear regression model. *The American Statistician*, 54(3).
- MacKinnon, J. and White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*, 29(3).
- Newey, W. and West, K. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3).
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4).
- Wooldridge, J. (2010). *Econometric Analysis of Cross Section and Panel Data*, 2nd ed. MIT Press.
