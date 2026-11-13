# Study guide

Questions to check that you understand the project well enough to defend it. Try each one before opening the answer. They go roughly from the course material to the kind of follow-ups an interviewer asks about a project.

---

### Foundations

<details>
<summary>1. Derive the OLS estimator two ways.</summary>

Calculus: minimise $\|y - Xb\|^2$; the gradient $-2X'y + 2X'Xb = 0$ gives $X'X\hat\beta = X'y$.

Geometry: the closest point to $y$ in $\text{col}(X)$ is the one whose residual is orthogonal to the subspace, so $X'(y - X\hat\beta) = 0$. Same equations.
</details>

<details>
<summary>2. Why is the trace of the hat matrix equal to p, and what does that tell you about leverage?</summary>

$\text{tr}(X(X'X)^{-1}X') = \text{tr}((X'X)^{-1}X'X) = \text{tr}(I_p) = p$ by the cyclic property of the trace. So leverages sum to $p$ and average $p/n$. A point with $h_{ii}$ well above $2p/n$ has unusual influence on its own fitted value.
</details>

<details>
<summary>3. Why divide the residual sum of squares by n − p instead of n?</summary>

$e = (I-H)\varepsilon$, so $E[e'e] = \sigma^2\,\text{tr}(I - H) = \sigma^2(n-p)$. Residuals live in an $(n-p)$-dimensional subspace; fitting used up $p$ dimensions.
</details>

<details>
<summary>4. Why does no serious library solve the normal equations directly?</summary>

$\kappa(X'X) = \kappa(X)^2$. Floating-point error scales with the condition number, so forming $X'X$ squares it. QR keeps $\kappa(R) = \kappa(X)$ because $Q$ is orthogonal. Figure `01b` shows the gap: at $\kappa \approx 10^8$, $10^{-9}$ error for QR versus 25% for the normal equations.
</details>

### Assumptions

<details>
<summary>5. Which assumptions does unbiasedness need? Which does the classical standard error need? Which does the t-distribution need?</summary>

Unbiasedness: linearity, full rank, exogeneity (A1 to A3). Classical SE: additionally spherical errors (A4). Exact $t$ in small samples: additionally normality (A5). Asymptotically the $t$-test only needs A1 to A4 plus regularity conditions.
</details>

<details>
<summary>6. "My coefficient is unbiased, so my conclusions are fine." What's wrong with this?</summary>

Unbiased only means right on average. Significance depends on the standard error. Under heteroskedasticity the estimate is unbiased but the classical SE is 41% too small in our setup, so a zero effect is declared significant 25% of the time.
</details>

<details>
<summary>7. Why does coverage go to zero under endogeneity as n grows, instead of improving?</summary>

$t = (\hat\beta - \beta)/\text{se}$. The numerator converges to the bias, a constant. The denominator shrinks like $1/\sqrt n$. So $|t| \to \infty$ and the interval eventually excludes the true value in every sample.
</details>

<details>
<summary>8. Can robust standard errors fix omitted variable bias?</summary>

No. Robust SEs change the denominator of the $t$-statistic. Omitted variable bias is in the numerator: the point estimate itself is wrong. You need the omitted variable, an instrument, or a different research design.
</details>

<details>
<summary>9. Why don't residual plots reveal endogeneity?</summary>

OLS forces $X'e = 0$ by construction. The residuals are orthogonal to $X$ whether or not the true errors are. Whatever correlation existed between $X$ and $\varepsilon$ has been absorbed into $\hat\beta$.
</details>

<details>
<summary>10. Measurement error in x biases the slope towards zero. Why doesn't it create false positives?</summary>

$\text{plim}\,\hat\beta_1 = \beta_1 \cdot \sigma_x^2/(\sigma_x^2 + \sigma_m^2)$. Multiplicative attenuation of zero is still zero. It hides real effects; it doesn't invent fake ones. Measurement error in $y$ doesn't bias anything; it just adds noise.
</details>

### Robust inference

<details>
<summary>11. Explain the sandwich formula in one sentence.</summary>

$\text{Var}(\hat\beta) = (X'X)^{-1}\,X'\Omega X\,(X'X)^{-1}$: the bread is the sensitivity of $\hat\beta$ to the data, the meat is the variance of the score $\sum x_i\varepsilon_i$; the classical formula is the special case $\Omega = \sigma^2 I$.
</details>

<details>
<summary>12. Why does HC3 do better than HC0 in small samples?</summary>

Residuals understate errors at high-leverage points: $\text{Var}(e_i) = \sigma_i^2(1 - h_{ii})$. HC0 plugs in raw $e_i^2$ and so underestimates the meat exactly where it matters most. HC3 divides by $(1-h_{ii})^2$, which roughly matches the jackknife. At $n = 10$ in our simulation: HC0 79% coverage, HC3 92%.
</details>

<details>
<summary>13. Which kind of heteroskedasticity actually hurts the slope SE?</summary>

The kind correlated with $(x - \bar x)^2$. The ratio of true to classical variance is $E[x^2\sigma^2(x)] / (E[x^2]E[\sigma^2(x)])$, which equals 1 when $\sigma^2$ is unrelated to $x^2$.
</details>

<details>
<summary>14. Why is HC3 useless for autocorrelation, and why does Newey-West still under-cover?</summary>

HC estimators only estimate the diagonal of $\Omega$. Autocorrelation lives off the diagonal. Newey-West estimates autocovariances up to lag $L$ with downweighting; with very persistent data, the rule-of-thumb $L$ is too short and the Bartlett weights shrink the long-lag terms, so it still underestimates the variance (59% coverage at $\rho = 0.95$).
</details>

<details>
<summary>15. Robust SEs fix inference under heteroskedasticity. What do they not fix?</summary>

Efficiency. OLS is no longer BLUE. WLS with correct weights is (Aitken), and cut the slope sd from 0.172 to 0.121 in our experiment.
</details>

### Interpretation and quant context

<details>
<summary>16. Your backtest regression on daily data has t = 3.1. What would you check?</summary>

Overlapping or persistent regressors (use HAC or non-overlapping samples; classical $t$ can be inflated by more than 2x). Volatility clustering (heteroskedasticity, use robust SEs). Influential days (leverage, Cook's distance; a few crisis days can drive a coefficient). Look-ahead or omitted factors (bias, which no SE fixes). And how many specifications were tried before this one.
</details>

<details>
<summary>17. Is multicollinearity a violation of the OLS assumptions?</summary>

Only perfect collinearity violates full rank. Near collinearity keeps OLS unbiased with honest standard errors; it just makes them large ($\text{VIF} = 1/(1-R_j^2)$). The data genuinely cannot separate the two effects. Coverage stayed at 95% even at $\rho = 0.99$.
</details>

<details>
<summary>18. Cauchy errors gave about 95% coverage. Does that mean OLS works with Cauchy errors?</summary>

No. The $t$-statistic is self-normalising: an extreme error inflates both $\hat\beta - \beta$ and $s$, so the ratio behaves. But $\hat\beta$ barely concentrates as $n$ grows, so the intervals are honest only by being very wide. Coverage measures calibration, not usefulness. A robust estimator (least absolute deviations, Huber) would be far more precise.
</details>

<details>
<summary>19. Why 5,000 replications? How precise is a simulated coverage rate?</summary>

A coverage estimate is a proportion, so its Monte Carlo standard error is $\sqrt{p(1-p)/R} \approx 0.003$ at $p = 0.95$, $R = 5000$. Anything in roughly [94.4%, 95.6%] is indistinguishable from 95%. That is why the figures shade a band around 95% rather than a single line.
</details>

<details>
<summary>20. How did you make sure your implementation is correct?</summary>

Every estimator (all four solvers, classical, HC0 to HC3, Newey-West, F-test, leverage, studentised residuals, Cook's distance) is tested against `statsmodels` to a relative tolerance of $10^{-10}$. The projection identities are tested directly. And in the baseline experiment, simulated variances match closed-form theory ($1/(n-3)$ for the slope, $\text{VIF}$ for collinearity, the AR(1) inflation factor), which is a check on the simulation engine itself.
</details>
