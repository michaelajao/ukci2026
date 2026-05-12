# 05 — Open Questions and Deferred Investigations

A short list of unresolved methodological concerns that we are choosing to
*ship as-is* for UKCI 2026 (12-page conference paper, 31 May deadline) but
that should be revisited for the journal-extension version of this work.

Each entry has: (a) what we observed, (b) why it's a concern,
(c) what would resolve it, (d) where it currently appears in the paper.

---

## Q1. Joint decline of $\gamma_{c,r}(t)$ and $\delta_{c,r}(t)$ in the PINN parameter network

**Observed.** The trained per-region PINN-SEIRD produces time-varying ICU
recovery rate $\gamma_{c,r}(t)$ and ICU mortality rate $\delta_{c,r}(t)$
that *both* decline monotonically across all seven NHS regions through the
2020–2022 period (see `figures/fig_pinn_params.png`).

**Concern.** A monotonic decline in $\delta_c$ is biologically plausible
and aligns with the well-documented reduction in COVID-19 critical-care
lethality from Alpha through Omicron. A simultaneous decline in $\gamma_c$
is harder to defend: it implies longer ICU stays for survivors, which is
not strongly supported by the literature for the Alpha→Omicron transition.
The more likely explanation is that the PINN is *trading off* the two
out-rates of the C-compartment to fit the observed mechanical-ventilation
bed count, because both parameters appear additively in
$\dot C_r = \phi d_H H_r - (\gamma_{c,r} + \delta_{c,r}) C_r$ and are not
individually identifiable from $C_r$ alone — distinguishing them requires
separate observations of ICU recoveries vs ICU deaths, which we do not have
at regional resolution.

**What would resolve it.**
1. Fit the PINN with an *additional* data term anchoring the cumulative
   ICU-death series ($D_r$ compartment) against published NHS England
   COVID-death-in-hospital records by region. This would disambiguate the
   two rates.
2. Compare $\delta_c(t)$ trajectories against the time-resolved
   case-fatality ratio in ICU admissions from ICNARC reports.
3. Add an identifiability sanity check: re-fit with $\gamma_c$ frozen at a
   plausible literature value (e.g., $1/14$ day$^{-1}$ in line with the
   median ICU stay among survivors) and see whether $\delta_c$ alone can
   carry the trajectory.

**Where it appears in the paper.** §3 currently refers readers to
"supplementary material" for the learned-parameter plot. We do *not*
explicitly claim biological interpretability of the $\gamma_c$ trajectory
in the body text. The load-bearing evidence for the parameter features is
the ablation row "w/o PINN parameter features" in Table 1, which inflates
$h=14$ RMSE almost two-fold; that claim is robust to whether the
individual parameters are biologically interpretable in isolation.

---

## Q2. London's $\beta_r \approx 0.75$ is ~2× any other region

**Observed.** The transmission-rate parameter $\beta_r(t)$ learned by the
PINN ranges over $\approx 0.30$–$0.45$ for six regions, but London is
consistently around $\approx 0.75$ — twice the second-highest value.

**Concern.** Two possible interpretations.
1. **Real signal.** London genuinely had higher transmission for the
   period due to population density, mobility, and the early-Alpha
   geographic origin of the lineage. $R_t$ estimates from PHE/UKHSA do
   show London peaking earlier and higher than other regions in
   Alpha/Delta waves.
2. **Model artefact.** $\beta_r$ is the only multiplicative scale on the
   $S \cdot I$ term in the ODE; if the PINN cannot reduce a London-specific
   model mismatch (e.g., commuter-flow externalities, super-spreader
   structure unaccounted for in mean-field SEIRD) via the other learnable
   parameters, it will absorb the residual into $\beta$, inflating it.

**What would resolve it.** Cross-validate $\beta_r(t)$ against
PHE/UKHSA's published regional $R_t$ time series (recompute
$R_{0,r}(t) \approx \beta_r(t)/d_s$, given $1/d_s = 4$ days infectious
period) for each region. If our estimates fall within reported confidence
intervals, the London signal is real; if not, we have a model-mismatch
flag.

**Where it appears in the paper.** Not discussed in the body text.
Reviewers may probe if the parameter-trajectory figure makes it into
supplementary material. The fall-back response — that the parameters are
treated as features for the downstream GRU rather than as
epidemiologically interpreted estimates — is honest but worth
strengthening with the cross-check above for the journal version.

---

*Both Q1 and Q2 are decoupled from the paper's load-bearing claims (Table 1
ablations, decision-aware forecasting wins, allocation-pipeline
contributions). They are documented here for the journal-extension review.*
