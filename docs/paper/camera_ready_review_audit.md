# UKCI 2026 Camera-Ready Review Audit

This audit compares the accepted CMT submission (`docs/main (1).pdf`), the
post-acceptance draft, and the final camera-ready manuscript. “Resolved” means
the requested correction or evidence is now in the paper. “Clarified” means the
paper now states the method's limitation accurately. “Deferred” identifies a
new experiment or dataset that was not added and is explicitly left as future
work.

| Status | Review point | Camera-ready response |
|---|---|---|
| Resolved | The equally weighted three-quantile loss is not aggregate cost-asymmetric. | Corrected throughout. The title and contribution now say *multi-quantile*; Section 2.4 explains that the \(q=0.1\) and \(q=0.9\) asymmetries oppose each other and limits the cost claim to the upper branch. |
| Resolved | Quantiles were not evaluated directly. | Added Table 2 with horizon-wise 80% coverage, interval width, pinball loss at \(q=0.1,0.5,0.9\), WIS, \(q=0.9\) miss rate, and crossing frequency. The source values are in `results/forecasting/table_quantile_metrics.csv`. |
| Resolved | Point recalibration does not update the LP scenarios. | Added separate notation for the recalibrated point forecast and raw quantile heads. Table 1 evaluates point forecasts; Table 2 and every LP use the unchanged raw heads. |
| Resolved | Scenario weights are representative rather than calibrated probabilities. | The manuscript now calls them design weights and the outputs stress scenarios. “Expected unmet” is replaced by scenario-weighted unmet \(U_\pi\), explicitly not a real-world expectation. |
| Clarified | Co-monotone \(q=0.9\) scenarios are stylised and conservative. | This limitation is stated in the model and conclusion. Direct calibration results show that the high head is conservative. A calibrated joint residual model was not added. |
| Resolved | Planning parameters and capacity assumptions need justification. | Added the rationale and made clear that the values are stress-test settings, not NHS standards. Budget, travel, and transfer caps are sensitivity-tested. |
| Clarified | The ablation does not prove that an identifiable physics prior caused the gain. | Reworded as evidence that the structured PINN-derived representation adds signal; the paper no longer claims to separate physical structure from representation capacity. |
| Deferred | Add a matched-capacity unconstrained time-feature control. | No unperformed experiment is claimed. It is listed as future work, and the present interpretation is limited accordingly. |
| Resolved | Optimisation evidence was limited and policies often tied. | Retained the transparent unconstrained-transfer tie, the all-origin analysis, and the sparse-network/transfer-cap experiments showing when placement affects coverage. |
| Resolved | Parameters and scenario probabilities need operational estimation guidance. | Added deployment inputs: staffed-bed inventories, expansion and transfer limits, network travel times, and joint forecast residuals. |
| Resolved | Sections 2 and 3 were dense and difficult to follow. | Tightened and consolidated the forecasting and optimisation prose, removed unnecessary paragraph breaks, and introduced explicit notation separating point recalibration from the raw LP scenario heads. |
| Resolved | Future directions should be explicit. | The conclusion now lists quantile recalibration, correlated scenarios, alternative loss weightings, a matched-capacity control, other health systems, and trust-level scaling. |
| Deferred | Validate on more varied outbreaks, datasets, and health systems. | No additional dataset was introduced after acceptance. Cross-system and trust-level evaluation is stated as future work. |
| Deferred | Replace stylised uncertainty with more realistic joint uncertainty. | The current scenario set is labelled a stress-test construction. Correlated, calibrated residual scenarios are stated as future work. |

The final PDF compiles to 12 pages with resolved references and no LaTeX
overfull/underfull box warnings. The Springer template class, bibliography
style, fonts, margins, and page geometry were not changed.
