# Development Roadmap

## Version 1
- [ ] Ledger system
  - [x] Ledger
  - [ ] BigQuery validation
  - [ ] Unit tests + explanatory notebooks
- [x] EarlySign Standard Schema manifest (ES3)
  - [x] Code generator
  - [x] Base Schema
- [ ] EarlySign development framework (CQRS-style)
  - [ ] Writers
  - [ ] Projectors
  - [ ] Trace management
  - [ ] Templates
    - [ ] General backtest support

- [ ] GST
  - [ ] ES3 Schema
  - [ ] Protocol Designer
  - [ ] AB-test template
  - [ ] Behavior Test: Designer
  - [ ] Behavior Test: Execution
  - [ ] Examples with OCE bench (backtest + online usages)

- [ ] MAMS
  - [ ] ES3 Schema
  - [ ] Protocol Designer
  - [ ] ABCD-test template
  - [ ] Examples with OCE bench (backtest + online usages)
  - References:
    - Whitehead, J., & Jaki, T. (2009). One- and two-stage design proposals for a phase II trial comparing three active treatments with control using an ordered categorical endpoint. Statistics in Medicine, 28(5), 828–847. https://doi.org/10.1002/sim.3508
    - Magirr, D., Jaki, T., & Whitehead, J. (2012). A generalized Dunnett test for multi-arm multi-stage clinical studies with treatment selection. Biometrika, 99(2), 494–501.
    - Jaki, T., & Magirr, D. (2013). Considerations on covariates and endpoints in multi-arm multi-stage clinical trials selecting all promising treatments. Statistics in Medicine, 32(7), 1150–1163. https://doi.org/10.1002/sim.5669
    - Magirr, D., Stallard, N., & Jaki, T. (2014). Flexible sequential designs for multi-arm clinical trials. Statistics in Medicine, 33(19), 3269–3279. https://doi.org/10.1002/sim.6183
    - Jaki, T., Pallmann, P., & Magirr, D. (2019). The R package MAMS for designing multi-arm multi-stage clinical trials. Journal of Statistical Software, 88, 1–25. https://doi.org/10.18637/jss.v088.i04
    - Scharfstein, D. O., Tsiatis, A. A., & Robins, J. M. (1997). Semiparametric efficiency and its implication on the design and analysis of group-sequential studies. Journal of the American Statistical Association, 92(440), 1342–1350. https://doi.org/10.1080/01621459.1997.10473655

- [ ] Repeated Confidence Interval
  - [ ] ES3 Schema
  - [ ] Protocol Designer
  - [ ] AB-test template
  - [ ] Examples with OCE bench (backtest + online usages)

- [ ] AVI
  - [ ] ES3 Schema
  - [ ] 1-sample Monitoring template
  - [ ] 2-sample Regression Monitoring template
  - [ ] Examples with OCE bench (backtest + online usages)

- [ ] CUSUM
  - [ ] ES3 Schema
  - [ ] 1-sample Monitoring template
  - [ ] Examples with OCE bench (backtest + online usages)

- [ ] Documentation
  - [x] Auto-generated API docs
  - [ ] Explanations
  - [ ] ADRs with dropped ideas

- [ ] Quality assurance
  - [x] mypy
  - [x] linting & auto-formatting
  - [ ] doctests
  - [ ] CI/CD
  - [ ] Test coverage

## Version 1.1

- [ ] Lindon, M., Sanden, C., & Shirikian, V. (2022). Rapid regression detection in software deployments through sequential testing. Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 3336–3346. https://doi.org/10.1145/3534678.3539099
- [ ] Zhao, Z., Liu, M., & Deb, A. (2018). Safely and quickly deploying new features with a staged rollout framework using sequential test and adaptive experimental design. 2018 3rd International Conference on Computational Intelligence and Applications (ICCIA), 59–70. https://doi.org/10.1109/ICCIA.2018.00019
- [ ] Johari, R., Koomen, P., Pekelis, L., & Walsh, D. (2022). Always valid inference: Continuous monitoring of A/B tests. Operations Research, 70(3), 1806–1821. https://doi.org/10.1287/opre.2021.2135
