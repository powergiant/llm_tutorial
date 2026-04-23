# Scaling limit of gradient descent

1. the SDE limit of SGD

2. the explanation of linear scaling law of lr and batch (also survey explanation that do not use SDE limit)

3. consider more complicated result of Adam

before survey, please first extend this list comprehensively

I’ll first turn your 3-item list into a fuller survey map, anchored in the main theory threads and recent optimizer-specific results. I’m checking both classical diffusion-limit papers and newer work on batch-size/lr scaling and adaptive methods; if you want, I can bias the eventual survey toward either rigorous probability/analysis or DL practice papers.

I’ve got the main classic threads pinned down: diffusion/SDE approximations of SGD, batch–learning-rate scaling via noise-scale arguments, and newer continuous-time analyses for Adam. I’m now filling gaps the initial list misses—especially rigorous-vs-heuristic distinctions, finite-step corrections, stability/edge-of-stability, and recent large-scale pretraining papers that revise Adam scaling rules.

Yes. Before the actual survey, I would widen your 3 bullets into a survey map like this. A literally exhaustive bibliography will keep moving, but this is the right near-complete structure.

1.  **First separate the different meanings of “scaling limit.”** You want to distinguish at least: vanishing-step continuous-time limits, constant-step stationary-law limits, batch-size/critical-batch scaling, joint learning-rate–batch scaling, and high-dimensional / infinite-width / pretraining limits. Many papers talk about “the limit” while proving only one of these. [arXiv+3机器学习研究杂志+3机器学习研究杂志+3](https://www.jmlr.org/papers/v25/23-0237.html)
    
2.  **Rigorous diffusion / SDE limits of plain SGD.** This section should include the SME line, weak convergence of SGD to diffusion processes, nonconvex escape/metastability results, and papers comparing ODE and SDE approximations for non-infinitesimal step sizes. [机器学习研究杂志+3机器学习研究杂志+3国际出版社档案+3](https://jmlr.org/papers/v20/17-526.html)
    
3.  **Constant-step SGD as a Markov chain with a stationary law.** This is a distinct thread from transient diffusion limits: Mandt-style “SGD as approximate Bayesian inference,” stationary covariance analyses, and more recent long-run / large-deviations descriptions of where constant-step SGD spends its time. [机器学习研究杂志+1](https://www.jmlr.org/papers/volume18/17-214/17-214.pdf)
    
4.  **Beyond Gaussian diffusion.** Add papers where the Gaussian SDE is corrected or replaced: Hessian-aware continuous-time models, anisotropic-noise corrections, and heavy-tailed / infinite-variance regimes where Lévy-type or non-Gaussian limits become relevant. [OpenReview+2arXiv+2](https://openreview.net/pdf?id=GH5XHcHTS5)
    
5.  **Finite-step and discrete-time corrections.** This deserves its own section because many practical scaling rules live outside the infinitesimal-step regime: modified-loss / backward-error analyses, phase-space or underdamped formulations, anomalous diffusion, and explicit demonstrations that finite-η\\etaη SGD can differ sharply from its naive continuous-time approximation. [arXiv+2arXiv+2](https://arxiv.org/abs/2012.03636)
    
6.  **Sampling protocol and noise geometry.** Separate with-replacement minibatching from without-replacement / random reshuffling, and separate isotropic-noise heuristics from structured-noise results such as alignment with sharp directions. These distinctions matter for both SDE validity and scaling-law claims. [arXiv+2arXiv+2](https://arxiv.org/abs/2312.16143)
    
7.  **SDE-based explanations of the linear learning-rate / batch-size rule for SGD.** This should include the Smith–Le noise-scale picture, η/B\\eta/Bη/B or ηN/B\\eta N/BηN/B invariance arguments, Bayesian-evidence explanations, and the gradient-noise-scale / critical-batch-size line of work. [OpenReview+3arXiv+3arXiv+3](https://arxiv.org/abs/1710.06451)
    
8.  **Explanations of linear scaling that do not rely on the SDE limit.** Include Goyal et al.’s “kkk small steps versus one large step” heuristic plus warmup, the “train longer / fewer parameter updates” explanation of large-batch gaps, broader data-parallelism studies showing strong workload dependence, and stability-based explanations such as edge-of-stochastic-stability. [arXiv+4arXiv+4arXiv+4](https://arxiv.org/pdf/1706.02677)
    
9.  **Breakdown of linear scaling and the meaning of critical batch size.** The survey should explicitly track when scaling saturates, when warmup/schedules become essential, and how modern pretraining papers revise older CBS heuristics, especially under WSD-style schedules. [arXiv+2arXiv+2](https://arxiv.org/abs/1812.06162)
    
10.  **High-dimensional / mean-field / exact-asymptotic dynamics.** This is a separate mathematical thread: homogenized SGD, exact risk trajectories in high dimensions, infinite-width or infinite-dimensional limits, and recent pretraining work tying CBS behavior to data size via infinite-width theory. [arXiv+2arXiv+2](https://arxiv.org/abs/2205.07069)
     
11.  **Momentum SGD as a bridge between plain SGD and Adam.** Do not fold SGDM into “plain SGD”: the effective learning rate changes, batch-size scaling depends on momentum, and finite-step stationary behavior changes with momentum. Some empirical large-batch rules survive better with momentum than without. [arXiv+2Proceedings of Machine Learning Research+2](https://arxiv.org/abs/1711.00489)
     
12.  **Adaptive methods: RMSprop / Adam / AdamW.** This should be much broader than “the Adam analogue of SGD.” Include rigorous SDE limits and square-root batch-scaling rules, but also Adam-specific implicit bias, instability/loss spikes at scale, gradient-heterogeneity explanations for why Adam beats SGD on transformers, recent nonmonotone “surge” learning-rate–batch behavior, invariant-law / Fokker–Planck analyses, and modern rules that scale not only learning rate but also β\\betaβ’s, ε\\varepsilonε, or weight decay. [arXiv+7OpenReview+7arXiv+7](https://openreview.net/pdf?id=F2mhzjHkQP)
     
13.  **Practical large-batch adaptive optimizers as their own subsection.** LARS and LAMB are not just optimizer footnotes; they are engineering responses to the failure of naive scaling rules and they shaped a large part of the empirical literature on huge batches. [arXiv+1](https://arxiv.org/abs/1708.03888?utm_source=chatgpt.com)
     
14.  **Conflicting explanations and unresolved tensions.** In the final survey, explicitly compare papers saying “noise improves generalization,” papers saying “the apparent large-batch gap is mostly an update-count/optimization artifact,” and papers saying the key mechanism is local stability or structured noise. Also flag papers where high-dimensional exact results rule out some kinds of implicit regularization in the simplified regime studied. [arXiv+4arXiv+4arXiv+4](https://arxiv.org/abs/1710.06451)
     
15.  **A paper-by-paper metadata grid for the eventual survey.** For each paper, record: optimizer, sampling scheme, asymptotic regime, theorem type, what is scaled (η\\etaη, BBB, 1−β1-\\beta1−β, ε\\varepsilonε, weight decay, tokens), whether the claim is transient or stationary, empirical domain (CV / NLP / LLM pretraining), and exactly where the proposed rule fails.

This is the scaffold I’d use for the actual survey. The cleanest full survey would then be organized into three blocks: plain SGD limits, SGD learning-rate/batch scaling laws, and Adam/AdamW generalizations.