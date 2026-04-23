# Generalization of deep learning

1. exprimental results

2. theoretical results (must include these using stationary phase to get the Hessian.)

3. implicit low dimensional effect, the pruning, implicit regularization

4. shape of loss landscope

5. neural tangent kernel and mean field theory

before survey, please first extend this list comprehensively

I’m building a survey-ready outline first, then I’ll turn it into a tighter map of the literature. Unless you want a narrower slice such as only deep nets or only post-2018 work, I’ll cover the classic flat-minima line together with newer Hessian, PAC-Bayes, NTK/mean-field, pruning, and low-dimensionality results.

I’ve got the core backbone now: early flat-minima claims, the reparameterization critique, PAC-Bayes/normalized-sharpness fixes, SGD-noise dynamics, and the tiny-subspace/loss-landscape papers. I’m filling in the pruning, low-dimensionality, and NTK/mean-field branches so the final outline is actually survey-complete rather than just a reading list.

A complete survey should be broader than your current five bullets. I would expand it into the following survey scaffold.

1.  **Preliminaries: what exactly is a “flat minimum”?**  
    Define the competing notions up front: λmax⁡(H)\\lambda\_{\\max}(H)λmax​(H), tr(H)\\mathrm{tr}(H)tr(H), log⁡det⁡H\\log\\det HlogdetH, perturbation-based sharpness, local entropy, volume-based flatness, normalized/adaptive sharpness, and relative flatness; also separate **parameter-space**, **representation-space**, and **function-space** flatness, because raw Hessian sharpness is not invariant under simple ReLU rescalings. [arXiv+3MIT Direct+3Proceedings of Machine Learning Research+3](https://direct.mit.edu/neco/article/9/1/1/6027/Flat-Minima?utm_source=chatgpt.com)
    
2.  **Experimental phenomenology.**  
    Keep your “experimental results” section, but make it broader: small-batch vs. large-batch training, learning-rate/batch-size effects, optimizer dependence, trajectory sharpening-then-flattening, architecture dependence, and the role of normalization and skip connections. This is where Keskar, Jastrzębski, Entropy-SGD, SAM, and SWA naturally sit. [arXiv+4OpenReview+4OpenReview+4](https://openreview.net/forum?id=H1oyRlYgg&utm_source=chatgpt.com)
    
3.  **How flatness is actually measured in practice.**  
    Add a measurement section: top Hessian eigenvalue, trace, spectral density, outlier spikes vs. bulk, 1D/2D loss interpolation, filter normalization, and practical Hessian tools. This prevents the survey from mixing geometry with visualization artifacts. [Proceedings of Machine Learning Research+2arXiv+2](https://proceedings.mlr.press/v97/ghorbani19b.html?utm_source=chatgpt.com)
    
4.  **Theoretical core I: Bayesian evidence via stationary phase / Laplace asymptotics.**  
    This should be a dedicated section because you explicitly want the Hessian from stationary phase. Around a regular nondegenerate minimum θ\\\*\\theta^\\\*θ\\\*, the evidence/posterior integral is expanded by stationary phase (equivalently Laplace’s method), and curvature enters through an Occam factor involving log⁡det⁡H(θ\\\*)\\log\\det H(\\theta^\\\*)logdetH(θ\\\*). This is the cleanest classical route from local geometry to model selection and generalization. [MIT Direct+2inference.org.uk+2](https://direct.mit.edu/neco/article-pdf/4/3/415/812340/neco.1992.4.3.415.pdf?utm_source=chatgpt.com)
    
5.  **Theoretical core II: PAC-Bayes, perturbation stability, and Hessian-based bounds.**  
    Separate this from stationary phase. Here the relevant literature links generalization to local perturbation robustness, Hessian and Hessian-Lipschitz terms, parameter scales, and noise stability; this also includes fine-tuning results where Hessian-based distances help explain overfitting. [Proceedings of Machine Learning Research+3OpenReview+3OpenReview+3](https://openreview.net/forum?id=BJxOHs0cKm&utm_source=chatgpt.com)
    
6.  **Singular-model caveat: when Hessian-based local theory is not enough.**  
    Deep networks are typically not regular statistical models globally; singularities and degenerate Fisher/Hessian structure are common, so pure Laplace intuition can fail. A serious survey should therefore include singular learning theory, WBIC/RLCT, and explain that Hessian arguments are often only local approximations. [arXiv+2机器学习研究杂志+2](https://arxiv.org/abs/1208.6338?utm_source=chatgpt.com)
    
7.  **Implicit regularization of SGD and optimizer-induced bias.**  
    Your “implicit regularization” item should be expanded to include noise scale, escape from sharp directions, local-entropy viewpoints, the learning-rate/batch-size ratio, and the distinction between SGD, Adam, SAM, ASAM, EMA, and SWA. Recent work on SAM also suggests that its flatness bias is especially strong late in training. [arXiv+3OpenReview+3OpenReview+3](https://openreview.net/forum?id=SkgEaj05t7&utm_source=chatgpt.com)
    
8.  **Implicit low-dimensional effect.**  
    This deserves its own major section, not a subsection. Include tiny dominant Hessian subspaces, intrinsic dimension of objective landscapes, low-dimensional feature learning, effective-rank viewpoints, and low-dimensional training/fine-tuning subspaces. This is one of the most important bridges between flatness, pruning, and modern PEFT methods. [OpenReview+3OpenReview+3OpenReview+3](https://openreview.net/forum?id=ByeTHsAqtX&utm_source=chatgpt.com)
    
9.  **Pruning, sparsity, compression, and MDL-style interpretations.**  
    Your pruning point should be widened to cover lottery tickets, PAC-Bayes analyses of winning tickets, Hessian-aware or directional pruning, compression-based generalization bounds, and recent methods that explicitly search for subnetworks that are both sparse and flat. [arXiv+4OpenReview+4arXiv+4](https://openreview.net/forum?id=rJl-b3RcF7&utm_source=chatgpt.com)
    
10.  **Shape and topology of the loss landscape.**  
     Keep your “shape of loss landscape” section, but make it global rather than only local: mode connectivity, low-loss manifolds, simplices/volumes of solutions, asymmetric valleys, connectedness, and how architecture changes landscape geometry. [arXiv+3arXiv+3Proceedings of Machine Learning Research+3](https://arxiv.org/pdf/1802.10026?utm_source=chatgpt.com)
     
11.  **Neural Tangent Kernel / lazy-training regime.**  
     Keep NTK as a distinct regime. In this limit, training is described in function space by kernel gradient descent with an approximately frozen tangent kernel, so generalization is explained less by finding a special weight-space basin and more by kernel bias and data geometry. [NeurIPS 会议记录+2arXiv+2](https://proceedings.neurips.cc/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html?utm_source=chatgpt.com)
     
12.  **Mean-field theory / feature-learning regime.**  
     Keep mean field separate from NTK. This is the PDE/distributional-dynamics line, where parameters evolve nontrivially and the kernel itself changes during training. This is the right place for two-layer landscape results, mean-field limits, and feature learning beyond lazy training. [arXiv+3arXiv+3arXiv+3](https://arxiv.org/abs/1804.06561?utm_source=chatgpt.com)
     
13.  **Finite-width bridges between NTK and mean field.**  
     Add an explicit bridge section: crossover regimes, finite-width corrections, when feature learning appears, and how this changes the relevance of flatness. This is also where modern fine-tuning and LoRA-style scale-invariant parameterizations fit. [arXiv+2Proceedings of Machine Learning Research+2](https://arxiv.org/abs/1906.08034?utm_source=chatgpt.com)
     
14.  **Robustness, calibration, distribution shift, and uncertainty.**  
     Flatness is often claimed to help not only test error but also robustness, calibration, and uncertainty; these claims should be surveyed separately rather than mixed into plain generalization. Relative-flatness/feature-robustness work and modern Laplace methods both belong here. [OpenReview+2NeurIPS 会议记录+2](https://openreview.net/pdf?id=sygvo7ctb_&utm_source=chatgpt.com)
     
15.  **Negative results, counterexamples, and recent reappraisals.**  
     This section is essential. The survey should not present flatness as a settled explanation. It needs the reparameterization critique, large-scale evidence that many sharpness measures are fragile, recent results where sharper solutions can generalize as well or better, and newer proposals such as soft-rank, Rényi sharpness, or function-centric sharpness. [OpenReview+5Proceedings of Machine Learning Research+5Proceedings of Machine Learning Research+5](https://proceedings.mlr.press/v70/dinh17b.html?utm_source=chatgpt.com)
     
16.  **Benchmarking protocol and survey methodology.**  
     Add one final section on *how to compare papers fairly*: dataset choice, architecture class, perturbation radius, normalization convention, whether flatness is measured before or after weight rescaling, and whether the claim is correlation, bound, or causal mechanism. Large-scale meta-studies show this matters a lot. [arXiv+1](https://arxiv.org/abs/1912.02178?utm_source=chatgpt.com)

The most important additions beyond your original list are: **(i)** a definitions/invariance section, **(ii)** a dedicated **stationary-phase/Laplace** section, **(iii)** a **singular-model caveat** section, **(iv)** a **negative-results** section, and **(v)** a **benchmarking protocol** section.

Next, the survey itself should probably be written in this order: **definitions → experiments → local theory (stationary phase / PAC-Bayes) → SGD implicit bias → low-dimensionality/pruning → global landscape → NTK vs mean field → counterexamples/open problems**.