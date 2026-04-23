# Expressiveness of MLP and CNN

> From: https://chatgpt.com/c/69e98b92-3a90-83ea-a239-1b9ba02c3e12

# you asked

survey all the papers about the expressiveness of mlp and CNN, consider

1. universality of mlp

2. essential improvement of multiple layer

3. mlp fits finite step function, finite step boolean function

4. cnn expressiveness improvement, for some idea theoretical case (may consider locality)

before survey, please first extend this list comprehensively

---

# chatgpt response

I’m going to turn your four bullets into a fuller survey taxonomy first, so the later paper review has a clean scope. Do you want the eventual survey to lean more toward exact representability theorems or toward approximation-rate / depth-separation results? I’ll cover both unless you want a narrower emphasis.

I’ve got the main MLP pillars pinned down: classic universality, width-vs-depth, linear-region/trajectory notions, and approximation-rate results. Next I’m filling the CNN-specific side—universality, locality/weight sharing, and depth-separation results that are genuinely architectural rather than just inherited from generic ReLU nets.

The CNN side is filling out into three distinct strands rather than one: pure approximation/universality, architectural separations from locality/weight sharing/overlap, and symmetry-constrained universality (shift/group equivariance). That distinction will matter later because some “CNN expressiveness” results are really approximation theorems, while others are comparative separations against LCNs/FCNs or against shallow/non-overlapping convnets.

Good direction. Before surveying papers, I would **refactor and expand your scope** like this.

Your original item **3** should be split into three different questions:  
(1) step / indicator / piecewise-constant functions on a continuous domain,  
(2) Boolean functions on the discrete cube $\{0,1\}^d$, and  
(3) finite-sample interpolation or memorization. These are related, but theorems and techniques differ a lot. In particular, ReLU networks are continuous on $\mathbb{R}^d$, so continuous-domain step functions are usually an **approximation** question, while Boolean-cube results can be **exact representation** questions. cit([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v70/safran17a/safran17a.pdf?utm_source=chatgpt.com))r original item **4** should also be split into several lines:  
(1) universality of CNNs,  
(2) quantitative approximation rates of CNNs,  
(3) genuine architectural advantages from locality / weight sharing / overlap / pooling,  
and (4) symmetry-constrained universality such as shift-equivariant or group-equivariant settings. Those strands are distinct in the literature. cit([arXiv](https://arxiv.org/abs/1805.10769?utm_source=chatgpt.com))e is the **comprehensive extended list** I’d use.

1. **What notion of expressiveness is being studied?**  
Separate: exact representability, universal approximation, quantitative $\varepsilon$-approximation rate, finite-sample interpolation/memorization, lower bounds / separation theorems, and geometric proxies such as linear regions or trajectory length. A serious survey has to keep these notions separate from the start. cite([Pinkus](https://pinkus.net.technion.ac.il/files/2021/02/acta.pdf?utm_source=chatgpt.com))Classical universality of MLPs.**  
This includes Cybenko/Hornik/Leshno-type results: one-hidden-layer universality, what activation assumptions are sufficient, and whether the target class is continuous, measurable, or $L^p$. citetu([Springer](https://link.springer.com/article/10.1007/BF02551274?utm_source=chatgpt.com))nimal architectural conditions for MLP universality.**  
Not just “is MLP universal,” but: what is the minimal width, how depth can compensate for width, and what fails below the threshold. For ReLU nets, bounded-width universality and minimal-width barriers are now a central part of the story. citetu([arXiv](https://arxiv.org/abs/1709.02540?utm_source=chatgpt.com))antitative approximation theory for MLPs.**  
Beyond universality, survey approximation rates for Barron/Fourier-type classes, Sobolev/Besov/Hölder classes, analytic functions, manifold-supported functions, and Kolmogorov-optimal approximation viewpoints. citetu([Springer](https://link.springer.com/article/10.1007/BF00993164?utm_source=chatgpt.com))sential improvement from depth: explicit depth-separation theorems.**  
This is your item 2, but it should explicitly include: Eldan–Shamir type $3$-vs-$2$ layer separation, Telgarsky oscillation / sawtooth constructions, and Safran–Shamir natural-function separations such as balls and radial functions. citeturn45([arXiv](https://arxiv.org/abs/1512.03965?utm_source=chatgpt.com))ial improvement from depth on structured function classes.**  
A different line from explicit hard examples: deep networks can exploit compositional or hierarchical target structure, while shallow networks cannot do so efficiently. This includes the Poggio–Mhaskar compositionality program. citeturn34([arXiv](https://arxiv.org/abs/1905.12882?utm_source=chatgpt.com))representability and sharp dependence on depth.**  
Besides approximation rates, there is a line asking whether increasing depth strictly enlarges the class of exactly representable functions for fixed width/architecture, and how sharply that dependence can be characterized. citeturn32([NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2020/file/78f7d96ea21ccae89a7b581295f34135-Paper.pdf?utm_source=chatgpt.com))indicator, and piecewise-constant functions.**  
This should be its own section: indicators of intervals/balls/polytopes, piecewise-constant functions, discontinuous targets, and whether one has exact representation, weak representation, or only approximation. citeturn43([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v70/safran17a.html?utm_source=chatgpt.com))n functions and circuit-complexity connections.**  
This should be distinct from continuous-domain step functions. Include threshold circuits, parity/majority/disjointness/inner product, exact representation of Boolean functions, and lower bounds for shallow threshold/ReLU combinations. citeturn18([cdam.lse.ac.uk](https://www.cdam.lse.ac.uk/Reports/Files/cdam-2003-01.pdf?utm_source=chatgpt.com))e-sample expressivity / memorization.**  
Also distinct from Boolean-function theory. This asks how many neurons/parameters/layers are needed to interpolate arbitrary finite datasets, and whether depth reduces that cost. citeturn32([arXiv](https://arxiv.org/abs/1810.07770?utm_source=chatgpt.com))–depth–parameter tradeoffs.**  
A proper survey should track which complexity measure each paper uses: neurons, nonzero parameters, width, depth, weight magnitude, or encodability of weights. These tradeoffs are often the real content of “expressiveness.” citeturn86([arXiv](https://arxiv.org/abs/1709.02540?utm_source=chatgpt.com))try/topology proxies for MLP expressiveness.**  
Include linear-region counting, oscillation counts, trajectory length, and related geometric/topological proxies. These do not equal approximation power, but they are a major subliterature. citeturn10([arXiv](https://arxiv.org/abs/1402.1869?utm_source=chatgpt.com))rsality of CNNs.**  
This is the first CNN section: deep CNN universality in unconstrained settings, quantitative universality, and the role of depth/downsampling/channels in making CNNs universal. citeturn79([arXiv](https://arxiv.org/abs/1805.10769?utm_source=chatgpt.com))ximation advantages of CNNs on structured targets.**  
Not only universal approximation, but cases where CNNs provably do better on radial/composite/sparse-structure functions, or where downsampling and multichanneling matter quantitatively. citeturn79([arXiv](https://arxiv.org/abs/2107.00896?utm_source=chatgpt.com)) convolutional universality under shift symmetry.**  
Separate from generic CNN universality: survey results for fully convolutional nets approximating shift-invariant or shift-equivariant functions, including necessity of channel and kernel-size conditions. citeturn42([arXiv](https://arxiv.org/abs/2211.14047?utm_source=chatgpt.com))ity vs weight sharing vs fully connectedness.**  
This deserves a dedicated CNN section. Compare CNNs, LCNs, and FCNs, and separate which gains come from locality and which from sharing. Some recent work gives actual separations rather than vague inductive-bias claims. citeturn27([arXiv](https://arxiv.org/abs/2305.08404?utm_source=chatgpt.com))t of overlapping receptive fields.**  
Overlaps are not a cosmetic architectural detail; there are theoretical results showing exponential expressive gains from overlapping local receptive fields. citeturn56([arXiv](https://arxiv.org/abs/1703.02065?utm_source=chatgpt.com))t of pooling choice.**  
Max vs average pooling should be treated separately. In the tensor-decomposition line of work, universality and depth efficiency can depend strongly on this choice. citeturn56([arXiv](https://arxiv.org/abs/1603.00162?utm_source=chatgpt.com))r/rank viewpoints for CNN expressiveness.**  
This is a major branch: convolutional arithmetic circuits, generalized tensor decompositions, separation rank, and complete/incomplete depth efficiency. citeturn56([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v49/cohen16.html?utm_source=chatgpt.com))inear-region complexity.**  
If you want a piecewise-linear expressiveness section for CNNs analogous to MLPs, include results on the number of linear regions for CNNs and per-parameter comparisons to fully connected ReLU nets. citeturn25([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v119/xiong20a.html?utm_source=chatgpt.com))-equivariant / invariant extensions of CNN expressiveness.**  
Once you move beyond standard translation-equivariant CNNs, there is a broader universality literature for invariant/equivariant architectures, including cases where higher-order tensors are necessary. citeturn51([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v97/maron19a.html?utm_source=chatgpt.com))ximation power vs separation power in equivariant architectures.**  
This is newer and important: some recent work shows that architectures with the same separation power can still differ in approximation power, so “distinguishing orbits” is not the whole story. citeturn51([arXiv](https://arxiv.org/abs/2506.02293?utm_source=chatgpt.com))dary but adjacent topics that should be clearly labeled, not mixed in.**  
Some papers are really about sample complexity, generalization, or trainability under locality/equivariance rather than pure representational expressiveness. They are relevant, but I would keep them in a separate section or appendix. citeturn27([arXiv](https://arxiv.org/abs/2403.15707?utm_source=chatgpt.com))very paper, record these metadata.**  
Architecture, activation, domain ($\mathbb R^d$ vs $\{0,1\}^d$), target function class, notion of approximation/error norm, complexity measure, whether the result is upper/lower/separation, and whether it is constructive.

So your revised high-level list is not just 4 items. It is closer to this:

- MLP universality  
- minimal width/depth for universality  
- quantitative MLP approximation rates  
- essential gain from depth: explicit separations  
- essential gain from depth: compositional targets  
- exact representation vs approximation of step/indicator/piecewise-constant functions  
- Boolean functions / threshold-circuit viewpoint  
- finite-sample interpolation / memorization  
- geometric proxies: linear regions / trajectory length  
- CNN universality  
- CNN approximation advantages on structured targets  
- fully convolutional shift-equivariant universality  
- locality vs weight sharing vs FCN/LCN  
- overlapping receptive fields  
- pooling effects  
- tensor/rank formulations of CNN expressiveness  
- CNN linear-region complexity  
- group-equivariant / invariant universality  
- separation power vs approximation power in equivariant nets

That is the scope I would use for the actual survey. Next I’d turn this into a **paper-by-paper survey map** with seminal papers, main theorem statements, and how the results relate.