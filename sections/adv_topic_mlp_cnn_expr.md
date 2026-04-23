# Expressiveness of MLP and CNN

1. universality of mlp

2. essential improvement of multiple layer

3. mlp fits finite step function, finite step boolean function

4. cnn expressiveness improvement, for some idea theoretical case (may consider locality)

before survey, please first extend this list comprehensively

I’m going to turn your four bullets into a fuller survey taxonomy first, so the later paper review has a clean scope. Do you want the eventual survey to lean more toward exact representability theorems or toward approximation-rate / depth-separation results? I’ll cover both unless you want a narrower emphasis.

I’ve got the main MLP pillars pinned down: classic universality, width-vs-depth, linear-region/trajectory notions, and approximation-rate results. Next I’m filling the CNN-specific side—universality, locality/weight sharing, and depth-separation results that are genuinely architectural rather than just inherited from generic ReLU nets.

The CNN side is filling out into three distinct strands rather than one: pure approximation/universality, architectural separations from locality/weight sharing/overlap, and symmetry-constrained universality (shift/group equivariance). That distinction will matter later because some “CNN expressiveness” results are really approximation theorems, while others are comparative separations against LCNs/FCNs or against shallow/non-overlapping convnets.

Good direction. Before surveying papers, I would **refactor and expand your scope** like this.

Your original item **3** should be split into three different questions:  
(1) step / indicator / piecewise-constant functions on a continuous domain,  
(2) Boolean functions on the discrete cube {0,1}d\\{0,1\\}^d{0,1}d, and  
(3) finite-sample interpolation or memorization. These are related, but theorems and techniques differ a lot. In particular, ReLU networks are continuous on Rd\\mathbb{R}^dRd, so continuous-domain step functions are usually an **approximation** question, while Boolean-cube results can be **exact representation** questions. [arXiv+3Proceedings of Machine Learning Research+3arXiv+3](https://proceedings.mlr.press/v70/safran17a/safran17a.pdf?utm_source=chatgpt.com)

Your original item **4** should also be split into several lines:  
(1) universality of CNNs,  
(2) quantitative approximation rates of CNNs,  
(3) genuine architectural advantages from locality / weight sharing / overlap / pooling,  
and (4) symmetry-constrained universality such as shift-equivariant or group-equivariant settings. Those strands are distinct in the literature. [OpenReview+3arXiv+3arXiv+3](https://arxiv.org/abs/1805.10769?utm_source=chatgpt.com)

Here is the **comprehensive extended list** I’d use.

1.  **What notion of expressiveness is being studied?**  
    Separate: exact representability, universal approximation, quantitative ε\\varepsilonε\-approximation rate, finite-sample interpolation/memorization, lower bounds / separation theorems, and geometric proxies such as linear regions or trajectory length. A serious survey has to keep these notions separate from the start. [arXiv+3Pinkus+3arXiv+3](https://pinkus.net.technion.ac.il/files/2021/02/acta.pdf?utm_source=chatgpt.com)
    
2.  **Classical universality of MLPs.**  
    This includes Cybenko/Hornik/Leshno-type results: one-hidden-layer universality, what activation assumptions are sufficient, and whether the target class is continuous, measurable, or LpL^pLp. [Springer+2科学直通车+2](https://link.springer.com/article/10.1007/BF02551274?utm_source=chatgpt.com)
    
3.  **Minimal architectural conditions for MLP universality.**  
    Not just “is MLP universal,” but: what is the minimal width, how depth can compensate for width, and what fails below the threshold. For ReLU nets, bounded-width universality and minimal-width barriers are now a central part of the story. [arXiv+1](https://arxiv.org/abs/1709.02540?utm_source=chatgpt.com)
    
4.  **Quantitative approximation theory for MLPs.**  
    Beyond universality, survey approximation rates for Barron/Fourier-type classes, Sobolev/Besov/Hölder classes, analytic functions, manifold-supported functions, and Kolmogorov-optimal approximation viewpoints. [arXiv+3Springer+3arXiv+3](https://link.springer.com/article/10.1007/BF00993164?utm_source=chatgpt.com)
    
5.  **Essential improvement from depth: explicit depth-separation theorems.**  
    This is your item 2, but it should explicitly include: Eldan–Shamir type 333\-vs-222 layer separation, Telgarsky oscillation / sawtooth constructions, and Safran–Shamir natural-function separations such as balls and radial functions. [arXiv+2arXiv+2](https://arxiv.org/abs/1512.03965?utm_source=chatgpt.com)
    
6.  **Essential improvement from depth on structured function classes.**  
    A different line from explicit hard examples: deep networks can exploit compositional or hierarchical target structure, while shallow networks cannot do so efficiently. This includes the Poggio–Mhaskar compositionality program. [arXiv+1](https://arxiv.org/abs/1905.12882?utm_source=chatgpt.com)
    
7.  **Exact representability and sharp dependence on depth.**  
    Besides approximation rates, there is a line asking whether increasing depth strictly enlarges the class of exactly representable functions for fixed width/architecture, and how sharply that dependence can be characterized. [NeurIPS Papers+1](https://papers.neurips.cc/paper_files/paper/2020/file/78f7d96ea21ccae89a7b581295f34135-Paper.pdf?utm_source=chatgpt.com)
    
8.  **Step, indicator, and piecewise-constant functions.**  
    This should be its own section: indicators of intervals/balls/polytopes, piecewise-constant functions, discontinuous targets, and whether one has exact representation, weak representation, or only approximation. [Proceedings of Machine Learning Research+1](https://proceedings.mlr.press/v70/safran17a.html?utm_source=chatgpt.com)
    
9.  **Boolean functions and circuit-complexity connections.**  
    This should be distinct from continuous-domain step functions. Include threshold circuits, parity/majority/disjointness/inner product, exact representation of Boolean functions, and lower bounds for shallow threshold/ReLU combinations. [cdam.lse.ac.uk+2arXiv+2](https://www.cdam.lse.ac.uk/Reports/Files/cdam-2003-01.pdf?utm_source=chatgpt.com)
    
10.  **Finite-sample expressivity / memorization.**  
     Also distinct from Boolean-function theory. This asks how many neurons/parameters/layers are needed to interpolate arbitrary finite datasets, and whether depth reduces that cost. [arXiv+2math.uci.edu+2](https://arxiv.org/abs/1810.07770?utm_source=chatgpt.com)
     
11.  **Width–depth–parameter tradeoffs.**  
     A proper survey should track which complexity measure each paper uses: neurons, nonzero parameters, width, depth, weight magnitude, or encodability of weights. These tradeoffs are often the real content of “expressiveness.” [arXiv+2arXiv+2](https://arxiv.org/abs/1709.02540?utm_source=chatgpt.com)
     
12.  **Geometry/topology proxies for MLP expressiveness.**  
     Include linear-region counting, oscillation counts, trajectory length, and related geometric/topological proxies. These do not equal approximation power, but they are a major subliterature. [arXiv+1](https://arxiv.org/abs/1402.1869?utm_source=chatgpt.com)
     
13.  **Universality of CNNs.**  
     This is the first CNN section: deep CNN universality in unconstrained settings, quantitative universality, and the role of depth/downsampling/channels in making CNNs universal. [arXiv+2ACM数字图书馆+2](https://arxiv.org/abs/1805.10769?utm_source=chatgpt.com)
     
14.  **Approximation advantages of CNNs on structured targets.**  
     Not only universal approximation, but cases where CNNs provably do better on radial/composite/sparse-structure functions, or where downsampling and multichanneling matter quantitatively. [arXiv+1](https://arxiv.org/abs/2107.00896?utm_source=chatgpt.com)
     
15.  **Fully convolutional universality under shift symmetry.**  
     Separate from generic CNN universality: survey results for fully convolutional nets approximating shift-invariant or shift-equivariant functions, including necessity of channel and kernel-size conditions. [arXiv+1](https://arxiv.org/abs/2211.14047?utm_source=chatgpt.com)
     
16.  **Locality vs weight sharing vs fully connectedness.**  
     This deserves a dedicated CNN section. Compare CNNs, LCNs, and FCNs, and separate which gains come from locality and which from sharing. Some recent work gives actual separations rather than vague inductive-bias claims. [arXiv+2arXiv+2](https://arxiv.org/abs/2305.08404?utm_source=chatgpt.com)
     
17.  **Effect of overlapping receptive fields.**  
     Overlaps are not a cosmetic architectural detail; there are theoretical results showing exponential expressive gains from overlapping local receptive fields. [arXiv+1](https://arxiv.org/abs/1703.02065?utm_source=chatgpt.com)
     
18.  **Effect of pooling choice.**  
     Max vs average pooling should be treated separately. In the tensor-decomposition line of work, universality and depth efficiency can depend strongly on this choice. [arXiv+1](https://arxiv.org/abs/1603.00162?utm_source=chatgpt.com)
     
19.  **Tensor/rank viewpoints for CNN expressiveness.**  
     This is a major branch: convolutional arithmetic circuits, generalized tensor decompositions, separation rank, and complete/incomplete depth efficiency. [Proceedings of Machine Learning Research+2arXiv+2](https://proceedings.mlr.press/v49/cohen16.html?utm_source=chatgpt.com)
     
20.  **CNN linear-region complexity.**  
     If you want a piecewise-linear expressiveness section for CNNs analogous to MLPs, include results on the number of linear regions for CNNs and per-parameter comparisons to fully connected ReLU nets. [Proceedings of Machine Learning Research+1](https://proceedings.mlr.press/v119/xiong20a.html?utm_source=chatgpt.com)
     
21.  **Group-equivariant / invariant extensions of CNN expressiveness.**  
     Once you move beyond standard translation-equivariant CNNs, there is a broader universality literature for invariant/equivariant architectures, including cases where higher-order tensors are necessary. [Proceedings of Machine Learning Research+2Proceedings of Machine Learning Research+2](https://proceedings.mlr.press/v97/maron19a.html?utm_source=chatgpt.com)
     
22.  **Approximation power vs separation power in equivariant architectures.**  
     This is newer and important: some recent work shows that architectures with the same separation power can still differ in approximation power, so “distinguishing orbits” is not the whole story. [arXiv](https://arxiv.org/abs/2506.02293?utm_source=chatgpt.com)
     
23.  **Secondary but adjacent topics that should be clearly labeled, not mixed in.**  
     Some papers are really about sample complexity, generalization, or trainability under locality/equivariance rather than pure representational expressiveness. They are relevant, but I would keep them in a separate section or appendix. [arXiv+2arXiv+2](https://arxiv.org/abs/2403.15707?utm_source=chatgpt.com)
     
24.  **For every paper, record these metadata.**  
     Architecture, activation, domain (Rd\\mathbb R^dRd vs {0,1}d\\{0,1\\}^d{0,1}d), target function class, notion of approximation/error norm, complexity measure, whether the result is upper/lower/separation, and whether it is constructive.

So your revised high-level list is not just 4 items. It is closer to this:

*   MLP universality
    
*   minimal width/depth for universality
    
*   quantitative MLP approximation rates
    
*   essential gain from depth: explicit separations
    
*   essential gain from depth: compositional targets
    
*   exact representation vs approximation of step/indicator/piecewise-constant functions
    
*   Boolean functions / threshold-circuit viewpoint
    
*   finite-sample interpolation / memorization
    
*   geometric proxies: linear regions / trajectory length
    
*   CNN universality
    
*   CNN approximation advantages on structured targets
    
*   fully convolutional shift-equivariant universality
    
*   locality vs weight sharing vs FCN/LCN
    
*   overlapping receptive fields
    
*   pooling effects
    
*   tensor/rank formulations of CNN expressiveness
    
*   CNN linear-region complexity
    
*   group-equivariant / invariant universality
    
*   separation power vs approximation power in equivariant nets

That is the scope I would use for the actual survey. Next I’d turn this into a **paper-by-paper survey map** with seminal papers, main theorem statements, and how the results relate.