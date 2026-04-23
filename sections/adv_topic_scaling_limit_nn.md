# Scaling limit of deep networks

> From: https://chatgpt.com/c/69e99704-1378-83ea-93c8-03303f7185ec

# you asked

survey all the papers about scaling limit of neural networks, consider

1. the neural tangent kernel

2. the mean field limit

3. the long context limit of transformer

before survey, please first extend this list comprehensively

---

# chatgpt response

I’m going to turn your three items into a fuller taxonomy of neural-network scaling limits, then anchor each branch with core papers and later work so the survey doesn’t miss an important regime.

I’ve got the classic width limits and I’m now broadening the map to the other asymptotic directions people actually study: initialization/GP, finite-width corrections, depth limits, tensor-program unifications, and sequence-length limits for attention/transformers.

The three branches you named are important, but they’re not enough for a clean literature map. I’m now splitting the space by which dimension goes to infinity—width, depth, context length, head dimension, number of heads, sample size—and by whether the limit is lazy, mean-field, GP, or genuinely feature-learning.

I’ve now pinned the main missing branches with primary sources. The big structural point is that “scaling limit of neural networks” is not one topic but a product of at least three choices: which size parameter scales, which parameterization is used, and whether the limit freezes features or keeps feature learning alive.

A literal “all papers” survey is too open-ended, so the right first step is to replace your 3-item list with a taxonomy. The clean split is: which quantity goes to infinity, and what kind of limit that produces. With that in mind, your list should be expanded to at least the following.

1. **Initialization-only infinite-width prior / NNGP limit.**  
   Random wide networks converge, before training, to Gaussian processes. This is the “prior” side of infinite-width theory, and it is distinct from training dynamics. Foundational references are Lee et al. (deep fully connected nets as GPs), Matthews et al. (wide deep networks as GPs), Novak et al. (CNNs as GPs), and Yang’s tensor-program extension to many architectures including attention/transformers. ([arXiv](https://arxiv.org/abs/1711.00165))
2. **Neural tangent kernel / lazy / linearized training limit.**  
   This is your item 1, but it should be broadened: Jacot et al. introduced the NTK; Lee et al. showed wide nets evolve as linearized models; Chizat et al. clarified that “lazy training” is a scaling regime rather than a universal explanation of deep learning; Woodworth et al. studied the transition between kernel and rich regimes; Yang extended NTK limits to broad architecture classes. ([arXiv](https://arxiv.org/abs/1806.07572))
3. **Mean-field / particle / PDE limit for shallow networks.**  
   This is your item 2, but in the literature it usually means that the empirical measure of neurons converges to a nonlinear measure-valued or PDE dynamics, especially for two-layer nets. Core papers are Mei–Montanari–Nguyen and Sirignano–Spiliopoulos, including both law-of-large-numbers and central-limit descriptions. ([arXiv](https://arxiv.org/abs/1804.06561))
4. **Multilayer and deep mean-field limits.**  
   Extending mean-field theory beyond two layers is its own branch, not just a minor variant of item 3. Representative papers are Sirignano–Spiliopoulos on deep networks, Nguyen–Pham’s rigorous multilayer framework, Pham–Nguyen on three-layer global convergence, and Lu et al. on deep ResNets. ([arXiv](https://arxiv.org/abs/1903.04440))
5. **Infinite-width but feature-learning (“rich”) limits.**  
   NTK and standard lazy limits freeze features too much for many modern phenomena, so there is a separate line on infinite-width limits that still permit feature learning. Key references are Yang–Hu and Bordelon–Pehlevan’s dynamical mean-field theory work. ([arXiv](https://arxiv.org/abs/2011.14522))
6. **Parameterization, universality, and scaling-rule theory.**  
   A large part of the field is really about *which parameterization* produces GP, NTK, mean-field, or feature-learning limits. This includes Tensor Programs I–II and the μP / muTransfer line, plus newer depthwise extensions. ([arXiv](https://arxiv.org/abs/1910.12478))
7. **Finite-width corrections and fluctuation theory.**  
   Once the leading infinite-width limit is known, another branch studies the next-order corrections: randomness of the empirical NTK, CLTs, diagrammatic expansions, and quantitative finite-width error. Hanin–Nica, Dyer–Gur-Ari, Sirignano–Spiliopoulos, and recent Feynman-diagram NTK papers belong here. ([arXiv](https://arxiv.org/pdf/1909.11304))
8. **Depth limits, signal propagation, and edge-of-chaos theory.**  
   Here the asymptotic variable is depth rather than width. The main objects are forward/backward correlation recursions, Jacobian spectra, dynamical isometry, and trainability at extreme depth. Core papers are Schoenholz et al., Yang–Schoenholz on ResNets, and Xiao et al. on CNN dynamical isometry. Continuous-depth models via Neural ODEs are adjacent to this branch. ([arXiv](https://arxiv.org/abs/1611.01232))
9. **Joint width–depth proportional limits.**  
   Modern models are both wide and deep, so another literature sends width and depth to infinity together at a fixed ratio. This produces genuinely new limits, often SDE-like or non-Gaussian, rather than the usual fixed-depth GP/NTK limits. Representative papers are Li–Nica–Roy and Noci et al.’s shaped transformer work. ([arXiv](https://arxiv.org/abs/2106.04013))
10. **High-dimensional proportional limits.**  
   A different asymptotic sends input dimension, sample size, and width together to infinity at comparable rates. This is closer to random-matrix/statistical-mechanics theory than to classical NTK or mean-field PDEs, and it is especially useful for studying feature learning and generalization. Ba et al. and Cui et al. are representative references. ([arXiv](https://arxiv.org/abs/2205.01445))
11. **Architecture-general GP/NTK limits.**  
   The taxonomy should explicitly note that most of the above regimes split again by architecture: MLP, CNN, RNN, ResNet, and transformer. CNN GP limits were derived explicitly by Novak et al., while Tensor Programs generalized GP and NTK results to recurrent nets, attention, transformers, batchnorm, and layernorm. ([arXiv](https://arxiv.org/abs/1810.05148))
12. **Transformer width/head/depth asymptotics.**  
   For transformers, “infinite width” is too coarse: one can scale key/query dimension, value dimension, number of heads, or depth, and these are not equivalent. Recent papers study multi-head feature-learning limits, proportional depth–width limits, and single-attention-layer infinite-width laws. ([arXiv](https://arxiv.org/abs/2405.15712))
13. **Transformer mean-field dynamics.**  
   Separate from NTK and separate from long-context extrapolation, there is now a mean-field literature treating attention as an interacting-particle or DMFT-type system. Kim–Suzuki, Poc-López–Aguilera, Burger et al., and Rigollet fit here. ([arXiv](https://arxiv.org/abs/2402.01258))
14. **Transformer long-context / sequence-length limits.**  
   Your item 3 should be split into at least three subbranches:  
   **(a)** length extrapolation and positional encodings,  
   **(b)** context-length-induced pathologies such as rank collapse, entropy collapse, dispersion, and localization,  
   **(c)** formal asymptotic theories of long-context behavior.  
   Core references here are Press et al. on ALiBi, Kazemnejad et al. on positional encodings and length generalization, Huang et al. on a formal framework for length generalization, Noci et al. on rank collapse, Bao et al. on localization, Rigollet on mean-field long-context phase transitions, and Liu on RoPE bounds for long-context transformers. ([arXiv](https://arxiv.org/abs/2108.12409))
15. **Optional but real niche branch: non-Gaussian infinite-width priors.**  
   If weights are heavy-tailed or scaled differently, the infinite-width prior need not be Gaussian; stable-process limits form a smaller but genuine branch of the literature. ([arXiv](https://arxiv.org/abs/2003.00394))
So, the shortest serious replacement for your original list is:

- GP/NNGP limit  
- NTK/lazy limit  
- shallow mean-field limit  
- deep/multilayer mean-field limit  
- feature-learning infinite-width limit  
- finite-width corrections  
- depth / edge-of-chaos / continuous-depth limit  
- joint width–depth limit  
- high-dimensional proportional limit  
- transformer width/head/depth limits  
- transformer mean-field limits  
- transformer long-context / sequence-length / positional-encoding limits ([arXiv](https://arxiv.org/abs/1711.00165))
That is the taxonomy I would use before writing the actual survey.