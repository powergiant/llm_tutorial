# Convergence of deep learning

1. exprimental results 

2. theoretical results in convex optimization related to DL (include the basic results)

3. theoretical results in non-convex optimization related to DL (include the basic results)

4. deep learning saddle

5. effect of momentum and adaptive learning rate

6. muon

7. multi-objective (loss = sum of several tasks)

8. edge of stability, smooth decay phase and instable decay phase

9. lr schedule

10. llm: grokking, double maximum of loss curve

please first extend this list comprehensively

I’m expanding your outline into a survey-grade taxonomy, and I’m checking recent primary sources for newer items like Muon, edge-of-stability variants, and LLM-specific phenomena so the added sections aren’t dated.

I’ve got the core branches. I’m now filling the missing pieces that usually matter in a convergence survey but aren’t yet in your list: overparameterization, implicit bias, normalization/scale-invariance, batch-size/noise, architecture effects, and interpolation/generalization regimes.

Yes — I’d broaden it quite a bit.

One naming tweak first: **“double maximum of loss curve” is not a standard label**. For a survey, I would rename that item to **non-monotone or multi-phase loss curves**, and put under it things like **double descent**, **catapult/spike dynamics**, **grokking plateaus**, and **WSD-style elevated stable-phase loss followed by a sharp decay-phase drop**. [arXiv+3OpenReview+3arXiv+3](https://openreview.net/forum?id=B1g5sA4twr)

Here is the expanded outline I would use.

## 1\. Foundations

1.  **Problem setup and notions of convergence**  
    1.1 objective convergence  
    1.2 gradient/stationarity convergence  
    1.3 parameter convergence vs function convergence  
    1.4 representation convergence  
    1.5 training convergence vs generalization convergence
    
2.  **Empirical convergence phenomena**  
    2.1 train/val/test loss and error curves  
    2.2 convergence speed vs final quality  
    2.3 wall-clock / step / token efficiency  
    2.4 dependence on width, depth, data size, and batch size  
    2.5 reproducibility and seed sensitivity
    
3.  **Convex optimization background (basic results)**  
    3.1 GD on smooth convex objectives  
    3.2 SGD on convex objectives  
    3.3 strongly convex and PL regimes  
    3.4 acceleration: heavy-ball, Nesterov  
    3.5 interpolation-regime convex results  
    3.6 line search and adaptive step sizes in convex settings
    
4.  **General non-convex optimization background (basic results)**  
    4.1 first-order stationarity guarantees  
    4.2 second-order stationarity  
    4.3 smooth vs nonsmooth nonconvex problems  
    4.4 weakly-convex / KL / PL-like conditions  
    4.5 local regularity and benign landscapes
    
5.  **Saddles, plateaus, symmetries, and poor local minima**  
    5.1 strict-saddle theory  
    5.2 perturbed GD / SGD escaping saddles  
    5.3 flat directions from parameter symmetries  
    5.4 spurious minima vs benign minima  
    5.5 plateaus and slow manifolds
    
6.  **Overparameterization and global convergence theory**  
    6.1 NTK / lazy-training regime  
    6.2 mean-field / Wasserstein gradient-flow regime  
    6.3 finite-width overparameterized convergence  
    6.4 residual-network convergence theory  
    6.5 deep linear networks and matrix factorization as tractable models

These are the core theory pillars a modern convergence survey needs: classical SGD/GD results, saddle escape, and the deep-learning-specific overparameterized regimes built around NTK, mean-field, and finite-width global convergence. [Proceedings of Machine Learning Research+5康奈尔大学计算机科学系+5arXiv+5](https://www.cs.cornell.edu/courses/cs6241/2019sp/readings/Bottou-2018-opt.pdf)

## 2\. Implicit bias, geometry, and late-training behavior

7.  **Implicit bias of gradient methods**  
    7.1 max-margin bias  
    7.2 minimum-norm / low-rank bias  
    7.3 architecture-dependent bias  
    7.4 optimizer-dependent bias
    
8.  **What happens after zero training loss**  
    8.1 interpolation  
    8.2 post-interpolation SGD dynamics  
    8.3 manifold of minima  
    8.4 label-noise effects  
    8.5 escape from kernel regime
    
9.  **Stochasticity, batch size, and gradient noise**  
    9.1 small-batch vs large-batch behavior  
    9.2 gradient noise scale / temperature view  
    9.3 critical batch size  
    9.4 interpolation-regime SGD  
    9.5 generalization gap in large-batch training
    
10.  **Loss-landscape geometry**  
     10.1 sharp vs flat minima  
     10.2 Hessian spectrum and sharpness proxies  
     10.3 mode connectivity / loss barriers  
     10.4 intrinsic dimension / effective rank  
     10.5 feature drift vs lazy dynamics
     
11.  **Terminal-phase phenomena**  
     11.1 neural collapse  
     11.2 representation collapse vs useful collapse  
     11.3 nearest-class-center geometry
     
12.  **Frequency-wise and representation-wise convergence**  
     12.1 spectral bias  
     12.2 low-frequency-first learning  
     12.3 frequency-dependent convergence rates

A separate geometry-and-implicit-bias block is essential now, because post-zero-loss SGD, batch/noise effects, mode connectivity, Neural Collapse, and spectral bias are no longer side topics; they are central to explaining why optimization keeps changing the solution even after interpolation. [arXiv+5arXiv+5OpenReview+5](https://arxiv.org/abs/1710.10345)

## 3\. Optimizers and phase behavior

13.  **Momentum and acceleration**  
     13.1 heavy-ball momentum  
     13.2 Nesterov momentum  
     13.3 momentum decay / scheduling  
     13.4 momentum and noise shaping  
     13.5 momentum near saddles and sharp regions
     
14.  **Adaptive and preconditioned methods**  
     14.1 AdaGrad  
     14.2 RMSProp  
     14.3 Adam  
     14.4 AMSGrad  
     14.5 AdamW / decoupled weight decay  
     14.6 second-order / natural-gradient / Shampoo / K-FAC family
     
15.  **Orthogonalized and matrix-structured optimizers**  
     15.1 Muon  
     15.2 Newton-Muon  
     15.3 Adam–Muon hybrids / adaptive orthogonalized momentum  
     15.4 comparison with Shampoo/SOAP-style matrix methods
     
16.  **Learning-rate schedules**  
     16.1 constant LR  
     16.2 step decay  
     16.3 cosine decay  
     16.4 polynomial / inverse-sqrt decay  
     16.5 warm restarts  
     16.6 one-cycle / cyclical LR  
     16.7 schedule-free and checkpoint-branching variants
     
17.  **Warmup and schedule dynamics**  
     17.1 why warmup helps  
     17.2 warmup for SGD vs Adam  
     17.3 stable phase vs decay phase  
     17.4 Warmup-Stable-Decay (WSD)
     
18.  **Sharpness-driven phase behavior**  
     18.1 progressive sharpening  
     18.2 edge of stability  
     18.3 catapult phase  
     18.4 stable/smooth decay vs unstable decay  
     18.5 training-instability phase diagrams  
     18.6 super-convergence

On the algorithmic side, I would definitely separate momentum, Adam-family methods, decoupled weight decay, warmup/WSD, and newer matrix-structured optimizers like Muon. Those are distinct literatures now, not one subsection. [arXiv+10OpenReview+10OpenReview+10](https://openreview.net/forum?id=ryQu7f-RZ)

## 4\. Parameterization, objectives, and architecture dependence

19.  **Normalization, scale invariance, and parameterization**  
     19.1 batch normalization  
     19.2 layer normalization  
     19.3 weight normalization  
     19.4 scale-invariant dynamics  
     19.5 μP / width-transfer parameterizations  
     19.6 initialization dependence
     
20.  **Multi-task / multi-objective optimization**  
     20.1 summed-loss training  
     20.2 loss-scale imbalance  
     20.3 gradient interference / conflict  
     20.4 Pareto stationarity  
     20.5 MGDA / PCGrad / CAGrad / Nash-MTL  
     20.6 fairness / priority / bargaining interpretations
     
21.  **Architecture-specific convergence**  
     21.1 deep linear nets  
     21.2 CNNs  
     21.3 ResNets / skip connections  
     21.4 normalization-heavy networks  
     21.5 Transformers  
     21.6 Mixture-of-Experts  
     21.7 architecture-dependent optimizer behavior
     
22.  **Systems-level convergence factors**  
     22.1 distributed SGD  
     22.2 gradient clipping  
     22.3 mixed precision / numerical stability  
     22.4 asynchronous or large-scale effects

I would definitely add separate sections for multitask optimization and architecture dependence: gradient conflict/Pareto trade-offs, ResNet skip connections, and Transformer-specific optimization issues are all technically distinct from generic nonconvex theory. [arXiv+8NeurIPS 论文集+8OpenReview+8](https://proceedings.neurips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html)

## 5\. LLM-specific convergence

23.  **Scaling laws and compute-optimal training**  
     23.1 parameter–data–compute tradeoffs  
     23.2 compute-optimal stopping  
     23.3 token budget and schedule coupling
     
24.  **LLM-specific delayed generalization**  
     24.1 grokking in toy settings  
     24.2 grokking-like behavior in LLM pretraining  
     24.3 memorization-to-generalization transition
     
25.  **Non-monotone and multi-phase loss curves in LLMs**  
     25.1 epoch-wise / model-wise double descent  
     25.2 WSD non-traditional loss curves  
     25.3 hidden phase transitions / breakthroughs  
     25.4 checkpoint branching and decay-phase gains
     
26.  **Data and objective structure**  
     26.1 data mixture effects  
     26.2 curriculum and data order  
     26.3 multi-domain / dual-objective training  
     26.4 instruction-tuning or fine-tuning overtraining

For LLMs, I would not stop at grokking. The recent frontier also includes compute-optimal training, WSD-style schedule effects, token-wise learning-curve analysis, and newer evidence for grokking-like delayed generalization in practical LLM pretraining. [arXiv+6arXiv+6arXiv+6](https://arxiv.org/pdf/2203.15556)

## 6\. Survey methodology and open problems

27.  **Diagnostics and measurements**  
     27.1 loss, error, calibration  
     27.2 gradient norm / update norm  
     27.3 Hessian sharpness / critical sharpness  
     27.4 gradient conflict metrics  
     27.5 representation and pathway metrics
     
28.  **How to compare results fairly**  
     28.1 same steps vs same compute vs same wall-clock  
     28.2 same data order vs reshuffling  
     28.3 same optimizer state / warmup / decay budget  
     28.4 same parameterization / normalization / weight decay
     
29.  **Open problems**  
     29.1 feature learning vs lazy theory  
     29.2 optimizer-dependent implicit bias at scale  
     29.3 Muon theory beyond current surrogates  
     29.4 multitask convergence criteria that predict transfer  
     29.5 Transformer/LLM-specific phase transitions  
     29.6 linking convergence diagnostics to downstream generalization

Relative to your original list, the biggest missing blocks were:

*   overparameterization / NTK / mean-field / finite-width theory,
    
*   implicit bias and post-zero-loss dynamics,
    
*   batch-size and noise effects,
    
*   normalization and scale invariance,
    
*   loss-landscape geometry beyond saddles,
    
*   architecture-specific convergence,
    
*   LLM scaling and WSD-style schedule dynamics,
    
*   survey methodology itself.

I’d also refactor several of your original items:

*   **5** → split into **momentum** and **adaptive/preconditioned methods**
    
*   **6** → widen from **Muon** to **orthogonalized and matrix-structured optimizers**
    
*   **8** → rename to **sharpness and phase behavior**
    
*   **9** → split into **LR schedules** and **warmup/WSD**
    
*   **10** → rename to **LLM-specific delayed generalization and non-monotone loss curves**

Once this structure is fixed, the next pass should attach landmark papers, exact results, assumptions, and open questions under each subsection.