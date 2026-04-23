# Expressiveness of transformers

expressiveness of long context test, niah

no cot as a bounded circuit

has cot parameter expressiveness

has cot prompt expressiveness

examples, associative recall, formal languages

before survey, please first extend this list comprehensively

I’m going to turn your seed list into a sharper survey taxonomy first, then anchor it in the literature so it is broad enough to organize a real survey rather than just a reading list.

I’ve got the backbone of the taxonomy: classic expressivity splits into approximation-theoretic, automata/language-theoretic, and circuit/logic views, but your CoT-focused list needs several more axes to be survey-complete. I’m now filling in the missing branches around prompting, precision, depth/looping, and task families.

Before the survey, I would broaden your list into a **taxonomy of questions** rather than a flat reading list.

Your current list is pointing at the right core distinction, but it mixes together three different things: **model assumptions**, **where extra computational power comes from**, and **which task family is used to separate models**. In the transformer-expressivity literature, those three axes are usually what make results look contradictory when they are actually compatible. The formal-languages survey literature makes this explicit, and newer papers sharpen it with exact logic/circuit characterizations for restricted transformer variants. [arXiv+6ar5iv+6ar5iv+6](https://ar5iv.labs.arxiv.org/html/2311.00208)

I would therefore replace your seed list with the following **comprehensive survey structure**.

1.  **Model assumptions and resource regime**  
    This should be the first section, because almost every theorem depends on it: encoder vs decoder vs recognizer, causal masking vs bidirectional attention, softmax vs hard/average/leftmost attention, positional encoding type, finite precision vs arbitrary precision, constant depth vs growing depth, fixed parameters vs parameters scaling with input length, and whether the object of study is exact recognition, next-token modeling, or continuous approximation. Existing surveys emphasize that these assumptions are not cosmetic; they often decide the complexity class completely. [ar5iv+4ar5iv+4ar5iv+4](https://ar5iv.labs.arxiv.org/html/2311.00208)
    
2.  **No-CoT transformers as bounded parallel computation**  
    This is your “no cot as a bounded circuit” bucket, but I would phrase it more broadly as: **what can a transformer do in one bounded-depth forward pass?** This includes circuit and logic characterizations such as AC0-, TC0-, FO-, FO+counting-, LTL-, or RASP-like views, and exact upper/lower bounds for finite-precision transformers without inference-time recurrence. This is one of the cleanest formal branches of the literature. [ar5iv+4arXiv+4arXiv+4](https://arxiv.org/abs/2301.10743)
    
3.  **Exact characterizations of restricted practical transformers**  
    Separate from generic “bounded circuit” results, there is now a line of work giving **exact expressiveness** for specific realistic transformer variants: masked hard-attention models, fixed-precision causal language models, or attention schemes like leftmost/unique-hard attention. This deserves its own section because these papers do not just give upper bounds; they often identify the exact language or logic class recognized by a specific architecture. [arXiv+3arXiv+3arXiv+3](https://arxiv.org/html/2310.13897v3)
    
4.  **Parameter-space expressiveness of transformers**  
    This is different from prompting. The question here is: **with trainable weights, what families of functions/computations can transformers represent at all?** This includes universal approximation, Turing completeness, decoder-only power, and recent constant-bit results. I would treat this as the “weight-space” or “architecture-level” expressiveness branch. [arXiv+3arXiv+3arXiv+3](https://arxiv.org/abs/1912.10077)
    
5.  **CoT/scratchpad expressiveness with trainable parameters**  
    This should replace your “has cot parameter expressiveness” item. The key question is: **how much power is gained when the model can emit intermediate tokens and continue computing over them?** Recent work studies logarithmic, linear, and polynomial CoT lengths; some results show modest gains at short CoT lengths, while others show strong separations once enough intermediate steps are allowed. This is now a major theoretical branch, not just an empirical prompt-engineering phenomenon. [arXiv+3arXiv+3OpenReview+3](https://arxiv.org/html/2310.07923v5)
    
6.  **Prompt-space expressiveness with a frozen backbone**  
    This should replace your “has cot prompt expressiveness” item. It is a distinct question: **with fixed transformer weights, how much computation can be induced by changing only the prompt or soft prompt?** This includes prompt tuning, prefix/soft-prompt universality, and even “prompting as programming” results. It should not be merged with trainable-parameter CoT. [arXiv+2OpenReview+2](https://arxiv.org/pdf/2305.18787)
    
7.  **In-context learning as an expressiveness question**  
    This is adjacent to prompting but not identical. Here the object is the function from context examples to predictions: **what mappings can transformers implement in-context, without gradient updates?** This includes universal in-context approximation results and task-specific studies on simple function classes and discrete functions. In a modern survey, this deserves its own section rather than being buried under prompting. [arXiv+2arXiv+2](https://arxiv.org/abs/2208.01066)
    
8.  **Memory and retrieval expressiveness**  
    Your “associative recall” example should be expanded into a whole branch: induction heads, key-value retrieval, associative memory, factual recall, and test-time retrieval capacity. This literature connects formal expressiveness, mechanistic circuits, and benchmark design. Associative recall is not just one toy task; it is a canonical probe of the model’s internal memory primitives. [arXiv+3变压器电路+3arXiv+3](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
    
9.  **Formal-language expressiveness**  
    Your “formal languages” item should be widened into a hierarchy: star-free languages, regular languages, counter-like languages, context-free subclasses, and general CFL/context-sensitive boundaries. This is one of the deepest mature branches of the literature, and recent papers now pin down several exact frontiers for masked, fixed-precision, looped, or CoT-augmented transformers. [arXiv+4ar5iv+4arXiv+4](https://ar5iv.labs.arxiv.org/html/2311.00208)
    
10.  **Canonical separator task families**  
     “Examples” should become a principled section of benchmark families used to separate expressiveness claims. I would at least include:

*   counting/majority/parity-like tasks,
    
*   associative recall and factual retrieval,
    
*   regular-language and automata tasks,
    
*   serial algorithmic tasks such as permutation composition, iterated squaring, graph connectivity, and circuit value,
    
*   arithmetic/compositional tasks,
    
*   latent-state or automaton-tracking tasks.  
    Different papers use different tasks because they witness different limits: parallelism limits, memory limits, precision limits, or sequential-computation limits. [arXiv+5OpenReview+5arXiv+5](https://openreview.net/forum?id=3EWTEy9MTM)

11.  **Counting as a standalone theme**  
     Counting should probably be broken out from “examples” into its own heading. There is now a substantial line of work on what transformers can count, when counting fails out of distribution, and how counting power relates to semilinear or semialgebraic structure. This is important because many lower bounds and separator tasks reduce to counting phenomena. [arXiv+1](https://arxiv.org/abs/2407.15160)
     
12.  **Looping, recurrence, padding, and latent-thought alternatives to explicit CoT**  
     A modern survey should not treat CoT as the only way to add inference-time computation. There is now a parallel line on looped transformers, padded transformers, and other recurrence-like mechanisms. Some recent work compares these directly with CoT and shows different strengths: loops help with parallel deterministic computation, while token-level CoT can help on other kinds of compositional inference. [arXiv+4arXiv+4arXiv+4](https://arxiv.org/abs/2301.13196)
     
13.  **Learnability, sample complexity, and length generalization**  
     Expressive power alone is not enough. Several papers show tasks that are representable by transformers but hard to learn without CoT, scratchpads, or the right inductive bias. This includes globality barriers, parity-style hardness, and length-generalization theory. I would make this a separate section, because it often explains why empirical behavior looks weaker than formal expressiveness suggests. [arXiv+4OpenReview+4arXiv+4](https://openreview.net/forum?id=N6pbLYLeej)
     
14.  **Mechanistic realizations of expressiveness results**  
     This section asks not only whether the architecture can express a computation, but **what internal circuit implements it**: induction heads, automata-like state tracking, associative-memory views, and finite-state behavior emerging inside MLP neurons. This provides a bridge between formal results and interpretability. [变压器电路+2arXiv+2](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
     
15.  **Comparative expressiveness against neighboring architectures**  
     A good survey should compare transformers with RNNs, looped/recurrent transformers, attention-free alternatives, and memory-augmented models. Some recent papers use these comparisons to identify which limitations are truly “transformer” limitations and which are just bounded-parallelism limitations. [arXiv+2arXiv+2](https://arxiv.org/abs/2505.19245)
     
16.  **Open problems and unresolved frontiers**  
     The survey should end with unresolved boundaries: exact power of practical finite-precision decoder-only models with realistic positional encodings, the gap between representability and learnability, the sharp boundary between CoT and looped computation, precise prompt-capacity tradeoffs, and stronger exact characterizations for CFL- and P-level behavior under realistic resource limits. Recent papers explicitly present the field as still split among circuit, approximation, and related viewpoints rather than fully unified. [OpenReview+3ACL Anthology+3arXiv+3](https://aclanthology.org/2026.eacl-long.87.pdf)

So the **extended list** I would actually use is:

*   assumptions / model regime
    
*   no-CoT bounded-pass expressiveness
    
*   exact restricted-model characterizations
    
*   parameter-space universality and Turing completeness
    
*   CoT/scratchpad expressiveness with trainable weights
    
*   prompt/soft-prompt expressiveness with frozen weights
    
*   in-context-learning expressiveness
    
*   memory / associative recall / retrieval
    
*   formal-language hierarchy
    
*   canonical separator tasks
    
*   counting
    
*   looping / padding / latent-thought alternatives
    
*   learnability / sample complexity / length generalization
    
*   mechanistic realizations
    
*   comparative architecture results
    
*   open problems.

That is the version I would use before doing the paper-by-paper survey.

---

## ok, for all the points mentioned above, make a comprehensive survey

I’m building this as a literature map rather than a loose bibliography: first I’m collecting the major theoretical branches, then I’ll synthesize them into one coherent survey with clear scope and gaps.

I’ve now got enough coverage to write a real survey rather than a paper dump. I’m filling two remaining gaps before I synthesize: the in-context-learning theory line and the sharpest prompt-tuning limitation results.

Below is a best-effort **theory-centered survey of the expressiveness of transformers through April 22, 2026**. I am focusing on papers that make formal claims about representational or computational power, plus a smaller set of adjacent papers on learnability, length generalization, and mechanistic realization. I am not trying to enumerate every empirical “reasoning” paper; instead I am organizing the main formal literature into one coherent picture. The right way to read this literature is not “are transformers expressive or not?”, but “**under which assumptions, and where does the extra computation come from?**” The TACL survey by Strobl et al. already makes this point explicitly: many apparently contradictory results are compatible once one separates assumptions about masking, positional information, numeric precision, depth, and whether inference-time intermediate computation is allowed. [arXiv+1](https://arxiv.org/abs/2311.00208?utm_source=chatgpt.com)

## 1\. What “expressiveness” means in this literature

There are at least four different notions of expressiveness in the transformer literature. One is **approximation-theoretic**: can a transformer approximate a class of continuous sequence-to-sequence functions? A second is **formal-language / circuit-theoretic**: what discrete languages or computations can a transformer recognize on unbounded-length inputs? A third is **frozen-model adaptation expressiveness**: what can be achieved by changing only the prompt or soft prompt while keeping weights fixed? A fourth is **inference-time expressiveness**: what happens if the model is allowed to generate intermediate tokens, loop, or use padding as extra test-time compute? A useful survey has to keep these apart, because a universal approximation theorem and an AC0^00/TC0^00 upper bound are talking about different settings, not contradicting each other. [arXiv+2arXiv+2](https://arxiv.org/pdf/2311.00208)

A second key distinction is between **representability** and **learnability**. A transformer family may be able to represent a function class in principle, yet still fail to learn it from realistic data, optimization, or context lengths. Recent work on globality barriers, sample complexity, and length generalization shows that this gap is not a side issue; it is central to understanding why empirical transformers often look weaker than their strongest expressiveness theorems suggest. [arXiv+2arXiv+2](https://arxiv.org/abs/2406.06467)

## 2\. No-CoT transformers as bounded parallel computation

The oldest clean lesson is that **fixed-depth transformers without test-time intermediate computation are fundamentally parallel machines**. Hahn’s 2020 result is the canonical starting point: across soft and hard attention settings, self-attention cannot model periodic finite-state languages or hierarchical structure unless the number of heads or layers grows with input length. In particular, languages built around parity-like periodicity and Dyck-style hierarchy witness strong limitations of pure bounded-pass self-attention. [arXiv+1](https://arxiv.org/abs/1906.06755)

This intuition was sharpened by circuit-complexity work. Hao, Angluin, and Frank proved that unique-hard-attention models such as UHAT and GUHAT are upper-bounded by **AC0^00**, while averaging-hard-attention (AHAT) is strictly stronger: it can recognize non-AC0^00 languages such as **MAJORITY** and **DYCK-1**, though it still sits under a broader bounded-circuit upper bound. In parallel, Merrill, Sabharwal, and Smith showed that saturated transformers are **constant-depth threshold circuits**, giving a **TC0^00** upper bound for that regime. Together these papers make the “no-CoT transformer as bounded circuit” viewpoint precise rather than metaphorical. [ACL Anthology+2ACL Anthology+2](https://aclanthology.org/2022.tacl-1.46.pdf)

More recent work made these upper bounds much tighter and more logic-like. Chiang, Cholak, and Pillay identify a first-order logic with counting quantifiers that is simultaneously an upper and lower bound for fixed-precision transformer encoders, bringing the field close to an exact characterization. Merrill and Sabharwal then show that **log-precision** transformers correspond to first-order logic augmented with majority-vote quantifiers. So the current state of the literature is that, under finite or logarithmic precision and no extra inference-time steps, transformer expressiveness is often best understood through **FO / FO+counting / FO+majority** and the corresponding **AC0^00/TC0^00** circuit classes. [arXiv+2arXiv+2](https://arxiv.org/abs/2301.10743)

## 3\. Exact characterizations of restricted practical variants

A major step forward after the early upper bounds was to stop asking only for broad inclusions and instead prove **exact expressiveness results for specific transformer variants**. Yang, Chiang, and Angluin show that masked hard-attention transformers with **strict masking** and **no positional encodings** are exactly equivalent to **linear temporal logic (LTL)** and therefore recognize exactly the **star-free languages**. This is one of the cleanest exact frontiers in the area. [arXiv](https://arxiv.org/abs/2310.13897)

That exact-characterization program was extended to more LM-like settings. Li and Cotterell analyze a restricted but practical causal LM regime: **fixed precision, strict future masking, soft attention, and no positional encodings**. They show that this model class is exactly as expressive as a fragment of LTL with only the **past** operator. This matters because it says that even soft-attention language models can have surprisingly crisp logical frontiers once the precision and positional assumptions are fixed. [arXiv+1](https://arxiv.org/abs/2505.23623)

A further refinement is that seemingly tiny design choices can change the exact class. Jerad et al. show that with unique hard attention, the tie-breaking rule matters: the familiar equivalence to LTL breaks if one keeps only **leftmost-hard attention**; that version becomes strictly weaker, and in their analysis it is equivalent to soft attention in the same setting. So the field has moved beyond “transformers are around AC0^00/TC0^00” to a more delicate landscape where masking direction, tie-breaking, and positional information determine exact logical classes. [arXiv+1](https://arxiv.org/abs/2503.14615)

## 4\. Universality and Turing completeness

Against those bounded-pass upper bounds stands a second, equally important literature showing strong **positive universality results**. Yun et al. proved that transformers are universal approximators of continuous permutation-equivariant sequence-to-sequence functions with compact support, and with positional encodings they can universally approximate arbitrary continuous sequence-to-sequence functions on compact domains. This is the canonical approximation-theoretic positive result. [arXiv](https://arxiv.org/abs/1912.10077)

At the discrete-computation end, Pérez, Barceló, and Marinkovic proved that the transformer is **Turing complete** under a setting with **arbitrary-precision rational activations**, using a direct simulation of Turing machines. Their paper is explicit that this does **not** contradict the fixed-precision limitation results: the assumptions differ, especially on numerical precision and attention idealization. [机器学习研究杂志+1](https://jmlr.org/papers/volume22/20-302/20-302.pdf)

The universality line has also become stronger recently. Li and Wang show that **constant bit-size transformers are Turing complete**, provided the context window is sufficiently long, and they characterize the expressive power of a constant-bit transformer with window length s(n)s(n)s(n) as exactly **SPACE\[s(n)\]\[s(n)\]\[s(n)\]**. This is a striking strengthening because it removes the need for precision or parameter growth with input length in the theorem’s regime. The price is that one is no longer in the simple bounded-pass, small-window setting that drove the AC0^00/TC0^00 results. [arXiv+2arXiv+2](https://arxiv.org/abs/2506.12027)

So the literature is not saying “transformers are weak” or “transformers are universal” in any absolute sense. It is saying that **fixed-depth, fixed-precision, no-extra-compute transformers are often bounded-parallel devices, while broader regimes with stronger precision, longer windows, or richer inference-time computation can become universal or even Turing complete**. [arXiv+2机器学习研究杂志+2](https://arxiv.org/abs/2301.10743)

## 5\. Chain of thought as an expressiveness amplifier

Your first CoT question has a clear answer in the modern theory: **yes, CoT is a genuine expressiveness amplifier**, not just an empirical prompting trick. Merrill and Sabharwal give the cleanest staircase theorem. For decoder-only transformers, a **logarithmic** number of decoding steps only slightly extends the standard model; a **linear** number of steps, under projected pre-norm, already suffices to recognize **all regular languages**; linear-step CoT stays within **context-sensitive languages**; and **polynomially many steps** under generalized pre-norm characterize exactly **PTIME**. This is one of the strongest and sharpest results in the whole area. [arXiv+1](https://arxiv.org/abs/2310.07923)

A complementary line by Li, Liu, Zhou, and Ma makes the serial-computation intuition extremely concrete. They show that without CoT, constant-depth constant-bit transformers sit in **AC0^00**, but with TTT CoT steps, constant-depth constant-bit transformers with O(log⁡n)O(\\log n)O(logn) embedding size can solve any problem solvable by Boolean circuits of size TTT. Their separator tasks are exactly the kind you named: **permutation composition, iterated squaring, and circuit value**. In this line of work, CoT is formalized as giving a parallel architecture access to a sequential work tape made of emitted tokens. [arXiv+1](https://arxiv.org/abs/2402.12875)

The next question is not just whether CoT helps, but **how much CoT is necessary**. Amiri et al. start a lower-bound theory for scratchpad length in hard-attention transformers, showing that the number of CoT steps needed on different algorithmic problems can be bounded from below, often tightly up to logarithmic factors. Their framing is important because it explains why CoT can remain necessary even for problems that live inside relatively low circuit classes such as parity or multiplication. [arXiv](https://arxiv.org/abs/2502.02393)

There is also now a learnability story around CoT. Wen et al. argue that CoT can yield **polynomial** rather than **exponential** sample complexity in a parity-learning setup by inducing sparse sequential dependencies and sparse attention. Yang, Li, and Wipf show in a PAC-style setting that CoT-style decomposition can make otherwise unlearnable concept classes learnable. So CoT does not only increase representational capacity; it can also change the statistical difficulty of learning the computation. [arXiv+1](https://arxiv.org/abs/2410.05459)

## 6\. Prompt-space expressiveness with a frozen backbone

Your second CoT-adjacent question is whether there is a meaningful notion of **prompt expressiveness** when the weights are frozen. The answer is again yes, but it is crucially weaker and structurally different from weight-space expressiveness. Wang et al. show that soft-prompt tuning can have a **universal approximation** flavor for Lipschitz sequence-to-sequence functions in an appropriate construction, while also proving limitations for finite-depth fixed-weight pretrained transformers. More recent work by Hu et al. strengthens the positive side by showing universality even for **single-layer, single-head** prompt tuning, while also proving exponential lower bounds on the prompt length needed for full memorization. [arXiv+1](https://arxiv.org/abs/2305.18787)

But the strongest conceptual result here is negative. Petrov, Torr, and Bibi analyze prompting and prefix-tuning as ways of steering a frozen network’s internal computation. Their conclusion is that while context-based methods can exploit the extra capacity of embedding space, they suffer **structural limitations**: they are effective at eliciting capabilities already latent in the pretrained model, but they cannot in general induce arbitrary new attention patterns over content tokens. The OpenReview summary puts the takeaway plainly: these methods may fail on genuinely novel tasks that require new attention structures. [arXiv+2arXiv+2](https://arxiv.org/abs/2310.19698)

Recent memory-limit results sharpen the negative side further. Meyer et al. prove that the amount of information memorized by prompt tuning cannot scale faster than **linearly in prompt length**, and they connect this to formal performance degradation with longer contexts. So “CoT prompt expressiveness” exists, but it should not be conflated with “the model can simulate arbitrary new algorithms just by a better prompt.” The theory increasingly treats prompt-space adaptation as a constrained interface to a fixed computation, not a full substitute for changing the model or adding genuine inference-time compute. [arXiv+1](https://arxiv.org/abs/2509.00421)

## 7\. In-context learning as a separate expressiveness question

Prompting and in-context learning are related but not identical. In-context learning asks what mapping from **context examples to predictions** a transformer can realize without weight updates. Garg et al. framed this as learning simple function classes from in-context examples, while von Oswald et al. showed that trained transformers can implement something very close to **gradient descent in the forward pass** on regression tasks. This inaugurated the “transformers as implicit optimizers” perspective. [OpenReview+1](https://openreview.net/pdf?id=flNZJ2eOet)

That optimizer view has since been broadened. Vladymyrov et al. show that each layer of a linear transformer can be interpreted as maintaining the weight vector of an implicit linear regression problem and performing a preconditioned-gradient-descent-like update. On the other hand, Furuya, de Hoop, and Peyré, and then Li et al., prove more direct **universal in-context learning** theorems: deep transformers can approximate continuous in-context mappings uniformly over compact token domains and, in the later line, support few-shot prediction for broad function classes without any further weight updates. So the ICL literature now has two complementary narratives: transformers as **algorithm approximators** and transformers as **universal in-context function approximators**. [arXiv+2arXiv+2](https://arxiv.org/abs/2402.14180)

## 8\. Memory, associative recall, and factual recall

Associative recall is not just an example benchmark; it has become one of the clearest probes of transformer memory primitives. On structured sequential data, Rajaraman et al. show that even a **single-head three-layer** transformer can represent the in-context conditional empirical distribution for kkk\-th order Markov sources, and that attention-only transformers with O(log⁡k)O(\\log k)O(logk) layers can do so by composing induction-like behavior over the last kkk symbols. This is strong evidence that shallow transformers can realize useful retrieval-style computation without needing full-blown recurrence. [arXiv+2OpenReview+2](https://arxiv.org/abs/2407.17686)

A parallel factual-recall line studies storage capacity directly. Nichani, Lee, and Bietti show that shallow transformers can implement factual recall by combining **associative memories** in attention value matrices and MLPs, with storage capacity scaling **linearly with parameter count**. This gives a formal bridge from “associative recall” toy tasks to the more practically relevant question of how transformers store and retrieve facts. [arXiv](https://arxiv.org/abs/2412.06538)

## 9\. Formal languages remain the cleanest mature branch

Among all subareas, the **formal-language** line remains the most mature and best organized. Strobl et al.’s survey is still the best single map of this territory, and its main message holds up: the field is a patchwork of upper bounds, lower bounds, and exact characterizations tied to specific transformer variants and assumptions. [arXiv+1](https://arxiv.org/abs/2311.00208?utm_source=chatgpt.com)

Within that line, the results now span both recognition and generation. On the generative side, Svete and Cotterell show that transformers with hard or sparse attention can **exactly represent any nnn\-gram language model**, giving a concrete lower bound on probabilistic representational capacity rather than only language acceptance. On the recognition side, the hierarchy now runs from star-free and LTL-equivalent masked hard-attention models, through fixed-precision past-LTL causal LMs, up to models with growing depth or test-time compute that reach regular languages and beyond. [arXiv+3arXiv+3arXiv+3](https://arxiv.org/abs/2404.14994)

A striking 2025 result is that **logarithmic depth** already changes the picture qualitatively. Merrill and Sabharwal show that highly uniform transformers with depth Θ(log⁡n)\\Theta(\\log n)Θ(logn) can recognize **regular languages** and solve **graph connectivity**, both of which fixed-depth transformers cannot express under standard complexity assumptions. This makes depth itself a first-class source of expressiveness, not just a quantitative scaling knob. [arXiv+1](https://arxiv.org/abs/2503.03961)

By early 2026, the frontier reached context-free recognition. Jerad et al. show that looped transformers with O(log⁡n)O(\\log n)O(logn) looping layers and O(n6)O(n^6)O(n6) padding can recognize **all context-free languages**, and that important subclasses such as **unambiguous CFLs** need only O(n3)O(n^3)O(n3) padding. So the field has now advanced from “transformers cannot recognize Dyck” to a much more nuanced statement: **standard fixed-depth transformers cannot, but looped and padded transformers can, with quantifiable resource costs**. [arXiv](https://arxiv.org/abs/2601.01754)

## 10\. Counting, parity, and separator tasks

Counting has become a standalone theme because so many separations reduce to it. Yehudai et al. study basic counting tasks and show strong dependence on embedding dimension, context length, and vocabulary; their theory and experiments identify clear regimes where transformers can and cannot “count to nnn.” Sälzer et al. then argue that the field had mostly established only **linear or semilinear** counting so far, and they propose a broader framework for the counting power of transformers. [arXiv+2arXiv+2](https://arxiv.org/abs/2407.15160)

One especially interesting twist is that removing positional encodings does not always destroy counting power. Köcher et al. show that **average-hard-attention transformers without positional encodings** can express very rich counting languages tied to semi-algebraic / Diophantine conditions, yet still cannot express even the simple counting property of **PARITY**. That is a good example of why the field no longer views “counting” as one-dimensional. [arXiv](https://arxiv.org/abs/2505.11199)

Parity itself has become a central stress test. Kozachinskiy, Steifer, and Wałȩga revisit whether practical-looking one-layer transformers can solve parity and show that earlier parity constructions depended on fairly special ingredients; their work gives new lower bounds and clarifies the fragility of parity computation under realistic assumptions. London and Kanade add that **pause tokens** strictly increase the expressivity of constant-depth transformers: with bounded precision, they lift models from a strict subset of AC0^00 to all of AC0^00, and with logarithmic precision they reach TC0^00. So pause tokens, like CoT and padding, are now understood as a distinct kind of inference-time resource. [Oxford University Research Archive+3arXiv+3arXiv+3](https://arxiv.org/abs/2602.05896)

The canonical separator tasks across the literature are now fairly stable: **parity and majority** for counting and nonlinearity; **Dyck and CFLs** for hierarchy; **associative recall / kkk\-gram / Markov prediction** for memory; **graph connectivity** for multistep reasoning; and **permutation composition, iterated squaring, and circuit value** for inherently serial computation. These are the examples to use when comparing no-CoT, CoT, looping, padding, and prompting results. [arXiv+4ACL Anthology+4arXiv+4](https://aclanthology.org/2022.tacl-1.46.pdf)

## 11\. Alternatives to explicit CoT: depth, looping, padding, latent thoughts

A modern survey should not treat explicit token-level CoT as the only way to add inference-time compute. The 2025 literature makes this very clear. Merrill and Sabharwal show that **padding** is a parallelizable form of test-time compute: polynomial padding gives exactly FO-uniform **TC0^00**, and when combined with O(log⁡dn)O(\\log^d n)O(logdn) looping it gives exactly FO-uniform **TCd^dd**; with polylogarithmic looping plus polynomial padding, the model reaches FO-uniform **NC**. That is an exact complexity-theoretic account of padding as extra workspace. [arXiv](https://arxiv.org/abs/2505.18948)

Looped transformers provide a second route. Saunshi et al. argue that many reasoning problems fundamentally require more **depth**, not necessarily more parameters, and show both theoretically and empirically that a kkk\-layer transformer looped LLL times can nearly match a kLkLkL\-layer non-looped model on several reasoning problems. This supports a “latent thought” view: rather than externalizing intermediate steps as tokens, the model can recycle a compact recurrent computation internally. [arXiv](https://arxiv.org/abs/2502.17416)

Taken together, these papers imply that the real comparison is not just “CoT or no CoT,” but **which test-time compute budget is being granted**: extra decoded tokens, extra loops, extra padding, or extra pause tokens. The theory increasingly suggests that these mechanisms occupy different niches rather than one simply dominating the others. [arXiv+2arXiv+2](https://arxiv.org/abs/2505.18948)

## 12\. Learnability, length generalization, and comparison with recurrent models

One of the healthiest recent changes in the literature is the move away from pure representability. Abbe et al. introduce the **globality barrier**, arguing that high-globality targets can be hard to learn efficiently even when transformers are expressive enough in principle. Their scratchpad analysis is especially useful: agnostic scratchpads do not automatically fix the problem, but more structured “inductive scratchpads” can break the barrier and improve out-of-distribution length generalization. [arXiv+2arXiv+2](https://arxiv.org/abs/2406.06467)

Length generalization is now starting to have real theory, not only anecdotes. Huang et al. give a formal framework for understanding when causal transformers with learnable absolute positional encodings can generalize to longer inputs, characterizing functions identifiable from sufficiently long inputs under an idealized inference scheme. This is relevant to expressiveness because it clarifies when a computation representable on all lengths is actually recoverable from training on shorter ones. [arXiv+1](https://arxiv.org/abs/2410.02140)

The comparison with recurrent architectures has also become sharper. Bhattamishra et al. show task-dependent separations: one-layer transformers of logarithmic width can perform **index lookup** where RNNs need linear hidden state, but constant-size RNNs can recognize **bounded Dyck languages** where one-layer transformers need linear size. This is a useful corrective to the habit of asking which architecture is “more expressive” globally; the better question is which architecture is more efficient for which computation. [arXiv+1](https://arxiv.org/abs/2406.09347)

## 13\. Direct answers to your original seed questions

On **“no CoT as a bounded circuit”**: for a large and important slice of the literature, that is exactly the right mental model. Fixed-depth, finite-precision, no-intermediate-generation transformers are often characterized by **AC0^00**\-, **TC0^00**\-, or first-order-logic-style classes. That viewpoint is now supported by multiple independent lines: Hahn’s limitations, the hard-attention circuit results, saturated-transformer threshold-circuit results, and logic characterizations of fixed- and log-precision transformers. [arXiv+4arXiv+4ACL Anthology+4](https://arxiv.org/abs/1906.06755)

On **“has CoT parameter expressiveness”**: yes, emphatically. When the model is allowed to generate and condition on intermediate tokens, the formal expressiveness can rise from weak bounded-parallel classes to **all regular languages**, then to **PTIME**, and in broader constructions even to **Turing completeness**. The exact power depends on CoT length, norm/architecture assumptions, and whether one counts window size as part of the resource budget. [arXiv+2arXiv+2](https://arxiv.org/abs/2310.07923)

On **“has CoT prompt expressiveness”**: yes, but this is a different and weaker question. With frozen weights, soft prompts and prefixes can be universal in certain approximation-theoretic constructions, yet they face hard structural and memory limits. In practice and in theory, prompt-space methods are better viewed as **steering or eliciting** capabilities that are already latent in the backbone, not as freely rewriting the model’s internal algorithm. [arXiv+3arXiv+3arXiv+3](https://arxiv.org/abs/2305.18787)

On your requested **examples**: the most informative ones are now well established. **Associative recall** and **kkk\-gram / Markov prediction** probe memory and induction-style retrieval; **formal languages** such as star-free languages, regular languages, Dyck, and CFLs probe hierarchy and state-tracking; **parity, majority, and counting tasks** probe bounded-parallel nonlinear computation; and **graph connectivity, circuit value, permutation composition, and iterated squaring** probe truly serial reasoning where CoT, looping, or depth scaling matter most. [arXiv+5arXiv+5arXiv+5](https://arxiv.org/abs/2407.17686)

## 14\. Overall synthesis

The main synthesis of the field is this: **a transformer’s expressiveness is determined less by the word “transformer” than by the source of sequential computation it is allowed to use**. With only a bounded forward pass, finite precision, and no intermediate generation, the model behaves like a bounded parallel circuit. Allow the model more numerical precision, more depth, more context, or more test-time workspace through CoT, looping, padding, or pause tokens, and the reachable complexity class can jump dramatically. [arXiv+2arXiv+2](https://arxiv.org/pdf/2311.00208)

That is why the literature should not be summarized as “transformers are weak” or “transformers are universal.” Both statements are true in different regimes. The real contribution of the post-2023 literature is that it has started to **map those regimes systematically** and to compare several different ways of injecting inference-time compute instead of treating CoT as the only option. [arXiv+3arXiv+3arXiv+3](https://arxiv.org/abs/2311.00208)

## 15\. Open problems

The biggest open problem is still to obtain an exact complexity-theoretic characterization of something close to a **real decoder-only LLM**: softmax attention, realistic positional encodings such as RoPE, normalization, finite precision, and practical masking, all at once. A second open problem is the **representability–learnability gap**: several papers now show that being able to express a function says little about whether gradient-based training will find it or generalize it. A third is to compare **explicit CoT, looping, padding, pause tokens, and latent-thought models** in one unified theory rather than pairwise. A fourth is to understand how far **frozen-backbone prompting** can really go on pretrained models, beyond universal-approximation constructions on stylized architectures. [OpenReview+4arXiv+4arXiv+4](https://arxiv.org/pdf/2311.00208)

The most useful next step is an **annotated bibliography** organized by these sections, with one paragraph per paper stating assumptions, theorem type, separator tasks, and the exact source of extra computation.