# Expressiveness of Transformers

## Outline

As in the MLP and CNN section, the basic expressiveness question starts from a ground-truth family and a parametric model family $\{f_\theta:\theta\in\Theta\}$. For transformers, the ground truth is usually not a function of a fixed number of arguments, but a family of sequence-to-sequence computations. Describing such a ground-truth class therefore naturally leads to the theory of computation. Depending on the assumptions on sequence length, positional information, masking, numerical precision, and whether inference is a single forward pass or an iterative generation process with intermediate tokens such as chain of thought, the relevant function classes can range from low-depth circuit classes such as $\mathsf{AC}^0$ and $\mathsf{TC}^0$ to polynomial-time computation and, in more idealized settings, Turing completeness.


1. Model assumptions and resource regime

   We first specify the computational regime. The most important distinction is between sequence length and model size, together with the form of inference: one-pass prediction or iterative generation, with or without chain of thought, prompting, or more agentic interaction. A second distinction is architectural: encoder versus decoder, encoder-decoder versus decoder-only, and transformer versus recurrent or other sequential models. A third distinction concerns structural assumptions such as positional information, masking, and numerical precision. We will also explain the relevant complexity classes and roughly which assumptions lead to classes ranging from $\mathsf{AC}^0$ and $\mathsf{TC}^0$ to polynomial-time computation and, in more idealized settings, Turing completeness.

2. No-CoT transformers as bounded parallel computation

   A transformer that reads the input and immediately answers with one bounded-depth forward pass is best understood as a parallel computation. This is the "no chain-of-thought as a bounded circuit" viewpoint. Under finite-precision or hard-attention assumptions, many transformer variants can be simulated by low-depth circuit or logic classes such as $\mathsf{AC}^0$, $\mathsf{TC}^0$, or first-order logic with counting or majority quantifiers. Such results explain why parity, exact counting, graph connectivity, and some formal languages become separator tasks.

3. Chain of thought as a general computation

   We distinguish parameter expressiveness and prompt expressiveness: the first corresponds more to learning, while the second corresponds more to execution. CoT changes the resource model because the model can write intermediate tokens and then attend to them, so the transformer is no longer only a bounded parallel computation. With trainable parameters, CoT can simulate much more general sequential machines; in the formal literature this is what leads from circuit-style expressiveness toward polynomial-time computation, and in more idealized settings toward Turing completeness. With a frozen pretrained model, CoT prompting is better viewed as steering or eliciting computations that the backbone can already implement, unless the prompt is allowed to be a powerful learned prefix in a theoretical construction.

4. Case studies

   We study case studies separately in the non-CoT and CoT settings.

   * non-CoT

        Formal languages, logic, and counting give the cleanest discrete theory. Results range from limitations on periodic languages and Dyck languages, to exact characterizations of restricted masked hard-attention transformers as star-free languages, to broader logic characterizations for finite- and log-precision models. Counting deserves special attention because many lower bounds and benchmark failures reduce to the difficulty of maintaining exact global counts.

        Associative recall and mechanistic circuits ask whether the model can bind keys to values in context and retrieve the value associated with a query key. This connects formal expressiveness, long-context retrieval, in-context learning, and mechanistic interpretability. Induction heads are one concrete circuit for copy-and-continue behavior; associative-memory analyses study how attention and MLPs can store and retrieve facts.

        Long context raises an expressiveness question because the model must preserve useful positional information as the sequence becomes long, so failures of positional encoding or attention scaling can already break simple retrieval; Needle-in-a-haystack (NIAH) is the standard basic probe for this. A separate but related issue is length generalization: a model trained on shorter sequences may be expressive enough in principle yet still fail on longer ones because it learned a shortcut tied to the training length rather than the underlying algorithm.

   * CoT

        We revisit the same kinds of case studies when intermediate tokens are allowed, and ask how the additional sequential workspace changes the complexity class and the practical behavior.

5. Alternative architectures

   We also compare alternative architectures and alternative ways of carrying state, because explicit CoT is not the only way to add inference-time computation.

   One direction stays close to transformers themselves: looped transformers and architectures that rely more heavily on persistent hidden states.

   A second direction compares transformers with recurrent models, including classical RNN-style systems and newer variants such as Mamba, RWKV, TTT and etc., as well as viewpoints in which transformers can be interpreted as RNN-like systems.

   A third direction adds external or auxiliary computation, such as transformers with memory, and RAG.

   The common theme is that these models use hidden state, recurrence, or memory as alternatives to explicit CoT.

6. In-context learning and prompt-space expressiveness

   In-context learning treats the context as a training set and asks whether a frozen transformer can map examples to a prediction without weight updates. Prompt tuning and prefix tuning ask a related but distinct question: how much can a frozen backbone be controlled by learnable or textual prompt tokens? The literature contains both positive universality constructions and negative limitations; the assumptions about the frozen backbone and allowed prompt length are decisive.

## 1. Model Assumptions and Resource Regime

A transformer is not expressive in the abstract; it is expressive relative to a resource regime. For a fixed length $n$, we can view it as a sequence-to-sequence map $f_{\theta,n}:\mathcal X^n\to\mathcal Y^{m(n)}$ or as a recognizer $r_{\theta,n}:\Sigma^n\to\{0,1\}$. The recognizer view matches formal languages $L=\{x:r_{\theta,n}(x)=1\}$, while the seq2seq view matches translation, parsing, retrieval, and next-token prediction. The asymptotic analogue is a circuit family $\{C_n\}$, one circuit per input length. Uniformity means a Turing machine can generate $C_n$ from $1^n$; nonuniformity means there need not be such a generator, so a separate parameter setting may hide length-specific advice. This is why circuit classes such as $\mathsf{AC}^0$ and $\mathsf{TC}^0$ are relevant background objects here, and why $P$ and $NP$ matter only as reference points: $P$ is polynomial-time computation by a Turing machine, while $NP$ is polynomial-time verifiable search. The main question is not "what is a transformer?" but "which resources are fixed, and which are allowed to grow?"

The basic attention layer makes those assumptions explicit. With hidden states $X\in\mathbb{R}^{n\times d}$, a head computes $Q=XW_Q$, $K=XW_K$, $V=XW_V$, and then $\mathrm{Attn}(X)=\mathrm{softmax}((QK^\top)/\sqrt{d_k}+M)V$, where the mask $M$ determines which positions can interact. Masking is a genuine computational assumption. A causal mask gives autoregressive access, a bidirectional mask lets every visible token interact with every other visible token, and stricter masks can force local or unique-choice behavior. Positional encoding is equally structural. Absolute encodings add a position vector to each token, relative encodings bias pairwise distances, and rotary encodings inject position through phase rotation; without positional information, self-attention is permutation-equivariant, so order-sensitive tasks are not directly available. The original encoder-decoder Transformer of [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762) is the baseline reference for this picture, because it already separates source encoding, target decoding, attention masking, and positional information.

Three more axes matter throughout the theory. First, sequence length versus model size: a theorem for a fixed $n$ is a density statement on a finite domain, while a theorem that works uniformly over all $n$ is a genuine asymptotic computation statement. Second, finite versus infinite precision: if each scalar is stored with only $O(1)$ or $O(\log n)$ bits, the model can only carry bounded information per coordinate; if exact reals or idealized unbounded precision are allowed, a single state variable can hide much more structure. Third, one-pass inference versus iterative inference: a model that answers immediately is closer to a bounded circuit, while a model that writes intermediate tokens, loops over hidden state, or uses a scratchpad is closer to a sequential machine. Section 3 will treat chain of thought in detail; here we only need the regime distinction.

The cleanest positive theorem in the fixed-length regime is [Yun, Bhojanapalli, Rawat, Reddi, and Kumar 2020](https://openreview.net/forum?id=ByxRM0Ntvr), which shows that transformers are universal approximators of continuous sequence-to-sequence functions on compact domains, once the architecture has enough width and the positional assumptions break permutation symmetry. The proof strategy is the familiar universality pattern: self-attention first constructs a contextual representation that mixes information across positions, and then position-wise feed-forward layers approximate the desired continuous map on that representation. This is a theorem about density, not about efficient asymptotics. It says that at fixed length the architecture is not missing whole classes of continuous targets, but it does not say that one bounded-size model handles arbitrary lengths. A useful cautionary companion is [Luo, Li, Zheng, Liu, Wang, and He 2022](https://openreview.net/forum?id=NQFFNdsOGD), which shows that relative positional encoding does not automatically restore universality: under their assumptions, some continuous sequence-to-sequence functions remain unapproximable no matter how large the model becomes. Other results in this fixed-length approximation family include [Alberti, Dern, Thesing, and Kutyniok 2023](https://proceedings.mlr.press/v221/alberti23a.html) and [Takakura and Suzuki 2023](https://proceedings.mlr.press/v202/takakura23a.html), along with the prompt-based universality results of [Petrov, Torr, and Bibi 2024](https://proceedings.mlr.press/v235/petrov24a.html) and [Wang, Chauhan, Wang, and Hsieh 2023](https://openreview.net/forum?id=zWxKYyW9ik).

At the opposite end, the bounded-circuit regime gives the cleanest asymptotic upper bounds for constant-depth transformers. A representative theorem is [Merrill, Sabharwal, and Smith 2022](https://aclanthology.org/2022.tacl-1.49/): saturated transformers with floating-point values can be simulated by constant-depth threshold circuits, so their formal-language power is upper-bounded by a $\mathsf{TC}^0$-style computation. The proof idea is to compile each layer into a threshold-circuit description. Saturated attention behaves like a controlled selection or comparison mechanism, bounded-precision arithmetic can be expressed by threshold predicates, and a constant number of layers composes to a constant-depth circuit family. A closely related result is [Hao, Angluin, and Frank 2022](https://aclanthology.org/2022.tacl-1.46/), which places unique-hard-attention and generalized unique-hard-attention transformers in $\mathsf{AC}^0$, while showing that averaging hard attention can recognize some non-$\mathsf{AC}^0$ languages such as $\mathsf{MAJORITY}$ and $\mathsf{DYCK}\text{-}1$ in their setting. Logic characterizations sharpen the same story: [Chiang, Cholak, and Pillay 2023](https://proceedings.mlr.press/v202/chiang23a.html) identify a first-order logic with counting quantifiers that is both an upper and lower bound for fixed-precision transformer encoders, and [Merrill and Sabharwal 2023](https://openreview.net/forum?id=uR8TtWCIsr) show that log-precision transformers admit a first-order characterization with majority quantifiers. The common proof pattern is always the same: bounded precision and bounded depth prevent the model from storing arbitrarily rich state, so attention becomes a form of constant-round counting or thresholding rather than a full sequential algorithm. [Hahn 2020](https://aclanthology.org/2020.tacl-1.11/) is the early limitation theorem that makes this intuition concrete by showing that self-attention cannot model periodic finite-state languages or hierarchical structure unless the number of layers or heads grows with input length.

To move from bounded circuits toward $P$-like or Turing-complete behavior, the decisive change is not just more parameters but more rounds of computation or a richer notion of state. A transformer that is allowed to generate a visible scratchpad, or to loop over its own hidden state, is no longer a one-shot circuit; it has become an iterated computation with externalized workspace. In the idealized exact-arithmetic setting, this can go all the way to Turing completeness. [Perez, Barcelo, and Marinkovic 2021](https://www.jmlr.org/papers/v22/20-302.html) prove that hard-attention transformers are Turing complete, and [Bhattamishra, Patel, and Goyal 2020](https://aclanthology.org/2020.conll-1.37/) give a simpler proof and show that even transformers with positional masking but no positional encoding can be Turing complete under their assumptions. The proof strategy is to encode the machine configuration in dense vectors or position-indexed slots, then use attention, masking, and residual updates to simulate one Turing-machine step per layer or decoding step. This is an exact simulation claim, not a practical claim: it depends on idealized arithmetic, carefully chosen encodings, and unbounded time or length. The point for this section is that the same architecture can live in a circuit regime or a Turing-machine regime depending on how much sequential state and numerical idealization we grant it.

Prompting and prefixing define a different axis again. Parameter expressiveness asks how the function class changes as $\theta$ varies; prompt expressiveness asks how the function class changes as a prompt or prefix $p$ varies while $\theta$ is frozen. Formally, one studies $f_{\theta,p}$ rather than $f_\theta$, so the prompt acts like an additional, inference-time control surface rather than a new set of learned parameters. This is not the same as ordinary fine-tuning, and it is not the same as chain of thought, though the resource logic is similar. The sample theorem here is [Petrov, Torr, and Bibi 2024](https://proceedings.mlr.press/v235/petrov24a.html): prefix-tuning a pretrained transformer can be a universal approximator of sequence-to-sequence functions, and in their construction a single attention head already suffices to approximate any continuous function. The proof idea is that the prefix steers attention into a rich family of basis features, so the frozen backbone plus prefix behaves like a programmable approximator. [Wang, Chauhan, Wang, and Hsieh 2023](https://openreview.net/forum?id=zWxKYyW9ik) complement this with both positive and negative results for soft-prompt tuning: prompt tuning can be universal in stylized settings, but finite-depth fixed-weight models also have hard limitations and lower bounds on the prompt budget. The broader lesson is that inference-time expressiveness is a separate resource from parameter expressiveness; a frozen model with a long or learnable prefix can implement behaviors unavailable to the same model with no prompt, but this is still an assumption about how the model is being used, not a theorem about the backbone alone.

The regime map is therefore the main content of this section. Fixed-length, continuous seq2seq approximation lives in the universality literature; bounded precision and constant depth push transformers into $\mathsf{AC}^0$, $\mathsf{TC}^0$, or first-order logic with counting and majority; iterated generation and explicit scratchpads shift the relevant notion of computation toward $P$ or, under idealized encodings, Turing completeness; and prompt or prefix tuning moves part of the expressive burden from learned parameters into inference-time control. Before asking what transformers can express, we have to say exactly which transformer, with which mask, which positional signal, which precision, which length regime, and which inference protocol.

## 2. No-CoT Transformers as Bounded Parallel Computation

A no-CoT transformer is best understood as a fixed-depth parallel machine. The model reads the entire input, runs a bounded number of attention-and-MLP layers, and produces an answer immediately. There is no visible scratchpad, no loop whose length grows with the input, and no opportunity to write intermediate tokens and read them back later. That is why the right comparison class is not a Turing machine or a general sequential program, but a bounded-depth circuit family or an equivalent logical formalism. CoT changes this resource model by adding visible intermediate state; this section stays on the no-CoT side and asks what one bounded forward pass can express.

To make that precise, fix an input length $n$. A circuit is a directed acyclic graph of gates computing a Boolean function on $n$ input bits. A circuit family $\{C_n\}_{n\ge 1}$ gives one circuit for each length $n$. The depth of $C_n$ is the longest input-to-output path, so bounded depth means $\sup_n \mathrm{depth}(C_n) < \infty$. The size is the number of gates, and the usual expressiveness questions allow polynomial size in $n$. Uniformity matters because otherwise the family can hide arbitrary advice in the description of each $C_n$; informally, a uniform family is one that can be generated from $n$ by a small algorithm. Two standard bounded-parallel classes are $\mathsf{AC}^0$, the class of constant-depth, polynomial-size, unbounded-fan-in AND/OR/NOT circuits, and $\mathsf{TC}^0$, which adds threshold or majority gates and is therefore strictly better at counting and aggregation. On strings, a formal-language recognizer is simply a yes/no computation deciding membership in a language $L \subseteq \Sigma^\ast$ uniformly over all lengths, usually by reading a designated output token or classifier position.

The same idea can be described in logic. First-order logic on strings, usually written $\mathsf{FO}[<]$, quantifies over positions $1,\dots,n$ and can mention the order relation and letter predicates such as "position $i$ contains $a$." Counting quantifiers extend this language by allowing statements of the form $\exists^{\ge k}x\,\varphi(x)$, meaning that at least $k$ positions satisfy $\varphi$. Majority quantifiers allow statements such as $\mathsf{Maj}_x\,\varphi(x)$, meaning that more than half of the positions satisfy $\varphi$. These are not decorative choices of notation. They are the logical mirrors of constant-depth circuits with counting or threshold power, so when transformer results land in $\mathsf{FO}$ with counting or majority, they are saying that a no-CoT transformer behaves like a bounded-parallel counting device rather than a sequential algorithm.

Two modeling assumptions are especially important. First, finite precision: if hidden states and scores have only $O(1)$ or $O(\log n)$ effective bits, then the forward pass can be discretized and analyzed using circuit and logic machinery. Without such a restriction, a single real-valued coordinate can in principle encode unbounded information. Second, the attention rule matters. In hard attention, a head selects a maximizing position by an $\arg\max$-type rule. In soft attention, it takes a softmax-weighted average over all positions. Average-hard and saturated variants sit between these extremes; they are stylized enough to analyze but still expressive enough to illuminate what real transformer layers are doing. With these definitions in place, the main question becomes straightforward: once we treat a no-CoT transformer as a bounded-parallel computation, where does it sit?

### Limitations

The first family of results is negative: fixed-depth transformers cannot robustly express some global counting and nesting patterns unless some resource grows with the input length. A canonical example is [Hahn 2020](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00306/43545/Theoretical-Limitations-of-Self-Attention-in), which shows that fixed-size self-attention cannot capture certain periodic finite-state languages and hierarchical languages unless the number of layers or heads scales with the sequence length. The point is not that transformers mysteriously fail on formal languages; the point is that a bounded number of parallel communication rounds cannot maintain arbitrarily fine global phase or stack-like information. Once the strings are long enough, many inputs that differ only in long-range periodicity or deep nesting become indistinguishable to the fixed architecture.

The proof strategy in this line is usually an indistinguishability argument. One shows that with only finitely many layers, heads, and precision states, the model can only partition long inputs into a bounded collection of coarse interaction patterns. Then one constructs strings that fall into the same pattern even though one should be accepted and the other rejected. For periodic languages, the separator is often exact phase information modulo some period. For hierarchical languages, the separator is unbounded nesting depth. In both cases, the bounded-parallel model runs out of room to propagate the needed dependency through the entire string.

Other major results sharpen the same message. [Hao, Angluin, and Frank 2022](https://aclanthology.org/2022.tacl-1.46/) analyze hard-attention transformers as formal-language recognizers and show that unique-hard-attention variants lie in $\mathsf{AC}^0$, which immediately rules out classic constant-depth separators such as parity. They also show that stronger average-hard variants can recognize tasks outside $\mathsf{AC}^0$, including MAJORITY and DYCK-1, where DYCK-1 is the one-bracket balanced-parentheses language. [Barcelo et al. 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5f0fdc1acd47431f7f3bb8ee85598cef-Abstract-Conference.html) strengthen the picture further by exhibiting an $\mathsf{AC}^0$ language that unique-hard-attention encoders still cannot recognize. Read together, these results say that no-CoT transformers are not generic sequential devices in disguise. They inherit the same separator tasks that organize bounded-depth circuit complexity.

### Circuit Upper Bounds

The second family of results is positive: many transformer variants can be simulated by shallow circuit families. A representative theorem is [Merrill, Sabharwal, and Smith 2022](https://aclanthology.org/2022.tacl-1.49/), which shows that saturated transformers are simulable by constant-depth threshold circuits, placing them naturally in a $\mathsf{TC}^0$-style regime. This is a strong upper bound. It says that even though attention looks like a global operation, once the number of layers is fixed and the numerical behavior is controlled, the whole forward pass can be flattened into a bounded-depth threshold computation.

The proof is constructive and proceeds layer by layer. Each attention head is compiled into a shallow gadget that compares scores, identifies the relevant positions, and routes values to the next layer. The value aggregation step becomes shallow arithmetic, which threshold gates can implement efficiently. Because the number of layers does not grow with $n$, stacking these gadgets preserves constant depth. The proof therefore follows the same logic as many classical circuit simulations: isolate the primitive operations, show that each primitive belongs to a shallow class, and then note that a constant number of shallow stages stays shallow.

Other upper-bound results fit the same template. [Hao, Angluin, and Frank 2022](https://aclanthology.org/2022.tacl-1.46/) place unique-hard-attention models inside $\mathsf{AC}^0$ and show how average-hard attention escapes that class on some examples. [Chiang, Cholak, and Pillay 2023](https://proceedings.mlr.press/v202/chiang23a.html) tie fixed-precision transformer encoders to a counting-logic fragment and therefore to a constant-depth upper-bound picture. [Barcelo et al. 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5f0fdc1acd47431f7f3bb8ee85598cef-Abstract-Conference.html) show that hard-attention encoders with richer positional resources can recognize correspondingly richer logical fragments. The unifying idea is simple: attention is global communication, but it is still only a bounded number of rounds of global communication.

### Logic Characterizations

Circuit upper bounds say what class a transformer computation belongs to. Logic characterizations say the same thing in a more structural language. The key papers here are [Chiang, Cholak, and Pillay 2023](https://proceedings.mlr.press/v202/chiang23a.html), [Merrill and Sabharwal 2023](https://openreview.net/forum?id=W668diqwp4l), and [Merrill and Sabharwal 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a48e5877c7bf86a513950ab23b360498-Abstract-Conference.html). Their shared claim is that, under finite-precision or $\log n$-precision assumptions, transformer computations can be expressed in first-order logic enriched with counting or majority. This matters because logic makes explicit what information the model is allowed to aggregate in one bounded pass: local predicates over positions, bounded-depth compositions of those predicates, and global counting or threshold summaries.

A sample result is [A Logic for Expressing Log-Precision Transformers](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a48e5877c7bf86a513950ab23b360498-Abstract-Conference.html), which characterizes log-precision transformers using first-order logic with majority quantifiers. The proof strategy follows a standard two-step route. First, compile the transformer into a highly uniform threshold circuit. Second, invoke the circuit-to-logic correspondence that matches threshold computation with majority quantification. Each layer becomes a formula describing which positions are reachable, which predicates hold at those positions, and whether sufficiently many of them satisfy some condition. Counting quantifiers express bounded aggregation directly; majority quantifiers express the threshold case directly. In this way, the logic is not an analogy layered on top of the model. It is a re-description of the same bounded-parallel computation.

Other results fill out the landscape. [Transformers Implement First-Order Logic with Majority Quantifiers](https://openreview.net/forum?id=W668diqwp4l) gives a broad first-order-with-majority viewpoint for transformer networks, while [Tighter Bounds on the Expressivity of Transformer Encoders](https://proceedings.mlr.press/v202/chiang23a.html) identifies a counting-logic formalism that both upper-bounds fixed-precision encoders and lower-bounds a more general encoder class. The practical moral is that parity, exact counting, and majority are not arbitrary benchmark choices. They are canonical probes of where a bounded-parallel architecture sits between $\mathsf{AC}^0$, $\mathsf{TC}^0$, and their logical counterparts.

### Exact Characterizations for Restricted Models

Upper bounds and logic embeddings are useful, but the sharpest results are exact characterizations, where a restricted transformer model matches a classical language class exactly. The cleanest example is [Yang, Chiang, and Angluin 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/13d7f172259b11b230cc5da8768abc5f-Abstract-Conference.html), which shows that masked hard-attention transformers with strict masking and no position embeddings recognize exactly the star-free languages. A star-free language is a regular language built using union, concatenation, and complement, but without Kleene star; equivalently, it is a language definable in $\mathsf{FO}[<]$. This is exactly the kind of theorem one wants in a mature expressiveness theory: a concrete architectural restriction lines up with a named classical class.

The proof strategy is especially pedagogical. The authors first translate the transformer into a Boolean sequence-program formalism based on [RASP](https://proceedings.mlr.press/v139/weiss21a.html). They then translate those programs into linear temporal logic, and finally use the classical equivalence between temporal logic, $\mathsf{FO}[<]$, and star-free languages. The architecture is therefore characterized by a chain of exact correspondences rather than by a loose simulation. Once that bridge is established, familiar closure properties and separations from automata theory become directly relevant to transformers.

Other exact or near-exact results for restricted models extend the same philosophy. [Barcelo et al. 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5f0fdc1acd47431f7f3bb8ee85598cef-Abstract-Conference.html) show that hard-attention encoders capture specific logical fragments once unary numerical predicates and, for stronger variants, counting terms are allowed. [Thinking Like Transformers](https://proceedings.mlr.press/v139/weiss21a.html) is important here not because it proves a language-theoretic theorem by itself, but because it provides the intermediate language that makes exact compilation arguments possible. The broader lesson is that exact characterizations usually appear only after the model class has been restricted enough to become rigid. But once that rigidity is in place, the resulting statement is much sharper and easier to interpret than a generic upper bound.

The overall picture is now fairly coherent. In the no-CoT regime, a transformer is best modeled as a uniform bounded-parallel computation. Under stricter hard-attention assumptions it often sits near $\mathsf{AC}^0$; under richer but still discretizable attention it can move toward $\mathsf{TC}^0$; and in logic it corresponds to first-order formalisms with counting or majority. The exact boundary depends on masking, positional information, precision, and the attention rule, but the conceptual boundary is stable: without externally written intermediate tokens, the model gets only a bounded number of global communication rounds. That is the core reason no-CoT transformers are fruitfully studied through circuits, logic, and formal-language recognition rather than through general sequential computation.

## 3. Chain of Thought as a General Computation

Chain of thought changes the resource model. In the no-CoT setting, a transformer reads the input and answers after one bounded forward pass, so the right abstractions are bounded-depth circuits and finite-variable logics. In the CoT setting, the model instead generates an intermediate trace
$$
x \to z_1 \to z_2 \to \cdots \to z_T \to y,
$$
and each new token is computed from the growing context $(x,z_1,\dots,z_{t-1})$. A scratchpad is precisely this visible intermediate trace when it is used as workspace rather than as part of the final task output. The crucial point is that the $z_t$ tokens are not merely text. They are append-only, rereadable state. Once the model can write intermediate symbols and later attend back to them, it is no longer only a bounded parallel map; it has acquired a sequential workspace.

This is also the place where parameter expressiveness and prompt expressiveness must be separated. Parameter expressiveness asks what functions or languages are realizable by some choice of model weights $\theta$. Prompt expressiveness fixes $\theta$ and asks what can be induced by varying the prompt or prefix. Formal CoT expressiveness results usually belong to the first category: they show that there exists a trained or chosen parameter setting that uses generated tokens as a work tape. Ordinary CoT prompting belongs to the second category: the backbone is frozen, and the prompt tries to elicit a computation the model already knows how to perform. Learned soft prompts or prefixes sit in between in a practical sense, but formally they are still trainable parameters rather than pure prompt execution.

Once the number of decoding steps $T$ is allowed to grow with input length $n$, the natural benchmark is no longer $\mathsf{AC}^0$ or $\mathsf{TC}^0$, but polynomial-time computation. A language lies in $\mathsf{PTIME}=\mathsf{P}$ if there is a Turing machine deciding inputs of length $n$ in at most $n^k$ steps for some fixed $k$. A problem is $P$-complete if it is in $P$ and every problem in $P$ logspace-reduces to it. Such problems matter here because they are the standard hard boundary cases once a theorem claims that a CoT-enabled architecture has reached full polynomial-time power. Turing completeness is a stronger claim still: it says that, under the assumptions of the theorem, the model can simulate arbitrary Turing machines. That is an important boundary result, but it is more idealized than the finite-precision $\mathsf{P}$-characterizations.

### 3.1 CoT as Externalized Sequential State

The right way to read CoT, in the formal literature, is therefore not as "reasoning in words" but as computation with visible memory. The generated symbols need not be grammatical language at all. They can be tags, counters, delimiters, or encoded machine states. What matters is that they persist in the context and can be revisited by later attention steps. This is why CoT can fundamentally change expressive power while leaving the basic transformer block unchanged: the architecture is the same, but the inference protocol now includes repeated write-read cycles through the context window.

This distinction also clarifies the relationship between explicit CoT and other ways of carrying state. A recurrent model keeps state in hidden activations. A looped transformer reuses its hidden layers across time. A CoT transformer stores part of its state in visible tokens. These are not identical mechanisms, but they play the same complexity-theoretic role: they add sequential computation to an otherwise bounded-depth parallel map. CoT is the most transparent version because the state is written into the prompt itself.

### 3.2 Formal CoT Hierarchies: From Regular Languages to $\mathsf{P}$

A representative theorem is [Merrill and Sabharwal 2023](https://arxiv.org/abs/2310.07923). Under their projected-pre-norm and generalized-pre-norm transformer assumptions, they show that CoT creates a genuine expressiveness hierarchy. Logarithmic-length CoT gives only a modest extension over the standard decoder-only model. Linear-length CoT is already strong enough to recognize all regular languages. Linear CoT still remains within the context-sensitive languages. Most importantly, polynomial-length CoT characterizes $\mathsf{P}$ exactly in their setting. This is the cleanest formal statement of the basic intuition: once the model can keep writing and rereading workspace for polynomially many steps, its natural computational class becomes polynomial time.

The proof has the standard two halves. For the lower bound, they construct transformers that use the generated scratchpad as a work tape. A key ingredient is a layer-norm-based hashing mechanism that lets the model recover and update previously written state. With that in place, one generated token can represent one step of an automaton or Turing-style computation. For the upper bound, they simulate the whole autoregressive process by an ordinary Turing machine and show that if the number of generated tokens is polynomial, then the total computation is also polynomially bounded. The result is not just that CoT helps; it is that CoT changes the asymptotic class. Other major results in the same paper include the near-no-go character of logarithmic CoT, the context-sensitive upper bound for linear-length traces, and the identification of polynomial-length CoT with standard polynomial-time computation.

This theorem is also where the trainable-weight versus frozen-prompt distinction matters most. The characterization is about what the architecture can realize for some choice of parameters. It is not a theorem that a frozen pretrained language model, prompted with "think step by step," thereby acquires the full power of $\mathsf{P}$. The existence claim belongs to architecture-plus-decoding expressiveness, not to prompt elicitation.

### 3.3 CoT as Serial Circuit Evaluation

A complementary result family makes the same point from the perspective of inherently serial tasks. [Li, Liu, Zhou, and Ma 2024](https://arxiv.org/abs/2402.12875) show that constant-depth, constant-bit transformers with $O(\log n)$-dimensional embeddings can solve any problem computable by Boolean circuits of size $T$ when they are given $T$ CoT steps. Without CoT, the same family is bounded by weak parallel classes such as $\mathsf{AC}^0$ under their assumptions. So the separator is not architectural depth alone; it is the ability to unfold computation across generated intermediate tokens.

Their proof strategy is constructive and easy to interpret. The scratchpad stores intermediate gate values or other partial results, and each decoding step advances the computation by one serial stage. Instead of forcing a full circuit evaluation into one shallow forward pass, the model uses the growing context to carry state from one stage to the next. This gives a direct explanation for why CoT helps on tasks such as iterated squaring, permutation composition, and Circuit Value: these problems are difficult precisely because they require many rounds of dependency propagation. Other major results in this family include the sharp no-CoT upper bounds under constant precision and the empirical demonstrations that the same separator tasks become substantially easier once generated workspace is available.

This viewpoint is pedagogically useful because it isolates what CoT is buying. It is not extra width, and it is not a mysterious "reasoning module." It is additional serial time together with a writable workspace. In complexity terms, CoT lets a constant-depth transformer simulate a long computation by turning many short passes into a sequential program.

### 3.4 Turing Completeness as the Idealized Limit

At the far end of the spectrum are Turing-completeness theorems such as [Perez, Barcelo, and Marinkovic 2021](https://www.jmlr.org/papers/v22/20-302.html) and the related construction of [Bhattamishra, Patel, and Goyal 2020](https://aclanthology.org/2020.conll-1.37/). These papers show that, under idealized assumptions, transformer architectures can simulate arbitrary Turing machines. The assumptions are stronger than in the finite-precision CoT papers: one typically relies on exact or highly expressive numerical encodings, carefully controlled masking, and a construction tailored to machine simulation rather than to realistic pretrained inference.

The proof pattern is standard in computability theory. One encodes the machine configuration, including the control state and tape contents, into the transformer's representations or positions. Attention is used to address the relevant tape location or neighboring cells, and feed-forward updates implement the transition rule. One layer or one decoding step then simulates one Turing-machine step. The importance of these results is conceptual. They show that transformer-like architectures are not inherently confined to shallow-circuit behavior. But they should be read as boundary theorems, not as statements about what ordinary CoT prompting on a frozen model can achieve in practice.

For this reason, Turing completeness and polynomial-time expressiveness should be kept distinct. A $\mathsf{P}$-characterization says that under finite precision and polynomially many steps, the model captures efficient computation. A Turing-completeness theorem says that with stronger idealizations, it can capture arbitrary computation. Both are about general computation, but they answer different questions.

### 3.5 Frozen-Prompt CoT and Empirical Reasoning Are Different Questions

The empirical CoT literature studies something weaker and different. [Wei et al. 2022](https://arxiv.org/abs/2201.11903) showed that few-shot CoT prompting can substantially improve reasoning accuracy in large language models. The support there is not a complexity-theoretic proof but benchmark evidence: prompting the model to emit intermediate steps changes the distribution of computations it performs, often making arithmetic and symbolic tasks easier. This is an execution result for a frozen or nearly frozen backbone, not an architecture-level expressiveness theorem. It shows that the model can be steered into using intermediate text effectively, not that prompting itself freely enlarges the formal function class.

This is why scratchpad training and CoT prompting should not be collapsed into one category. If the model is trained to emit and use a scratchpad, then the weights are being optimized to exploit visible workspace, and the result is closer in spirit to the formal trainable-weight literature. If the model is frozen and only the prompt changes, then CoT is better viewed as elicitation: the prompt selects or stabilizes a computation the backbone can already implement. Unless the prompt itself is a learned high-capacity prefix in a theoretical construction, prompt-space CoT does not by itself prove a new expressiveness theorem.

The clean synthesis is therefore this. In the no-CoT regime, transformers look like bounded parallel machines. In the formal trainable-weight CoT regime, generated tokens become a sequential workspace, and the natural complexity class rises from circuit classes toward $\mathsf{P}$, with Turing completeness as an idealized upper boundary. In the frozen-prompt empirical regime, CoT is a control strategy for a fixed backbone, and the right question is not "what class does the architecture characterize?" but "what computations can the prompt successfully elicit?" Keeping these regimes separate is the only way to make the literature coherent.

## 4. Case Studies

This literature is easiest to read if we separate two regimes from the start.

In the **non-CoT** regime, the model reads the input and answers in one bounded forward pass. The cleanest case studies here are tasks that expose the limits of bounded parallel computation: formal-language recognition, exact counting, associative recall, long-context retrieval, and length generalization. In the **CoT** regime, the model may generate intermediate tokens and then condition on them. The natural case studies shift accordingly: instead of asking what can be done in one pass, we ask what extra computation becomes possible once the model has visible scratch space.

### 4.1 Non-CoT: one-pass transformers

#### Formal languages

**Definition.** Formal-language tasks ask which languages $L \subseteq \Sigma^\ast$ a transformer can recognize uniformly over all input lengths. The main benchmark families should be kept distinct. **Regular languages** are the finite-state languages. **Star-free** languages are the aperiodic regular languages, equivalently those definable in first-order logic over words or in linear temporal logic. **Dyck** languages are balanced-parentheses languages; Dyck-1 is already a stack-like test, and Dyck-$k$ is the standard context-free family when one wants to probe nesting rather than finite-state memory.

A good anchor result is [Yang, Chiang, and Angluin 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/13d7f172259b11b230cc5da8768abc5f-Abstract-Conference.html): masked hard-attention transformers with strict masking and no position embeddings recognize **exactly** the star-free languages. This is stronger than a one-sided upper or lower bound. The proof strategy is especially pedagogical. They introduce Boolean RASP as an intermediate language, show how this transformer family and Boolean RASP simulate each other, and then connect Boolean RASP to linear temporal logic, which is already known to characterize the star-free languages. So the theorem works by reducing transformer expressiveness to a better-understood symbolic formalism and then importing the classical logic result.

This family also contains the main positive and negative separator results. [Strobl et al. 2024](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00663/120983/What-Formal-Languages-Can-Transformers-Express-A) survey the area and, importantly, organize the many apparently conflicting claims by architecture, positional encoding, masking, precision, and recognition definition. [Bhattamishra, Ahuja, and Goyal 2020](https://aclanthology.org/2020.emnlp-main.576/) give constructive results for nested and counter-style languages, including Dyck-style examples. [Weiss, Goldberg, and Yahav 2021](https://proceedings.mlr.press/v139/weiss21a.html) make the symbolic side concrete with RASP programs for tasks such as Dyck recognition and histograms. [Lindner et al. 2023](https://arxiv.org/abs/2301.05062) push this compilation viewpoint further with Tracr, which is useful because it turns transformer programs into controlled laboratory objects. The broader lesson is that "formal languages" is not one benchmark family but several: regular, star-free, and Dyck probe genuinely different memory resources.

#### Counting

**Definition.** Counting tasks ask the model to maintain exact global multiplicities, not just detect presence or local patterns. Canonical examples are token histograms, majority, parity, or "how many times has symbol $a$ appeared so far?" These are simple to state but often hard for bounded-depth architectures because they require a stable global summary.

A representative result is [Yehudai et al. 2024](https://arxiv.org/abs/2407.15160), which studies the task of counting token occurrences and identifies a sharp phase transition controlled by embedding dimension and vocabulary size. Their positive construction shows that when the representation dimension is at least as large as the vocabulary, a transformer can maintain exact counts. Their negative story is equally important: when vocabulary size exceeds dimension, non-orthogonal token representations interfere, the required weights scale badly, and the exact counting solution becomes numerically unstable and practically unlearnable. The paper then validates this experimentally by varying vocabulary size, context length, and model size and showing the predicted drop in both in-distribution and out-of-distribution performance.

Counting also appears throughout the circuit-style expressiveness literature because it is the smallest family that already separates weak parallel classes. In practice, histogram tasks in [Weiss, Goldberg, and Yahav 2021](https://proceedings.mlr.press/v139/weiss21a.html) serve as a useful symbolic counting benchmark, while majority and parity recur in logic and circuit characterizations from earlier sections. The survey point is that counting should be treated as its own case-study family rather than folded vaguely into "algorithmic reasoning": exact aggregation is the issue.

#### Associative recall

**Definition.** Associative recall asks whether the model can store key-value associations in context and retrieve the correct value for a queried key. A typical input looks like $(k_1,v_1),\dots,(k_m,v_m),q$, and the target is the value paired with $q$. This is the smallest synthetic form of factual recall, entity binding, and content-addressed retrieval.

A useful anchor result is [Nichani, Lee, and Bietti 2024](https://openreview.net/forum?id=PtYojIoW0u), which analyzes factual recall through the lens of associative memory. They show that a shallow transformer with one self-attention layer followed by an MLP can achieve perfect recall on a synthetic factual-recall task whenever either the attention parameters or the MLP parameters scale essentially linearly with the number of facts, up to log factors. The proof idea is to decompose the model into two possible storage mechanisms: the value matrices can act as an associative memory, or the MLP can do so. The theorem is therefore not just a capacity statement; it identifies where the memory can live inside the architecture.

This family also has a strong mechanistic literature. [Olsson et al. 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) identify induction heads, which are not identical to general associative recall but are a nearby circuit for copying and continuing repeated patterns. That matters because it shows one concrete retrieval mechanism learned by real models. Taken together, the theoretical associative-memory papers and the induction-head analyses explain why recall tasks sit naturally between formal expressiveness and mechanistic interpretability: they are simple enough to model exactly, but rich enough to reveal real circuits.

#### Long-context retrieval and NIAH

**Definition.** Long-context retrieval asks whether the model can find and use relevant information when the context is so long that positional encoding, distractor suppression, and attention scaling become part of the problem. The basic probe is **needle in a haystack (NIAH)**: one relevant fact is hidden in a long distractor context, and the model must retrieve it.

The right representative result here is [Hsieh et al. 2024](https://arxiv.org/abs/2404.06654), because it explains why vanilla NIAH is only a starting point. Their benchmark, RULER, explicitly expands NIAH to multiple needles, different needle types, multi-hop tracing, and aggregation tasks. The experimental logic is simple and convincing: many long-context models score nearly perfectly on the vanilla needle test, but performance drops sharply once the context becomes longer or the task requires more than single-fact retrieval. So the paper does not prove a theorem in the formal-language sense; instead, it isolates the gap between "can retrieve one fact somewhere" and "can robustly use long context."

[LongBench](https://arxiv.org/abs/2308.14508) complements this by widening the task distribution to multi-document QA, summarization, code, synthetic tasks, and bilingual long-context understanding. The main organizational point is that NIAH and long-context retrieval are not the same family. NIAH is the minimal sanity check; richer retrieval benchmarks test whether the model can locate, combine, and aggregate information under realistic long-context pressure.

#### Length generalization

**Definition.** Length generalization asks whether a model trained on shorter sequences continues to solve the same task on longer ones. This is not the same as long-context retrieval. Here the issue is not only whether the context fits, but whether the learned solution is algorithmic rather than tied to training length.

A representative result is [Zhou et al. 2024](https://openreview.net/forum?id=AssIuHnmHX), which frames the problem using RASP-L. Their central claim is that length generalization correlates with the existence of a short RASP-L program for the target task. The logic is partly theoretical and partly experimental: if a task admits a short transformer-native symbolic description, then models trained from scratch are more likely to learn a solution that extrapolates in length. The same paper also gives a clean lesson about scratchpads: a scratchpad helps when it simplifies the underlying program, and hurts when it makes the induced program more complicated.

This line is best read together with [Weiss, Goldberg, and Yahav 2021](https://proceedings.mlr.press/v139/weiss21a.html), which introduced RASP as a language for writing transformer programs, and [Lindner et al. 2023](https://arxiv.org/abs/2301.05062), which compile such programs into exact transformers. Those papers do not themselves solve the empirical generalization problem, but they provide the symbolic vocabulary for stating it precisely. The survey takeaway is that length generalization is not only about model scale or context window. It is about whether training finds the short algorithmic solution at all.

### 4.2 CoT: transformers with intermediate workspace

The CoT setting changes the resource model. Once the model may generate intermediate tokens and then attend to them, the question is no longer "what fits in one bounded parallel pass?" but "what can be built by repeated state updates?" This is why the canonical CoT case studies are serial separator tasks.

#### CoT as a complexity shift

The cleanest formal result is [Merrill and Sabharwal 2023](https://arxiv.org/abs/2310.07923). They show that the expressive gain from CoT depends strongly on how many intermediate steps are allowed. In their framework, logarithmic CoT gives only a modest extension, linear CoT already reaches all regular languages under projected pre-norm assumptions, and polynomially many steps characterize polynomial-time computation. The proof idea is to treat the generated scratchpad as an external work tape and then prove matching upper and lower bounds as the scratchpad length grows. This is the formal reason CoT deserves its own subsection: it is not just a prompting trick, but a different computational regime.

This perspective also clarifies the simplest CoT formal-language example. In the no-CoT regime, standard transformers have sharp limits even on regular-language recognition. In the CoT regime, the decoder can explicitly carry forward automaton-like state in generated tokens. That is the smallest example of visible sequential workspace changing what the architecture can do.

#### Canonical separator tasks: permutation composition, iterated squaring, circuit value

The most direct benchmark paper here is [Li, Liu, Zhou, and Ma 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3309b4112c9f04a993f2bbdd0274bba1-Abstract-Conference.html). Their theorem shows that constant-depth, constant-bit transformers with $T$ CoT steps can solve any problem computable by Boolean circuits of size $T$. The constructive proof simulates circuit computation step by step through generated intermediate tokens, so the theorem directly formalizes the intuition that CoT adds serial computation to an otherwise shallow parallel architecture.

Their experiments are just as important for organizing the case studies. They evaluate modular addition, permutation composition, iterated squaring, and circuit value. The first is a control task that already lies in a weak parallel class; the other three are the real separators. **Permutation composition** asks the model to repeatedly update a state by composing permutations, so each step depends on the previous one. **Iterated squaring** is the same repeated-update pattern in an arithmetic form. **Circuit value** is the classical complexity-theoretic separator: evaluate a Boolean circuit on a given input. The experimental finding is exactly what the theory predicts: low-depth transformers without CoT struggle badly on the serial tasks, while the same models improve sharply once CoT is allowed.

These tasks are canonical for a reason. They do not merely demand "reasoning" in a loose sense. They isolate repeated dependency propagation, which is precisely the kind of computation that bounded parallel depth handles poorly.

#### Arithmetic derivations and dynamic programming

A complementary representative result is [Feng et al. 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dfc310e81992d2e4cedc09ac47eff13e-Abstract-Conference.html). They prove impossibility results for direct-answer bounded-depth transformers on basic arithmetic and equation-solving tasks unless the model size grows super-polynomially with input length. They then give constructive upper bounds showing that autoregressive transformers can solve the same tasks by generating CoT derivations in a suitable math-language format. The proof template is especially clean for a tutorial: first a lower bound for direct prediction, then a constructive algorithm in the scratchpad regime.

The same paper also extends the argument to dynamic programming. That matters because it broadens the family beyond school arithmetic. The point is not that CoT is only useful for "math," but that many tasks with an iterative dependency graph become natural once the model can externalize intermediate states. In this sense, arithmetic, equation solving, dynamic programming, permutation composition, and circuit value are all instances of one larger CoT family: tasks whose difficulty lies in serial update rather than one-shot pattern matching.

The big picture is therefore simple. In the non-CoT setting, the best case studies are tasks that expose the limits of bounded parallel computation: formal languages, counting, associative recall, long-context retrieval, and length extrapolation. In the CoT setting, the best case studies are tasks that expose the value of extra sequential workspace: regular-language recognition via scratchpad, arithmetic derivations, dynamic programming, permutation composition, iterated squaring, and circuit value. Keeping these two regimes separate makes the literature much easier to read, and it prevents a common confusion: the same transformer architecture can sit in very different expressiveness classes depending on whether it must answer immediately or may first write down intermediate state.

## 5. Alternative Architectures

Explicit chain of thought adds computation by writing intermediate state into visible tokens. That is only one way to get extra test-time computation. A useful organizing question is: **where does the scratchpad live instead?**

There are four main answers.

- In **looped depth**: the same transformer block is applied repeatedly, so computation unfolds over iterations rather than over newly generated tokens.
- In a persistent **hidden state**: a recurrent model carries latent state from one token to the next.
- In **external memory**: the model reads from and writes to a separate store, or retrieves from an external corpus.
- In **fast weights**: the model updates a small inner learner at test time, so part of the state lives in parameters rather than activations.

The common theme is the same as in CoT: more computation requires more state. What changes is whether that state is visible text, latent activations, memory slots, or temporary weights.

### 5.1 Looped Transformers: Reusing Depth as Time

A **looped transformer** reuses the same block of layers for multiple iterations. If we write the hidden sequence at iteration $t$ as $H^{(t)}$, then a looped block has the form
$$
H^{(t+1)} = B_\theta(H^{(t)}),
$$
where the parameters $\theta$ are shared across iterations. This makes depth behave like time: each loop is another round of computation on the same latent state.

The clean conceptual starting point is the [Universal Transformer](https://arxiv.org/abs/1807.03819), which introduced recurrence across depth. The key expressiveness point is that looping adds computation without forcing the model to emit intermediate tokens.

**Representative result.** [Giannou et al. 2023](https://proceedings.mlr.press/v202/giannou23a.html) show that looped transformers can be programmed as general-purpose computers.

**Proof idea.** The proof is constructive. The input is arranged so that some tokens act like instructions and others act like memory. One pass of the shared transformer block simulates one step of an abstract machine: attention routes information between the relevant locations, and the feed-forward sublayers implement the local state transition. Repeating the block then simulates repeated machine steps. The important takeaway is that the extra computational power comes from **iterating on hidden state**, not from generating a visible scratchpad.

**Other major results.** The original [Universal Transformer](https://arxiv.org/abs/1807.03819) motivated this architecture class, and recent formal work shows that looping can raise expressiveness even when the model remains much more parallel than CoT. In particular, [Merrill and Sabharwal 2025](https://arxiv.org/abs/2505.18948) analyze looping together with padding and show exact characterizations in terms of $\mathsf{TC}^d$ classes. This is useful because it separates two resources cleanly: padding increases available workspace, while looping increases available depth.

### 5.2 Recurrent Hidden State: Computation Stored Latently

A **hidden state** is a latent vector or collection of vectors carried forward during inference. A **recurrent computation** has the form
$$
h_{t+1} = F_\theta(h_t, x_t), \qquad y_t = G_\theta(h_t),
$$
so the model processes a sequence by updating internal state one step at a time. Here the scratchpad is not written into tokens at all; it stays inside $h_t$.

This class includes classical RNNs, gated variants such as LSTMs and GRUs, and newer architectures that make the recurrent structure more explicit or more scalable.

**Representative result.** [Siegelmann and Sontag 1995](https://doi.org/10.1006/jcss.1995.1013) prove that recurrent neural networks are Turing complete under idealized real-valued precision assumptions.

**Proof idea.** The proof encodes the configuration of a Turing machine into continuous hidden-state values. The recurrent update is then designed so that one RNN step corresponds to one machine transition: decode the current control state and tape symbol, update them, and re-encode the new configuration back into the hidden state. This shows, at least in principle, that a latent hidden state can play the same computational role as an explicit work tape.

That positive result is idealized, so the next question is what happens under practical precision constraints.

**Other major results.** [Weiss, Goldberg, and Yahav 2018](https://aclanthology.org/P18-2117/) show that finite-precision recurrent models fall into a hierarchy: some variants can implement counting, while others cannot. [Merrill et al. 2020](https://aclanthology.org/2020.acl-main.43/) refine this into a broader hierarchy of recurrent architectures for language recognition. These papers matter because they show that "recurrent" is not one expressiveness class; gates, precision, and update rules matter.

A particularly important bridge back to transformers is [Katharopoulos et al. 2020](https://proceedings.mlr.press/v119/katharopoulos20a.html), which shows that **linear attention** can be written recurrently. After kernelizing attention, the model can maintain running prefix statistics such as
$$
S_t = S_{t-1} + \phi(k_t) v_t^\top, \qquad z_t = z_{t-1} + \phi(k_t),
$$
and compute the current output from $(S_t, z_t, q_t)$. So some transformers are literally recurrent at inference time once the attention mechanism is rewritten in the right algebraic form.

This viewpoint helps place newer architectures. [RWKV](https://arxiv.org/abs/2305.13048) is an explicit RNN/attention hybrid in which the state tracks decayed key-value summaries, so it behaves like a recurrent model at inference while retaining transformer-like training structure. [Mamba](https://arxiv.org/abs/2312.00752) is a **selective state-space model**: the recurrent state update is input-dependent, so the model can decide what to keep and what to forget based on the current token. The theoretical message in recent work is nuanced. [Cirone et al. 2024](https://arxiv.org/abs/2402.19047) argue that selectivity substantially increases what deep state-space models can represent, while [Merrill, Petty, and Sabharwal 2024](https://arxiv.org/abs/2404.08819) show formal limitations for state-tracking tasks. The right conclusion is not that recurrent hidden-state models are uniformly stronger or weaker than transformers, but that they trade visible workspace for a compressed latent state whose power depends sharply on the update rule.

### 5.3 External Memory and Auxiliary Workspace

An **external memory** is a storage object distinct from the model's ordinary hidden state. If the hidden state is $h_t$, then a memory-augmented model typically has an additional state
$$
M_{t+1} = U_\theta(M_t, h_t, x_t),
$$
together with a read operation that lets the controller access selected parts of $M_t$. The point is to avoid forcing all relevant information through a fixed-size latent vector.

The simplest version is not a separate memory module at all, but **auxiliary workspace tokens**. Padding tokens and pause tokens give the model extra slots in which to stage intermediate variables.

**Representative result.** [Merrill and Sabharwal 2025](https://arxiv.org/abs/2505.18948) show that padding can strictly increase transformer expressiveness, and characterize padded transformers exactly in terms of circuit classes. Closely related, [London and Kanade 2025](https://arxiv.org/abs/2505.21024) show that pause tokens strictly increase the expressivity of constant-depth transformers.

**Proof idea.** The proofs treat extra blank tokens as additional workspace. With enough padding, the transformer can store intermediate quantities across many positions and use them to implement reductions from complete problems for majority-logic and threshold-circuit classes. In other words, padding is not just a convenience for alignment; it is extra computational space. This is the cleanest formal example of the broader idea that **memory slots can replace explicit CoT tokens**.

**Other major results.** In the broader memory-augmented literature, [Neural Turing Machines](https://arxiv.org/abs/1410.5401) introduced differentiable read/write memory controlled by a neural network, and later systems such as memory networks and differentiable neural computers push the same idea further. Transformer-native versions appear in work such as [Token Turing Machines](https://arxiv.org/abs/2211.09119), where a transformer controller reads and writes a bounded set of memory tokens across time.

This is also the right place to define **retrieval-augmented generation (RAG)**. In RAG, the model does not write to external memory; it **retrieves** from a read-only corpus and then conditions generation on the retrieved documents. Formally, one can think of the model as first selecting a latent document variable $z$ from a database and then generating from $(x,z)$, as in [Lewis et al. 2020](https://arxiv.org/abs/2005.11401). From an expressiveness viewpoint, this enlarges the effective state available at inference time. The generator no longer has to store all relevant facts in weights or prompt tokens alone. The caveat is that the formal power of RAG depends heavily on the retriever and the corpus: unlike the cleaner circuit-style theorems above, the computational story is now about a **composed system** rather than a single parametric backbone.

### 5.4 Fast Weights and Test-Time Training

A final alternative keeps the scratchpad in **weight space**. The idea goes back to fast weights, but recent work makes it explicit through **test-time training (TTT)**. Here the model updates a small inner learner during inference:
$$
W_{t+1} = W_t - \eta \nabla \ell(W_t; x_t), \qquad y_t = f_{W_t}(x_t).
$$
So the temporary state is not just a hidden vector $h_t$; it is a parameter matrix or small model that changes online.

This deserves to be separated from ordinary recurrence. In an RNN, the state is a vector passed through a fixed update rule. In TTT, the state is itself a learned predictor, and inference includes an optimization step.

**Representative result.** [Sun et al. 2024](https://arxiv.org/abs/2407.04620) introduce TTT layers and make the core expressiveness claim explicit: a model can use test-time learning itself as the hidden-state mechanism.

**Proof idea.** The key argument is by unrolling the inner update. Each token does not merely update activations; it changes the fast weights of a small inner model. That gives the architecture a much richer state space than a fixed-width hidden vector, because the current "memory" is a function approximator learned on the fly from the recent context.

**Other major results.** The older fast-weights view appears in [Ba et al. 2016](https://arxiv.org/abs/1610.06258). More recent theory strengthens the connection to in-context computation: [Gozeten et al. 2025](https://arxiv.org/abs/2503.11842) prove that test-time training can strictly improve transformers as in-context learners in a stylized setting. The conceptual message is clear even when the formal assumptions are specialized: if the model can update an inner learner at inference time, then part of the computation moves from forward propagation into the update rule itself.

### 5.5 What This Section Adds

These architecture classes should not be collapsed into a single slogan such as "more memory helps." They help in different ways.

Looping adds **more latent computation rounds**. Recurrent models add a **persistent hidden state**. Memory-augmented models add a **separate storage mechanism**. RAG adds **read-only external knowledge**. TTT adds **fast adaptive weights**. All are alternatives to explicit CoT because all provide extra state or extra computation at inference time without requiring the model to spell out intermediate reasoning in text.

For expressiveness theory, the right comparison is therefore not just transformer versus non-transformer. The sharper question is: **where is the scratchpad stored, how is it updated, and what precision or resource bounds are assumed?** That is what determines whether the model behaves like a bounded parallel circuit, a recurrent state machine, a memory-augmented controller, or a more general adaptive system.

## 6. In-Context Learning and Prompt-Space Expressiveness

This section studies a different notion of expressiveness from the one used in Sections 2 to 5. There, the main question was what functions can be represented by changing the model weights. Here, the backbone is typically fixed, and the question is what can be achieved by changing only the context or a small prompt-like interface.

Two ideas must be separated from the start.

First, **in-context learning (ICL)** asks whether a transformer can use examples in its context as a temporary training set. A typical prompt has the form
$$
D=((x_1,y_1),\ldots,(x_k,y_k),x_{\mathrm{query}}),
$$
and the model should output a good prediction for $x_{\mathrm{query}}$. No parameter update occurs during this process. If the model is really "learning" here, then the learning algorithm must be implemented inside the forward pass.

Second, **prompt-space expressiveness** asks what functions can be induced when the backbone is frozen and only a prompt is allowed to vary. If $B$ is a frozen transformer and $\Pi$ is the allowed prompt family, then the induced function class is
$$
\mathcal F_{\mathrm{prompt}}(B,\Pi)
=
\{x \mapsto B(p,x): p\in\Pi\}.
$$
This is usually much smaller than the full trainable-weight class
$$
\mathcal F_{\mathrm{weights}}
=
\{x \mapsto B_\theta(x): \theta\in\Theta\}.
$$

The main prompt interfaces are as follows. **Prompt tuning** or **soft prompting** learns continuous prompt embeddings prepended to the input while keeping the backbone fixed; see [Lester, Al-Rfou, and Constant 2021](https://aclanthology.org/2021.emnlp-main.243/). **Prefix tuning** is a stronger interface in which learned vectors are injected as virtual key/value states, often at every layer, again with a frozen backbone; see [Li and Liang 2021](https://aclanthology.org/2021.acl-long.353/). The fixed model itself is the **frozen backbone**.

This is also different from **chain of thought (CoT)**. CoT enlarges the model's test-time workspace by letting it generate intermediate tokens and then attend to them. Prompt tuning and prefix tuning do not change the weight matrix and, by themselves, do not create new sequential workspace in the same sense. So trainable-weight expressiveness, prompt-space expressiveness, and CoT-based expressiveness are related, but they are not the same question.

A useful formal phrase is that a transformer **implements a learning algorithm in-context** if there exists an algorithm $\mathcal A$ such that, on prompts $D$ drawn from a task family, the transformer output agrees with $\mathcal A(D)(x_{\mathrm{query}})$. In that view, the model's hidden states act as temporary parameters, sufficient statistics, or optimizer state.

With these distinctions in place, the literature splits naturally into three families: ICL as implicit learning, prompt universality, and prompt limitations.

### 6.1 In-Context Learning as Implicit Learning

This family asks what algorithm, if any, is being executed inside the forward pass when the model receives labeled examples in context.

A clean starting point is [Garg, Tsipras, Liang, and Valiant 2022](https://openreview.net/forum?id=flNZJ2eOet), which set up ICL as a controlled learning problem: can a transformer trained on prompts of the form $(x_i,f(x_i))$ learn unseen functions from a target class at test time? They show empirically that transformers can in-context learn linear functions, sparse linear functions, small neural networks, and decision trees, often matching task-specific baselines. This paper is not yet a mechanistic theorem, but it makes the question precise.

**Representative result.** [von Oswald et al. 2023](https://proceedings.mlr.press/v202/von-oswald23a.html) show that a single linear self-attention layer can be made equivalent to one step of gradient descent for linear regression. This is one of the sharpest formal statements of the "ICL as learning algorithm" viewpoint.

**Proof idea.** The construction identifies coordinates of the residual stream with quantities such as the current parameter vector, residuals, and aggregated statistics like $X^\top X$ and $X^\top y$. Self-attention is used to collect information from all support examples, and the following linear maps implement the update
$$
w_{t+1}=w_t-\eta X^\top(Xw_t-y).
$$
So the layer is not merely fitting the input-output relation by a static lookup. It is explicitly carrying out an optimizer step inside the forward computation.

This point is sharpened in [Akyurek et al. 2023](https://openreview.net/forum?id=0g0X4H8yN4I), which argue that transformers can implicitly implement standard estimators such as gradient descent and ridge regression by storing a context-dependent model in their hidden representations and updating it as new examples arrive. In this language, the prompt is acting as a dataset and the activations are acting as temporary learned parameters.

**Other major results in this family.** [Mahankali, Hashimoto, and Ma 2024](https://openreview.net/forum?id=8p3fu56lKc) prove that, for one-layer linear self-attention trained on noisy linear regression, the global optimum implements one step of gradient descent under isotropic Gaussian covariates and preconditioned gradient descent under anisotropic covariates. [Fu et al. 2023](https://openreview.net/forum?id=dE5MEi9906) argue that trained transformers can implement higher-order methods close to iterative Newton updates, and prove that $k$ Newton steps can be implemented with $O(k)$ layers. [Cheng, Chen, and Sra 2024](https://openreview.net/forum?id=ah1BlQcLv4) extend the picture beyond linear tasks and show that transformers can implement functional gradient descent for certain nonlinear in-context learning problems.

The main takeaway is that ICL theory is not only about "pattern matching from examples." In the positive results above, the transformer is expressive enough to realize an actual learning rule at inference time.

### 6.2 Prompt Universality

The second family asks a different question. Suppose the backbone is frozen and only the prompt may change. How large can the induced function class $\mathcal F_{\mathrm{prompt}}(B,\Pi)$ be?

**Representative result.** [Petrov, Torr, and Bibi 2024](https://proceedings.mlr.press/v235/petrov24a.html) prove a strong universality statement for prefix tuning. In particular, they show that prefix-tuning a single attention head is enough to approximate any continuous function, and that a transformer whose depth is linear in the sequence length can approximate arbitrary sequence-to-sequence functions.

This is a striking result because it shows that a sufficiently expressive frozen transformer can be "programmed" through its prefix alone. In that sense, prompt-space expressiveness can be very large even when trainable-weight expressiveness is unavailable.

**Proof idea.** The proof is constructive. The prefix is chosen so that the fixed attention mechanism acts as a programmable routing and interpolation device. First one shows that a small prefixed attention module can approximate scalar continuous functions. Then one lifts this construction to sequence-to-sequence maps by composing such modules across positions and layers. The same paper also gives Jackson-type bounds relating the approximation error to the needed prefix length.

A related universality result appears in [Wang, Chauhan, Wang, and Hsieh 2023](https://openreview.net/forum?id=zWxKYyW9ik), which studies soft prompt tuning rather than prefix tuning. Their positive theorem shows that, for a sufficiently strong fixed transformer, prompt tuning can approximate any Lipschitz sequence-to-sequence function. This is conceptually important because it says universality is not restricted to full fine-tuning.

**Other major results in this family.** The theoretical prompt-universality literature is still small, and the two papers above are the central references. On the empirical side, [Lester, Al-Rfou, and Constant 2021](https://aclanthology.org/2021.emnlp-main.243/) show that soft prompt tuning becomes increasingly competitive as model scale grows, while [Li and Liang 2021](https://aclanthology.org/2021.acl-long.353/) introduce prefix tuning and show that it can match full fine-tuning on some generation tasks with far fewer trainable parameters. These empirical papers are not universality theorems, but they motivate why the frozen-backbone regime is worth studying theoretically.

The right conclusion is therefore conditional. Prompts can be universal controllers only when the frozen backbone already has the computational structure needed for the target function.

### 6.3 Prompt Limitations

The third family studies the opposite question: what cannot be changed by prompting or prefix tuning when the backbone is frozen?

**Representative result.** [Petrov, Torr, and Bibi 2023](https://openreview.net/forum?id=GYOXIRXI7W) prove that prompting and prefix tuning are strictly less expressive than full fine-tuning. Their core structural claim is that context-based fine-tuning cannot arbitrarily change the relative attention pattern over the content tokens, and can only bias an attention layer's output in a restricted direction.

This gives a precise sense in which prompt-space expressiveness can be fundamentally smaller than trainable-weight expressiveness. A prompt can often select or combine capabilities already present in the model, but it may fail on tasks that require a genuinely new attention pattern.

**Proof idea.** The proof isolates quantities that remain invariant when the content-processing weights are frozen. Since the query-key maps for the content tokens are fixed, the geometry of content-content attention is largely fixed as well. Adding prompt or prefix vectors can modify scores and outputs through extra interactions with the prompt tokens, but it cannot freely rewrite the relative ordering among all content-token pairs. Any task that requires a new dependency structure among content tokens therefore lies outside the prompt-induced function class.

This limitation result is complemented by [Wang et al. 2023](https://openreview.net/forum?id=zWxKYyW9ik). In the same paper that proves universality for strong frozen transformers, they also show lower bounds for limited-depth backbones: there are datasets that a prompt of any length cannot memorize for a single encoder layer, and they derive lower bounds on the number of tunable prompt parameters needed in related settings. For deeper models, they also identify conditions under which only tasks induced by invertible functions remain learnable.

**Other major results in this family.** The central negative message is shared across the papers just cited: prompting is powerful when it can exploit existing primitives in the backbone, but weak when the task requires altering the model's internal attention geometry or computation graph. This is the precise sense in which prompting differs from full weight training.

### 6.4 Synthesis

The three families fit together cleanly.

In-context learning asks whether the forward pass itself can act as a learning procedure on examples in the prompt. Prompt universality asks how large the induced function class can be when the backbone is frozen and only prompt parameters vary. Prompt limitations identify the invariants that survive all such prompt choices.

Keeping these apart avoids a common confusion. A transformer may be expressive enough to implement gradient descent in-context, yet a frozen instance of that transformer may still be impossible to retarget to a new task by short prompts alone. Conversely, a strong universality theorem for prefix tuning does not say that ordinary textual prompts on a practical model are automatically universal. And none of these statements is the same as a CoT statement, because CoT changes the inference-time resource model by adding intermediate workspace.

So the correct conceptual picture is layered: trainable weights determine the largest architecture-level function class; prompts expose only a restricted control surface over a frozen backbone; and ICL studies whether that control surface can itself implement a learning algorithm on the fly.
