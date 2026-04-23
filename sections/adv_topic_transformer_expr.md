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

Before asking what transformers can express, we have to specify the resource regime. For transformers, expressiveness is not a single property of an architecture name. It depends on what counts as input, what grows with sequence length, how positions are represented, what numerical precision is allowed, and whether inference is a single bounded forward pass or an iterative procedure that writes intermediate tokens.

At a high level, this section does three things:

- define the basic computational objects and complexity classes that appear later;
- separate the main assumption axes that change the answer;
- explain why different assumptions move transformers between fixed-length universality, bounded-circuit regimes such as $\mathsf{AC}^0$ and $\mathsf{TC}^0$, and more sequential regimes that can reach polynomial-time or Turing-complete behavior in idealized settings.

### Core definitions

A useful starting point is to distinguish the two main mathematical views of a transformer.

- `Seq2seq map`: for fixed input length $n$, a transformer can be viewed as a function
  $$
  f_{\theta,n}:\mathcal X^n\to\mathcal Y^{m(n)}
  $$
  or, in fixed-length tasks, $f_{\theta,n}:\mathcal X^n\to\mathcal Y^n$. This is the natural viewpoint for translation, tagging, retrieval, or next-token prediction.

- `Recognizer map`: for discrete strings over an alphabet $\Sigma$, the transformer may instead be viewed as
  $$
  r_{\theta,n}:\Sigma^n\to\{0,1\},
  $$
  which asks whether a length-$n$ input is accepted. Over all $n$, such a family defines a formal language $L\subseteq\Sigma^\ast$.

These two viewpoints lead to different kinds of theorems. Fixed-length approximation theorems usually study $f_{\theta,n}$ for one $n$ at a time. Formal-language and complexity results study the whole family $\{r_{\theta,n}\}_{n\ge 1}$ and ask what happens uniformly as $n$ grows.

A few complexity-theoretic definitions are therefore unavoidable.

- `Circuit`: a directed acyclic graph of logic gates computing a Boolean function on inputs of one fixed length $n$.
- `Circuit family`: a sequence $\{C_n\}$ of circuits, one for each input length $n$. This is the circuit analogue of an algorithm that works on all lengths.
- `Uniform` versus `nonuniform`: a family is uniform if there is an efficient Turing machine that outputs $C_n$ given $1^n$. It is nonuniform if no such requirement is imposed. This matters because a theorem that allows the parameters or wiring to depend arbitrarily on $n$ is closer to a nonuniform circuit statement than to a single algorithmic procedure.
- `Turing machine`: the standard abstract model of sequential computation, with finite control plus an unbounded tape used as memory.
- `Turing completeness`: an architecture is Turing complete if, under the assumptions of the theorem, it can simulate any Turing machine computation.
- `\mathsf{P}`: the class of languages decidable in polynomial time by a Turing machine.
- `\mathsf{NP}`: the class of languages whose yes-instances have polynomial-size certificates verifiable in polynomial time. $\mathsf{NP}$ is not the main target class in this section, but it is a useful reference point when comparing bounded parallel computation with general sequential computation.
- `\mathsf{AC}^0`: constant-depth, polynomial-size Boolean circuits with unbounded-fan-in AND, OR, and NOT gates.
- `\mathsf{TC}^0`: constant-depth, polynomial-size threshold-circuit families, where a gate can compute a threshold or majority-type predicate. Since attention naturally performs weighted comparisons and aggregation, $\mathsf{TC}^0$ often appears as the right upper-bound class for idealized bounded-depth transformers.

The reason these classes appear is simple: a transformer with fixed depth performs only a bounded number of global communication rounds. That makes it look much more like a parallel circuit than like a sequential machine with reusable memory.

### The basic transformer resource model

For a length-$n$ sequence with hidden states $X\in\mathbb{R}^{n\times d}$, a single attention head computes
$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
$$
followed by
$$
\operatorname{Attn}(X)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V,
$$
where $M$ is the mask. This formula already exposes the main resources.

- `Depth` $L$: number of layers, hence number of global rounds of interaction.
- `Heads` $H$: number of parallel attention channels.
- `Width` $d$: embedding dimension or state size.
- `Sequence length` $n$: number of input positions. This is a problem resource, not the same thing as model size.
- `Precision` $b$: how many bits of information each scalar can effectively store.
- `Inference steps` $T$: whether the model answers in one bounded forward pass or is allowed to run autoregressively for additional steps.
- `Prompt or scratchpad budget` $P$: extra tokens available at inference time, including learned prefixes or generated intermediate tokens.

It is useful to think of the regime as a tuple
$$
\mathcal R=(L,H,d,n,b,T,P).
$$
Different papers keep different parts of $\mathcal R$ fixed and let others grow. Most apparent disagreements in the literature are really differences in which entries of $\mathcal R$ are being treated as resources.

### Structural assumptions that change expressiveness

Some assumptions look architectural but are really computational.

- `Masking`: the mask $M$ determines which positions can read which others.
  - In an encoder, masking is typically bidirectional, so each token can attend to all visible input positions.
  - In a decoder-only model, masking is usually causal, so position $i$ can only use tokens up to $i$.
  - In encoder-decoder models, the encoder is bidirectional while the decoder is causal and can cross-attend to the encoder output.

  These are not minor implementation choices. They change the dependency graph of the computation.

- `Positional information`: without positional encodings, self-attention is permutation equivariant, so it can react to content but not to order in the intended way.
  - Absolute positional encodings attach a position vector to each token.
  - Relative positional encodings represent distances or offsets between positions.
  - Rotary encodings inject position through rotations in query-key space.

  Positional information is therefore a structural assumption about whether the model can distinguish sequence order and distance.

- `Finite vs infinite precision`:
  - Under finite precision, each hidden coordinate carries only bounded information, so the architecture behaves more like a bounded-state machine or a circuit with arithmetic gates.
  - Under idealized exact real arithmetic or unbounded precision, a single vector entry may encode much more information, which can dramatically enlarge the formal expressive power.

These assumptions explain why the same high-level architecture can appear weak in one theorem and extremely strong in another.

### The main regime split: fixed length vs asymptotic computation

The first major split is between fixed-length approximation and uniform computation over unbounded lengths.

- `Fixed-length regime`:
  - Here $n$ is fixed.
  - One studies whether the model family can represent or approximate arbitrary maps on $\mathcal X^n$.
  - This is analogous to universal approximation results for MLPs on compact domains.

- `Asymptotic regime`:
  - Here one asks for a single architecture family that works for all $n$, or equivalently for a recognizer family $\{r_{\theta,n}\}$.
  - This is where formal languages, circuit classes, and Turing-machine comparisons become natural.

This distinction matters because universality at fixed $n$ does not imply algorithmic power over arbitrary lengths. A theorem saying "for every $n$, there exists parameters $\theta_n$ that approximate the target on length $n$" is much weaker than a theorem saying "one uniform construction works for all $n$."

### The second split: one-pass parallel computation vs iterative computation

The second major split is the form of inference.

- `One-pass inference`:
  - The model reads the input and outputs after one bounded forward pass.
  - This is the natural setting for circuit-style upper bounds.
  - Depth is then the number of communication rounds, so constant depth suggests constant-depth circuit classes such as $\mathsf{AC}^0$ or $\mathsf{TC}^0$.

- `Iterative inference`:
  - The model is allowed to generate intermediate tokens, loop, or otherwise reuse its own outputs as state.
  - This introduces a form of sequential workspace.
  - Once intermediate state can be written and reread, the architecture is no longer only a bounded parallel computation.

This section only sets up that distinction. The dedicated chain-of-thought section will analyze the iterative regime in detail.

### The third split: parameter expressiveness vs prompt expressiveness

A third distinction is easy to overlook but important.

- `Parameter expressiveness`: vary the learned weights $\theta$, keep the input format fixed, and ask what functions are representable.
- `Prompt expressiveness`: keep $\theta$ fixed and vary an input-side control object such as a textual prompt, a learned soft prompt, or a prefix $p$. Then the relevant family is $f_{\theta,p}$, not just $f_\theta$.

These are different notions. Parameter expressiveness is closer to what the architecture could realize after training. Prompt expressiveness is closer to what a frozen pretrained backbone can be induced to do at inference time. In practice they interact, but they should not be conflated.

### Representative result class 1: fixed-length universality

`Sample result.` [Yun, Bhojanapalli, Rawat, Reddi, and Kumar 2020](https://arxiv.org/abs/1912.10077) show that transformers are universal approximators of fixed-length sequence functions under suitable assumptions.

`Why it belongs here.` This is the canonical positive theorem for the fixed-length seq2seq regime. It says that, once length is fixed and the architectural assumptions are favorable, transformers are not missing whole classes of continuous sequence functions.

`Proof idea at a high level.` The proof is constructive. Attention is used to move and aggregate information across positions, creating the right contextual representation, and feed-forward blocks then approximate the desired continuous map on that representation, much as in standard universal approximation arguments for feed-forward networks.

`What it does not say.` It does not show that one bounded-size transformer solves arbitrary-length problems, and it does not imply a strong asymptotic complexity class.

`Other results in this category.` [Alberti, Dern, Thesing, and Kutyniok 2023](https://proceedings.mlr.press/v221/alberti23a.html) and [Takakura and Suzuki 2023](https://proceedings.mlr.press/v202/takakura23a.html) study related universality questions. The negative side also matters: [Luo, Li, Zheng, Liu, Wang, and He 2022](https://openreview.net/forum?id=NQFFNdsOGD) show that relative positional encoding does not automatically give universal approximation in the relevant sense.

### Representative result class 2: bounded-depth transformers as low-depth circuits or logic

`Sample result.` [Merrill, Sabharwal, and Smith 2022](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00493/112604/Saturated-Transformers-are-Constant-Depth) analyze saturated transformers and relate them to constant-depth threshold circuits, giving a $\mathsf{TC}^0$-style upper-bound picture.

`Why it belongs here.` This is the clearest expression of the "no-CoT transformer as bounded parallel computation" viewpoint. If depth is fixed and precision is controlled, then each layer is only one more parallel aggregation round, so the overall computation behaves like a constant-depth circuit.

`Proof idea at a high level.` One compiles each transformer layer into an equivalent threshold-style circuit description. Saturated attention behaves like a controlled selection or comparison mechanism, bounded-precision arithmetic can be implemented by threshold predicates, and composing a constant number of layers yields a constant-depth circuit family.

`Other results in this category.` [Hao, Angluin, and Frank 2022](https://aclanthology.org/2022.tacl-1.46/) place unique-hard-attention encoders in $\mathsf{AC}^0$ and show that average-hard-attention can reach some languages outside $\mathsf{AC}^0$, including $\mathsf{MAJORITY}$ and $\mathsf{DYCK}$-1 in their setting. [Chiang, Cholak, and Pillay 2023](https://proceedings.mlr.press/v202/chiang23a.html) give a logic characterization with counting quantifiers for fixed-precision encoders, and [Merrill and Sabharwal 2023](https://openreview.net/forum?id=uR8TtWCIsr) show that log-precision transformers can be expressed in first-order logic with majority quantifiers. [Hahn 2020](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00306/43545/Theoretical-Limitations-of-Self-Attention-in) is the canonical limitation result showing that fixed-size self-attention cannot robustly capture some periodic and hierarchical languages unless resources grow with input length.

### Representative result class 3: idealized Turing-completeness results

`Sample result.` [Perez, Barcelo, and Marinkovic 2021](https://www.jmlr.org/papers/v22/20-302.html) and [Bhattamishra, Patel, and Goyal 2020](https://aclanthology.org/2020.conll-1.37/) give Turing-completeness results for transformers under idealized assumptions.

`Why it belongs here.` These papers sit at the opposite end of the regime map from the $\mathsf{AC}^0$/$\mathsf{TC}^0$ results. They show that if one allows the right kind of positional access, state encoding, and arithmetic idealization, a transformer can simulate general sequential computation.

`Proof idea at a high level.` The construction encodes a Turing-machine configuration into token representations or position-indexed memory slots. Attention, masking, and residual updates are then arranged so that one layer or one decoding step simulates one machine transition. Repeating this simulation yields an arbitrary Turing computation.

`What assumption is doing the work.` The key gain is not merely more parameters. It is the availability of a richer state representation and an effectively unbounded sequential process. That is exactly what separates these theorems from bounded-circuit upper bounds.

`Caution.` These are idealized formal results. They do not mean that practical finite-precision transformers with bounded context and bounded inference depth are automatically universal computers in the same operative sense.

### Representative result class 4: prompt-space expressiveness

`Sample result.` [Petrov, Torr, and Bibi 2024](https://proceedings.mlr.press/v235/petrov24a.html) show that prefix-tuning a pretrained transformer can be a universal approximator of sequence-to-sequence functions under their assumptions.

`Why it belongs here.` This is the cleanest illustration of prompt expressiveness as distinct from weight expressiveness. The backbone is frozen; the control variable is the prefix.

`Proof idea at a high level.` The prefix is used to steer attention into a rich enough family of basis features so that the frozen backbone, together with the prefix, can approximate the target sequence map.

`Other results in this category.` [Wang, Chauhan, Wang, and Hsieh 2023](https://arxiv.org/abs/2305.18787) give both positive universality results and limitations for prompt tuning in stylized transformer settings. The broader lesson is that a frozen model may still have substantial inference-time expressiveness when the prompt interface is sufficiently powerful.

### What assumptions move the model between regimes?

The main transitions can now be summarized cleanly.

- `Toward bounded circuits`:
  - constant depth;
  - one-pass inference;
  - finite or logarithmic precision;
  - no extra sequential workspace;
  - fixed masking pattern.

  These assumptions support upper bounds in terms of $\mathsf{AC}^0$, $\mathsf{TC}^0$, or related logical formalisms.

- `Toward richer sequential computation`:
  - iterative generation or looping;
  - visible or hidden scratchpad state;
  - stronger positional access;
  - idealized arithmetic or more permissive state encoding.

  These are the assumptions that move the model toward polynomial-time or Turing-complete regimes in formal analyses.

- `Orthogonal control through prompting`:
  - frozen weights;
  - expressive prompt or prefix interface;
  - enough prompt length or learned control tokens.

  This does not change the backbone architecture, but it does change the effective function family available at inference time.

### Takeaway

The main lesson of this section is organizational. There is no single answer to "how expressive is a transformer?" If sequence length is fixed, universality theorems are the right language. If depth is bounded and inference is one-shot, circuit and logic classes such as $\mathsf{AC}^0$ and $\mathsf{TC}^0$ become the right comparison. If the model is allowed to iterate, write intermediate state, or exploit idealized encodings, the comparison shifts toward polynomial-time and even Turing-complete computation. Prompting adds another axis by changing what can be done with a frozen backbone. Later sections will study these regimes separately, but the core discipline is already here: always specify the resource model before stating an expressiveness claim.

## 2. No-CoT Transformers as Bounded Parallel Computation

A no-CoT transformer is best read as a bounded parallel computation. The model sees the whole input, runs a fixed number of attention-and-MLP layers, and outputs an answer immediately. There is no visible scratchpad and no length-dependent loop. That is why the natural comparison class is not a general sequential machine, but a shallow circuit or an equivalent logical formalism. CoT is the contrasting regime: once the model can write intermediate tokens and attend to them later, it gains explicit sequential workspace and leaves the bounded-parallel picture.

The basic vocabulary is worth fixing up front.

- `Circuit family`: a sequence $\{C_n\}_{n\ge 1}$ with one circuit for each input length $n$.
- `Bounded depth`: the circuit depth is bounded by a constant independent of $n$, that is, $\sup_n \mathrm{depth}(C_n) < \infty$.
- `Uniformity`: the family is generated by a simple algorithm from $n$, rather than hiding arbitrary advice in the description of each $C_n$.
- `\mathsf{AC}^0`: constant-depth, polynomial-size, unbounded-fan-in AND/OR/NOT circuits.
- `\mathsf{TC}^0`: the same bounded-depth setting, but with threshold or majority gates, so it can express stronger counting behavior.
- `Formal-language recognizer`: a yes/no computation deciding whether a string $x \in \Sigma^\ast$ belongs to a language $L$, uniformly over all input lengths.
- `\mathsf{FO}[<]`: first-order logic over string positions $1,\dots,n$, with order and letter predicates.
- `Counting quantifier`: an expression such as $\exists^{\ge k}x\,\varphi(x)$, meaning that at least $k$ positions satisfy $\varphi$.
- `Majority quantifier`: an expression such as $\mathsf{Maj}_x\,\varphi(x)$, meaning that more than half of the positions satisfy $\varphi$.
- `Finite precision`: hidden states and scores carry only $O(1)$ or $O(\log n)$ effective bits, which prevents a single real number from encoding unbounded information.
- `Hard attention`: a head selects a maximizing position by an $\arg\max$-type rule.
- `Soft attention`: a head outputs a softmax-weighted average over positions.
- `Average-hard` and `saturated` attention: intermediate variants that are easier to analyze than full softmax but richer than strict one-position hard attention.

With these definitions in place, the literature falls into four closely related families: limitations, circuit upper bounds, logic characterizations, and exact characterizations for restricted models.

### Limitations

The negative results say that a fixed-depth transformer cannot robustly express certain global counting or nesting behaviors unless some architectural resource grows with input length. This is the cleanest first-pass interpretation of the no-CoT regime: one bounded forward pass gives only a bounded number of global communication rounds.

- `Sample result`: [Hahn 2020](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00306/43545/Theoretical-Limitations-of-Self-Attention-in) shows that fixed-size self-attention cannot model some periodic finite-state languages or hierarchical languages unless the number of layers or heads grows with input length.
- `Why it matters`: periodicity and hierarchical structure are canonical tests of global coordination. If a fixed architecture cannot maintain exact phase or deep nesting information over arbitrary lengths, then it is behaving like a bounded-parallel device rather than a general sequential machine.
- `Proof idea`: the proof uses an indistinguishability argument on long strings. With only finitely many layers, heads, and precision states, the model collapses many long inputs into the same coarse interaction pattern. One then constructs strings that the language must separate but the fixed architecture cannot reliably distinguish.

The same theme reappears in sharper circuit-style statements.

- `Other results`: [Hao, Angluin, and Frank 2022](https://aclanthology.org/2022.tacl-1.46/) show that unique-hard-attention variants lie in $\mathsf{AC}^0$, so classic constant-depth separator tasks such as parity are out of reach in that setting. The same paper shows that stronger average-hard variants can recognize languages outside $\mathsf{AC}^0$, including MAJORITY and DYCK-1.
- `Other results`: [Barcelo et al. 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5f0fdc1acd47431f7f3bb8ee85598cef-Abstract-Conference.html) strengthen the picture by exhibiting an $\mathsf{AC}^0$ language not recognized by unique-hard-attention encoders.
- `Takeaway`: no-CoT transformers are not generic sequential devices in disguise. Their failure modes line up with the same separator problems that organize bounded-depth circuit complexity.

### Circuit Upper Bounds

The positive side of the theory asks how far a no-CoT transformer can go within bounded parallelism. Here the main strategy is to compile the forward pass into a classical shallow circuit family.

- `Sample result`: [Merrill, Sabharwal, and Smith 2022](https://aclanthology.org/2022.tacl-1.49/) show that saturated transformers are simulable by constant-depth threshold circuits, giving a natural $\mathsf{TC}^0$-style upper bound.
- `Why it matters`: this places a fairly realistic attention variant inside a standard classical class. The message is not just that transformers are limited, but that they are limited in a very specific and familiar way.
- `Proof idea`: the simulation proceeds layer by layer. Each attention head becomes a shallow comparison-and-routing gadget, and each aggregation step becomes shallow threshold-style arithmetic. Because the number of layers is fixed, composing these gadgets preserves constant depth.

This upper-bound viewpoint also clarifies how model variants differ.

- `Other results`: [Hao, Angluin, and Frank 2022](https://aclanthology.org/2022.tacl-1.46/) place unique-hard-attention models in $\mathsf{AC}^0$, while showing that average-hard attention can already express stronger counting behavior such as MAJORITY and DYCK-1.
- `Other results`: [Chiang, Cholak, and Pillay 2023](https://proceedings.mlr.press/v202/chiang23a.html) connect fixed-precision transformer encoders to a counting-logic fragment and therefore to a bounded-depth upper-bound regime.
- `Other results`: [Barcelo et al. 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5f0fdc1acd47431f7f3bb8ee85598cef-Abstract-Conference.html) show that hard-attention encoders with richer numerical predicates can recognize correspondingly richer logical fragments.
- `Takeaway`: attention is global communication, but in the no-CoT setting it is still only a bounded number of communication rounds. That is why these models fit naturally into shallow circuit classes.

### Logic Characterizations

Circuit upper bounds say what class the computation belongs to. Logic characterizations say the same thing in a more structural language: they describe exactly what a model can state about positions in a string, how those local facts are combined, and what forms of counting are available.

- `Sample result`: [A Logic for Expressing Log-Precision Transformers](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a48e5877c7bf86a513950ab23b360498-Abstract-Conference.html) shows that log-precision transformers can be expressed in first-order logic with majority quantifiers.
- `Why it matters`: majority quantifiers are the logical counterpart of threshold gates, so this result explains why log-precision transformers sit naturally near $\mathsf{TC}^0$-style computation.
- `Proof idea`: the transformer is first compiled into a highly uniform threshold circuit. One then uses the standard circuit-to-logic correspondence to translate that circuit into a first-order formula with majority. Each layer becomes a formula describing which positions can influence which others and whether enough of them satisfy some property.

Counting-based logic gives a closely related picture.

- `Other results`: [Transformers Implement First-Order Logic with Majority Quantifiers](https://openreview.net/forum?id=W668diqwp4l) gives a broad first-order-with-majority perspective for transformer networks.
- `Other results`: [Chiang, Cholak, and Pillay 2023](https://proceedings.mlr.press/v202/chiang23a.html) identify a first-order logic with counting quantifiers that upper-bounds fixed-precision transformer encoders and lower-bounds more general encoders.
- `Takeaway`: parity, exact counting, and majority are not arbitrary benchmark tasks. They are canonical probes of what bounded-depth aggregation can or cannot express once precision and attention are fixed.

### Exact Characterizations For Restricted Models

The sharpest theorems are exact characterizations: instead of merely placing a transformer in a large upper-bound class, they identify the precise language class recognized by a restricted transformer family.

- `Sample result`: [Yang, Chiang, and Angluin 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/13d7f172259b11b230cc5da8768abc5f-Abstract-Conference.html) prove that masked hard-attention transformers with strict masking and no position embeddings recognize exactly the star-free languages.
- `Why it matters`: a star-free language is a regular language built using union, concatenation, and complement, but not Kleene star; equivalently, it is definable in $\mathsf{FO}[<]$. So this theorem turns a concrete architectural restriction into an exact classical characterization.
- `Proof idea`: the proof goes through an intermediate programming language. Transformer computations are first rewritten as Boolean RASP-style programs, then translated into temporal logic, and finally matched to the classical equivalence between temporal logic, $\mathsf{FO}[<]$, and star-free languages.

This exact style of result works because the restricted architecture is rigid enough to analyze completely.

- `Other results`: [Barcelo et al. 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5f0fdc1acd47431f7f3bb8ee85598cef-Abstract-Conference.html) show that hard-attention encoders capture specific logical fragments once unary numerical predicates, and for stronger variants counting terms, are added.
- `Other results`: [Thinking Like Transformers](https://proceedings.mlr.press/v139/weiss21a.html) is important here because RASP provides the intermediate language that makes exact compilation arguments possible.
- `Takeaway`: exact characterizations usually require a sufficiently restricted model, but when they apply they are much more informative than a generic upper bound. They tell us precisely which formal-language phenomena the architecture can and cannot realize.

The overall picture is now fairly clean. In the no-CoT regime, a transformer is best treated as a uniform bounded-parallel computation. Under stricter hard-attention assumptions it often sits near $\mathsf{AC}^0$; under richer but still discretizable attention it moves toward $\mathsf{TC}^0$; and in the logical mirror it becomes a fragment of first-order logic with counting or majority. The exact boundary depends on masking, positional information, precision, and the attention rule, but the guiding idea is stable: without externally written intermediate tokens, the model gets only a bounded number of global communication rounds.

## 3. Chain of Thought as a General Computation

### Definitions

- **No-CoT inference:** a decoder-only transformer reads the input context $x$ and produces an answer immediately, after one bounded forward computation.
- **CoT / scratchpad inference:** the model first generates intermediate tokens
  $$
  x \to z_1 \to z_2 \to \cdots \to z_T \to y,
  $$
  and step $t$ is computed from the growing context $(x,z_1,\ldots,z_{t-1})$.
- **Sequential workspace:** the generated tokens $z_1,\ldots,z_T$ act as externalized state. They are not just extra text; they are visible memory that later steps can read.
- **Parameter expressiveness:** what the architecture can realize when the weights are chosen or trained appropriately.
- **Prompt expressiveness:** what a fixed pretrained model can be made to do by changing only the prompt.

The main conceptual shift is simple: without CoT, the transformer is still a bounded parallel computation; with CoT, it gets $T$ extra rounds of sequential computation and $T$ pieces of writable workspace.

### Formal CoT Expressiveness

- **Sample result:** [Merrill and Sabharwal 2023](https://arxiv.org/abs/2310.07923) give a clean complexity-theoretic account of CoT for decoder-only transformers.
- **Statement:** under their projected-pre-norm and generalized-pre-norm assumptions, logarithmic-length CoT only modestly extends a standard decoder-only transformer, linear-length CoT can recognize all regular languages, linear CoT remains within context-sensitive languages, and polynomial-length CoT characterizes polynomial-time computation.
- **Proof idea:** the model uses scratchpad tokens as a work tape. The construction shows how later decoding steps can recover and update state written earlier in the trace, so one generated token can play the role of one step of a more general sequential machine. The converse direction simulates the whole decoding process by a standard machine model to obtain the upper bounds.
- **Why this matters:** the theorem formalizes the idea that CoT is not merely a stylistic change in output format. It changes the resource model from bounded parallel depth toward general sequential computation.

### CoT as Serial Computation

- **Sample result:** [Li, Liu, Zhou, and Ma 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3309b4112c9f04a993f2bbdd0274bba1-Abstract-Conference.html) make the serial-computation view explicit.
- **Statement:** under their construction, constant-depth constant-bit transformers without CoT are bounded by weak parallel classes, while the same models with $T$ CoT steps can solve problems computable by Boolean circuits of size $T$.
- **Proof idea:** each CoT step carries intermediate results forward through the growing context, so the model can unfold a serial computation over multiple decoding rounds instead of compressing everything into one shallow pass.
- **Literature survey:** this result complements the hierarchy view above. Merrill and Sabharwal show how longer CoT traces move the architecture toward polynomial-time computation; Li et al. show the same phenomenon from the angle of circuit size and inherently serial tasks.

### Turing Completeness as the Idealized Limit

- **Sample result:** [Perez, Barcelo, and Marinkovic 2021](https://www.jmlr.org/papers/v22/20-302.html) and [Bhattamishra, Patel, and Goyal 2020](https://aclanthology.org/2020.conll-1.37/) prove Turing-completeness results for transformers under idealized assumptions.
- **Statement:** with sufficiently expressive numerical encodings and positional access, transformer architectures can simulate arbitrary Turing machines.
- **Proof idea:** the machine configuration is encoded into token representations or position-indexed slots, and attention plus local updates simulate one machine transition per layer or decoding step.
- **Why this matters:** these theorems mark the outer boundary of formal expressiveness. They show that transformers are not inherently confined to shallow-circuit behavior, while also relying on stronger assumptions than the finite-precision $\mathsf{P}$-style results.

### Prompt-Space CoT Is Different

- **Sample result:** [Wei et al. 2022](https://arxiv.org/abs/2201.11903) popularized CoT prompting as a way to improve reasoning behavior in large language models.
- **Interpretation:** this is an empirical prompting result, not the same kind of expressiveness theorem as the trainable-weight results above.
- **Why the distinction matters:** a frozen pretrained model may improve when prompted to produce intermediate reasoning because the prompt changes the distribution of computations the model performs, gives it more tokens for intermediate state, and matches patterns seen during training.
- **Scope note:** this does not mean that prompting freely adds a new algorithm to the backbone. Unless the prompt itself is allowed to be a powerful learned prefix in a theoretical construction, prompt-space CoT is better viewed as steering or eliciting computations the frozen model can already implement.

The synthesis is that CoT should be understood as a computation model, not only as a reasoning style. With trainable parameters, scratchpad tokens can enlarge the formal expressive power of the architecture by providing sequential workspace. With a frozen model, CoT prompting is a weaker notion: it changes execution, not the underlying architecture's parameter-level expressiveness.

## 4. Case Studies

The case studies are easiest to organize by **inference regime**.

- In the **non-CoT** setting, the transformer reads the input and answers in one bounded forward pass. The natural probes are tasks that stress bounded parallel computation: formal-language recognition, exact counting, associative recall, long-context retrieval, and length generalization.
- In the **CoT** setting, the model may generate intermediate tokens and then condition on them. The natural probes shift to tasks that require visible sequential workspace, such as repeated composition, arithmetic derivation, and circuit evaluation.

### 4.1 Non-CoT: one-pass transformers

- **Formal languages.**
  Definition: this family asks which languages $L \subseteq \Sigma^\ast$ a transformer can recognize uniformly over all lengths. The standard subfamilies should be separated: regular languages are finite-state languages; star-free languages are the aperiodic regular languages, equivalently those definable in first-order logic over words; Dyck languages are balanced-parentheses languages and are the standard context-free test family when one wants to probe stack-like nesting.
  Anchor result: [Yang, Chiang, and Angluin 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/13d7f172259b11b230cc5da8768abc5f-Abstract-Conference.html) show that masked hard-attention transformers with strict masking and no position embeddings recognize **exactly** the star-free languages.
  Proof idea: they introduce Boolean RASP as an intermediate symbolic language, show an equivalence between that language and the transformer family under study, and then import the classical characterization of star-free languages via temporal or first-order logic.
  Other results in the family: [Strobl et al. 2024](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00663/120983/What-Formal-Languages-Can-Transformers-Express-A) survey the area; [Bhattamishra, Ahuja, and Goyal 2020](https://aclanthology.org/2020.emnlp-main.576/) give constructive results for Dyck-style and counter-like languages; [Weiss, Goldberg, and Yahav 2021](https://proceedings.mlr.press/v139/weiss21a.html) make the symbolic side concrete with RASP programs; [Lindner et al. 2023](https://arxiv.org/abs/2301.05062) push this laboratory setup further with Tracr. The main lesson is that "formal languages" is not one benchmark family but several, and they probe different memory resources.

- **Counting.**
  Definition: counting tasks ask the model to maintain exact global multiplicities rather than merely detect presence. Canonical examples are token histograms, majority, parity, and occurrence counting.
  Anchor result: [Yehudai et al. 2024](https://arxiv.org/abs/2407.15160) study exact occurrence counting and identify a sharp dependence on embedding dimension and vocabulary size.
  Experimental logic: they give a positive construction when the representation dimension is large enough to support effectively independent token directions, and then show that below that regime non-orthogonality creates interference, large required weights, and practical non-learnability. They validate this by varying vocabulary size, context length, and model scale and measuring both in-distribution and out-of-distribution counting accuracy.
  Other results in the family: counting also appears throughout the logic and circuit characterizations from earlier sections because majority and parity are natural separator tasks. In practice, histogram-style tasks in [Weiss, Goldberg, and Yahav 2021](https://proceedings.mlr.press/v139/weiss21a.html) are a useful symbolic benchmark. The broader point is that exact aggregation is its own expressive bottleneck.

- **Associative recall and mechanistic circuits.**
  Definition: associative recall asks whether the model can bind keys to values in context and later return the value associated with a queried key. This is the smallest synthetic version of factual recall, entity binding, and content-addressed retrieval.
  Anchor result: [Nichani, Lee, and Bietti 2024](https://openreview.net/forum?id=PtYojIoW0u) analyze factual recall through associative memory and show that a shallow transformer with one self-attention layer and an MLP can achieve perfect recall when either the attention or MLP parameters scale essentially linearly with the number of stored facts, up to log factors.
  Proof idea: the model is decomposed into two possible storage mechanisms. Facts can be stored in the value pathway or in the MLP, and the theorem tracks how much capacity each route provides.
  Other results in the family: [Olsson et al. 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) identify induction heads as a concrete circuit for copy-and-continue behavior. Induction heads are narrower than general associative recall, but they are a key mechanistic example of how attention implements retrieval in practice. This makes associative recall a bridge case study: it connects formal expressiveness, in-context factual memory, and mechanistic interpretability.

- **Long-context retrieval and NIAH.**
  Definition: long-context retrieval asks whether the model can find and use relevant information in a context long enough that positional encoding, distractor suppression, and attention scaling become part of the task. The minimal probe is **needle in a haystack (NIAH)**: one relevant fact is hidden among many distractors.
  Representative benchmark result: [RULER](https://arxiv.org/abs/2404.06654) shows why vanilla NIAH is only a starting point. It expands the setting to multiple needles, multiple needle types, tracing tasks, and aggregation tasks over long contexts.
  Experimental logic: many models score nearly perfectly on the single-needle probe, but performance drops sharply once the context grows longer or the task requires more than one retrieval step. The benchmark therefore isolates the difference between "can find one fact" and "can robustly use long context."
  Other benchmarks in the family: [LongBench](https://arxiv.org/abs/2308.14508) broadens the evaluation to multi-document QA, summarization, code, synthetic tasks, and bilingual long-context understanding. The main survey point is that NIAH and long-context retrieval should not be conflated: NIAH is the floor, not the full task family.

- **Length generalization.**
  Definition: length generalization asks whether a model trained on shorter sequences still solves the same task on longer ones. This is distinct from long-context retrieval: the issue is not only whether the model can attend far enough, but whether it learned the underlying algorithm instead of a length-specific shortcut.
  Anchor result: [Zhou et al. 2024](https://openreview.net/forum?id=AssIuHnmHX) study this question through RASP-L and argue that length generalization tracks the existence of a short symbolic transformer-native program for the task.
  Experimental logic: they compare tasks by the complexity of their RASP-L descriptions and show that tasks with shorter descriptions are more likely to extrapolate in length. They also show that scratchpads help only when they simplify the induced program; otherwise they can make generalization worse.
  Other results in the family: [Weiss, Goldberg, and Yahav 2021](https://proceedings.mlr.press/v139/weiss21a.html) provide the original RASP language for writing transformer programs, and [Lindner et al. 2023](https://arxiv.org/abs/2301.05062) compile such programs into exact transformers. These papers do not solve length generalization by themselves, but they provide the symbolic vocabulary needed to state the problem precisely.

### 4.2 CoT: transformers with intermediate workspace

- **Why CoT changes the case studies.**
  In the CoT regime, the model may generate intermediate tokens and then reread them. This changes the resource model: the model is no longer limited to one bounded parallel pass, because the generated tokens act as visible external state.
  Anchor result: [Merrill and Sabharwal 2023](https://arxiv.org/abs/2310.07923) make this precise. In their framework, logarithmic CoT gives only a modest extension, linear CoT already reaches all regular languages under projected pre-norm assumptions, and polynomially many steps characterize polynomial-time computation.
  Proof idea: the generated scratchpad is treated as a work tape. The paper then proves upper and lower bounds as a function of scratchpad length, showing how intermediate tokens change the reachable complexity class.

- **Regular-language recognition with generated scratchpad.**
  This is the simplest formal-language example of the CoT advantage. In the no-CoT regime, transformers face sharp limits even on some regular-language recognition tasks. In the CoT regime, the model can explicitly write down and update automaton-like state across generated tokens.
  The point of this case study is conceptual: it is the smallest example where visible sequential workspace, rather than a wider one-pass network, changes the expressive story.

- **Serial separator tasks: permutation composition, iterated squaring, circuit value.**
  Definition: these are the canonical CoT separator tasks because they require repeated dependency propagation. In **permutation composition**, the model must repeatedly update a state by composing permutations. In **iterated squaring**, each step depends on the previous arithmetic state. In **circuit value**, the model must evaluate a Boolean circuit whose internal dependencies cannot be collapsed into a trivial one-shot shortcut.
  Anchor result: [Li, Liu, Zhou, and Ma 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3309b4112c9f04a993f2bbdd0274bba1-Abstract-Conference.html) show that constant-depth, constant-bit transformers with $T$ CoT steps can solve any problem computable by Boolean circuits of size $T$.
  Proof idea: they construct a step-by-step simulation in which generated intermediate tokens encode the evolving computational state. This directly turns visible CoT steps into a serial computation resource.
  Other results in the family: their experiments use modular addition as a control task and then show large CoT gains on permutation composition, iterated squaring, and circuit value. These tasks are canonical precisely because they isolate repeated state update rather than vague "reasoning."

- **Arithmetic derivations and dynamic programming.**
  This family gives a complementary view of the same CoT phenomenon. Instead of formal-language recognition or circuit evaluation, the tasks are arithmetic, equation solving, and dynamic-programming problems whose solution naturally unfolds through intermediate states.
  Anchor result: [Feng et al. 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dfc310e81992d2e4cedc09ac47eff13e-Abstract-Conference.html) prove that bounded-depth transformers cannot directly solve several basic arithmetic and equation tasks without super-polynomial growth in model size, but autoregressive transformers can solve them by generating CoT derivations in a suitable math-language format.
  Proof idea: the paper first gives lower bounds for direct-answer prediction, then gives explicit constructive upper bounds once the model is allowed to emit intermediate derivation tokens. The same logic extends to dynamic programming, not just school arithmetic.
  The main survey point is that arithmetic, dynamic programming, permutation composition, iterated squaring, and circuit value all belong to one broader CoT family: tasks whose hardness comes from serial update, and whose tractability improves once the model can externalize intermediate state.

- **Summary of the split.**
  In the non-CoT setting, the cleanest case studies are the ones that expose the limits of bounded parallel computation: formal languages, counting, associative recall, long-context retrieval, and length extrapolation. In the CoT setting, the cleanest case studies are the ones that expose the value of extra sequential workspace: scratchpad-based language recognition, arithmetic derivations, dynamic programming, permutation composition, iterated squaring, and circuit value. Keeping these two lists separate makes the literature easier to scan and prevents a common confusion: the same transformer architecture can fall into very different expressiveness regimes depending on whether it must answer immediately or may first write down intermediate state.

## 5. Alternative Architectures

Explicit CoT is not the only way to add inference-time computation. The general organizing question is:

- Where does the extra computational state live?
- Is it written into visible tokens, or kept latent?
- Is the model reusing the same block, carrying a hidden state, consulting external memory, or updating fast weights?

The common theme is the same as in CoT: more computation requires more state. What changes is whether that state is visible text, latent activations, memory slots, or temporary weights.

### 5.1 Looped Transformers: Reusing Depth as Time

- `Definition: looped transformer`
  A looped transformer reuses the same block of layers for multiple iterations. If we write the hidden sequence at iteration $t$ as $H^{(t)}$, then a looped block has the form
  $$
  H^{(t+1)} = B_\theta(H^{(t)}),
  $$
  where the parameters $\theta$ are shared across iterations. This makes depth behave like time: each loop is another round of computation on the same latent state.

- `Why it matters`
  Looping adds computation without forcing the model to emit intermediate tokens. It is therefore a natural hidden-state alternative to explicit CoT.

- `Sample result`
  The [Universal Transformer](https://arxiv.org/abs/1807.03819) introduced recurrence across depth, and [Giannou et al. 2023](https://proceedings.mlr.press/v202/giannou23a.html) show that looped transformers can be programmed as general-purpose computers.

- `Proof idea`
  The proof is constructive. Some tokens act like instructions and others act like memory. One application of the shared transformer block simulates one computational step: attention routes information to the right locations, and the local update rules implement the step transition. Repeating the block then simulates an arbitrary multi-step computation.

- `Other results`
  [Merrill and Sabharwal 2025](https://arxiv.org/abs/2505.18948) analyze looping together with padding and show exact characterizations in terms of $\mathsf{TC}^d$ classes. This is useful because it separates two resources cleanly: padding increases workspace, while looping increases computation depth.

### 5.2 Recurrent Hidden State and Recurrence-Like Variants

- `Definition: recurrent computation`
  A recurrent model processes a sequence one step at a time, updating an internal state
  $$
  h_{t+1} = F_\theta(h_t, x_t), \qquad y_t = G_\theta(h_t).
  $$
  Here the scratchpad is not written into tokens at all; it stays inside $h_t$.

- `Definition: hidden-state alternatives`
  This family includes classical RNNs, LSTMs, and GRUs, as well as newer architectures such as linear-attention models, RWKV, and Mamba that behave recurrently at inference time.

- `Sample result`
  [Siegelmann and Sontag 1995](https://doi.org/10.1006/jcss.1995.1013) prove that recurrent neural networks are Turing complete under idealized real-valued precision assumptions.

- `Proof idea`
  The proof encodes the configuration of a Turing machine into continuous hidden-state values. One RNN update then simulates one machine transition by decoding the current control state and tape symbol, updating them, and re-encoding the new configuration back into the hidden state.

- `Finite-precision theory`
  [Weiss, Goldberg, and Yahav 2018](https://aclanthology.org/P18-2117/) show that finite-precision recurrent models fall into a hierarchy: some variants can implement counting, while others cannot. [Merrill et al. 2020](https://aclanthology.org/2020.acl-main.43/) refine this into a broader hierarchy of recurrent architectures for language recognition. These results matter because "recurrent" is not one expressiveness class; gates, precision, and update rules matter.

- `Bridge back to transformers`
  [Katharopoulos et al. 2020](https://proceedings.mlr.press/v119/katharopoulos20a.html) show that **linear attention** can be written recurrently. After kernelizing attention, the model can maintain running prefix statistics such as
  $$
  S_t = S_{t-1} + \phi(k_t) v_t^\top, \qquad z_t = z_{t-1} + \phi(k_t),
  $$
  and compute the current output from $(S_t, z_t, q_t)$. So some transformers are literally recurrent at inference time once the attention mechanism is rewritten in the right algebraic form.

- `Newer variants`
  [RWKV](https://arxiv.org/abs/2305.13048) is an explicit RNN/attention hybrid in which the state tracks decayed key-value summaries. [Mamba](https://arxiv.org/abs/2312.00752) is a selective state-space model whose recurrent update is input-dependent, so the model can decide what to keep and what to forget. [Cirone et al. 2024](https://arxiv.org/abs/2402.19047) argue that selectivity substantially increases what deep state-space models can represent, while [Merrill, Petty, and Sabharwal 2024](https://arxiv.org/abs/2404.08819) show formal limitations for state-tracking tasks.

- `Takeaway`
  Recurrent hidden-state models trade visible workspace for a compressed latent state whose power depends sharply on the update rule.

### 5.3 External Memory and Auxiliary Workspace

- `Definition: external memory`
  External memory is a storage object distinct from the model's ordinary hidden state. If the hidden state is $h_t$, then a memory-augmented model typically has an additional state
  $$
  M_{t+1} = U_\theta(M_t, h_t, x_t),
  $$
  together with a read operation that lets the controller access selected parts of $M_t$.

- `Simplest version`
  The simplest version is not a separate module at all, but **auxiliary workspace tokens**. Padding tokens and pause tokens give the model extra slots in which to stage intermediate variables.

- `Sample result`
  [Merrill and Sabharwal 2025](https://arxiv.org/abs/2505.18948) show that padding can strictly increase transformer expressiveness, and [London and Kanade 2025](https://arxiv.org/abs/2505.21024) show that pause tokens strictly increase the expressivity of constant-depth transformers.

- `Proof idea`
  The extra tokens act as workspace positions. They give the model additional places to store and manipulate intermediate values, even when the model depth is otherwise fixed. The separation results show that this extra workspace can strictly enlarge the class of computations the model can realize.

- `Other results`
  In the broader memory-augmented literature, [Neural Turing Machines](https://arxiv.org/abs/1410.5401) introduced differentiable read/write memory controlled by a neural network, and later systems such as differentiable neural computers and memory networks push the same idea further. Transformer-native versions appear in work such as [Token Turing Machines](https://arxiv.org/abs/2211.09119), where a transformer controller reads and writes a bounded set of memory tokens across time.

- `RAG`
  Retrieval-augmented generation (RAG) should be viewed as external read-only memory. The model retrieves from an external corpus and then generates conditioned on the retrieved documents. Formally, one can think of the model as first selecting a latent document variable $z$ and then generating from $(x,z)$, as in [Lewis et al. 2020](https://arxiv.org/abs/2005.11401). From an expressiveness viewpoint, this enlarges the effective state available at inference time, although the formal power now depends on the retriever and the corpus as well as on the backbone.

### 5.4 Fast Weights and Test-Time Training

- `Definition: fast weights / TTT`
  A final alternative keeps the scratchpad in **weight space**. In test-time training (TTT), the model updates a small inner learner during inference:
  $$
  W_{t+1} = W_t - \eta \nabla \ell(W_t; x_t), \qquad y_t = f_{W_t}(x_t).
  $$
  The temporary state is not just a hidden vector; it is a parameter matrix or small model that changes online.

- `Why it matters`
  This differs from ordinary recurrence. In an RNN, the state is a vector passed through a fixed update rule. In TTT, the state is itself a learned predictor, and inference includes an optimization step.

- `Sample result`
  [Sun et al. 2024](https://arxiv.org/abs/2407.04620) introduce TTT layers and make the core expressiveness claim explicit: a model can use test-time learning itself as the hidden-state mechanism.

- `Proof idea`
  The key argument is by unrolling the inner update. Each token does not merely update activations; it changes the fast weights of a small inner model. That gives the architecture a richer state space than a fixed-width hidden vector, because the current "memory" is a function approximator learned on the fly from the recent context.

- `Other results`
  The older fast-weights view appears in [Ba et al. 2016](https://arxiv.org/abs/1610.06258). More recent theory strengthens the connection to in-context computation: [Gozeten et al. 2025](https://arxiv.org/abs/2503.11842) prove that test-time training can strictly improve transformers as in-context learners in a stylized setting.

### 5.5 What to Take Away

The architecture families above differ in where they place the scratchpad:

- `Looped transformers`: reuse transformer blocks and keep intermediate computation in hidden state.
- `Recurrent and recurrence-like models`: carry a latent state across sequence steps.
- `External-memory models and RAG`: add storage outside the ordinary hidden activations.
- `Fast-weight / TTT models`: keep part of the state in parameters that are updated during inference.

So the right comparison is not just "CoT versus no CoT." A better question is:

- Does the model get extra computation by looping?
- By carrying hidden state?
- By reading or writing memory?
- By retrieving external information?
- By updating fast weights?

That is the common thread across these alternatives: they all add computation by adding state, but they store that state in different places.

## 6. In-Context Learning and Prompt-Space Expressiveness

This section shifts the expressiveness question from **changing the weights** to **changing the context**. The model is usually a fixed transformer backbone, and the question is what computations can be carried out by supplying examples or learned prompt vectors at test time.

### Core concepts

- **In-context learning (ICL).** The prompt contains examples
  $$
  D=((x_1,y_1),\ldots,(x_k,y_k),x_{\mathrm{query}}),
  $$
  and the model should output a good prediction for $x_{\mathrm{query}}$ without updating its weights.

- **Implements a learning algorithm in-context.** A transformer implements a learning algorithm $\mathcal A$ in-context if its forward pass on $D$ behaves like "run $\mathcal A$ on the support examples, then apply the learned predictor to $x_{\mathrm{query}}$." In this view, the hidden states act as temporary parameters, optimizer state, or sufficient statistics.

- **Frozen backbone.** The transformer's internal weights are fixed. Only the prompt or prefix is allowed to vary.

- **Prompt tuning / soft prompts.** One learns continuous prompt embeddings prepended to the input while keeping the backbone frozen; see [Lester, Al-Rfou, and Constant 2021](https://aclanthology.org/2021.emnlp-main.243/).

- **Prefix tuning.** One learns continuous vectors injected as virtual prefix states, typically as extra key/value vectors and often at multiple layers, again with a frozen backbone; see [Li and Liang 2021](https://aclanthology.org/2021.acl-long.353/).

- **Prompt-space expressiveness.** For a frozen backbone $B$ and an allowed prompt family $\Pi$, the induced function class is
  $$
  \mathcal F_{\mathrm{prompt}}(B,\Pi)=\{x\mapsto B(p,x): p\in\Pi\}.
  $$
  This is the "expressive power of prompting" as opposed to the expressive power of changing weights.

- **Different from trainable-weight expressiveness.** In the usual expressiveness problem, we vary $\theta$ over all model parameters. Here we vary only $p$, so $\mathcal F_{\mathrm{prompt}}(B,\Pi)$ is a constrained subfamily controlled by a fixed backbone.

- **Different from chain of thought (CoT).** CoT changes the test-time resource model by letting the model generate intermediate tokens and then reason over them. Prompt tuning and prefix tuning change the conditioning interface to a frozen model, but do not by themselves create the same sequential workspace.

The theory in this area is easiest to read as three families of results: ICL as implicit learning, prompt universality, and prompt limitations.

### 6.1 In-Context Learning as Implicit Learning

- **Representative result.** [von Oswald et al. 2023](https://proceedings.mlr.press/v202/von-oswald23a.html) show that linear self-attention can implement an update equivalent to gradient descent for linear regression. This is a clean formal example of a transformer carrying out a learning rule inside its forward pass.

- **Proof idea.** The construction stores quantities such as the current predictor and aggregated statistics from the support examples inside the residual stream. Self-attention aggregates information across examples, and the following linear maps implement the update rule for linear regression. The point is not just that the transformer can fit the task, but that it can realize the **algorithmic update** itself.

- **Setup and nearby results.** [Garg, Tsipras, Liang, and Valiant 2022](https://openreview.net/forum?id=flNZJ2eOet) formulate ICL on controlled function classes such as linear functions and decision trees, showing that trained transformers can learn such tasks from examples in the prompt. [Akyurek et al. 2023](https://openreview.net/forum?id=0g0X4H8yN4I) argue that transformers can implicitly implement estimators such as gradient descent and ridge regression by storing a context-dependent model in their hidden representations.

- **Further theory.** [Mahankali, Hashimoto, and Ma 2024](https://openreview.net/forum?id=8p3fu56lKc) prove that one-layer linear self-attention trained on noisy linear regression implements one-step gradient descent or preconditioned gradient descent depending on the covariance structure. [Fu et al. 2023](https://openreview.net/forum?id=dE5MEi9906) argue that trained transformers can implement higher-order methods close to iterative Newton updates, and [Cheng, Chen, and Sra 2024](https://openreview.net/forum?id=ah1BlQcLv4) extend the picture to functional gradient descent for certain nonlinear in-context learning problems.

- **Takeaway.** In the positive theory papers, ICL is not merely nearest-neighbor lookup or superficial pattern matching. A transformer can be expressive enough to use the prompt as training data and execute a genuine learning rule at inference time.

### 6.2 Prompt Universality

- **Representative result.** [Petrov, Torr, and Bibi 2024](https://proceedings.mlr.press/v235/petrov24a.html) prove positive universality results for prompting frozen transformers. In particular, they show that prefix tuning a single attention head is sufficient to approximate arbitrary continuous functions, and that with depth growing linearly in the sequence length, a frozen transformer can approximate arbitrary sequence-to-sequence functions through its prefix alone.

- **Proof idea.** The proof is constructive. The learned prefix is chosen so that the frozen attention mechanism acts as a programmable routing and interpolation device. One first builds approximation results for simple continuous functions, then composes these modules to obtain sequence-level universality. The paper also gives quantitative approximation bounds linking the required prefix length to the target error.

- **Other results.** [Wang, Chauhan, Wang, and Hsieh 2023](https://openreview.net/forum?id=zWxKYyW9ik) prove universality results for prompt tuning in stylized transformer settings, showing that a sufficiently expressive frozen backbone plus learned prompts can approximate broad function classes. On the empirical side, [Lester, Al-Rfou, and Constant 2021](https://aclanthology.org/2021.emnlp-main.243/) show that soft prompt tuning becomes more competitive as model scale grows, while [Li and Liang 2021](https://aclanthology.org/2021.acl-long.353/) introduce prefix tuning and show that it can match full fine-tuning on some generation tasks with far fewer trainable parameters.

- **Takeaway.** Prompt-space expressiveness can be surprisingly large. A frozen transformer may still be a universal approximator with respect to its prompt interface, but only because the backbone already contains the right computational primitives for the prompt to activate and compose.

### 6.3 Prompt Limitations

- **Representative result.** [Petrov, Torr, and Bibi 2023](https://openreview.net/forum?id=GYOXIRXI7W) show that context-based fine-tuning methods such as prompting and prefix tuning are strictly less expressive than full fine-tuning under their assumptions. In particular, they show that such methods cannot arbitrarily change the relative attention pattern among the content tokens.

- **Proof idea.** The key invariants come from freezing the backbone. The content-token query and key maps are fixed, so the geometry of content-content attention is also largely fixed. Prompt or prefix vectors can add bias terms and redirect some mass through the prompt tokens, but they cannot freely rewrite all relative attention relationships among the original content tokens. Tasks that require a genuinely new attention pattern are therefore out of reach.

- **Other results.** [Wang et al. 2023](https://openreview.net/forum?id=zWxKYyW9ik) also prove lower-bound and impossibility results in restricted settings. For example, for some finite-depth frozen models, prompt tuning cannot memorize arbitrary datasets regardless of prompt length, and they derive lower bounds on the amount of tunable prompt capacity needed for certain tasks.

- **Takeaway.** Prompting is powerful when it can **select, bias, or combine** computations already available in the frozen backbone. It is limited when the task requires changing the backbone's internal computation graph itself, especially the relative attention structure over the content tokens.

### 6.4 Synthesis

The synthesis is straightforward.

- In-context learning asks whether a transformer can **learn from the prompt** without updating weights.
- Prompt universality asks how large the function class is when only the prompt is trainable.
- Prompt limitation results identify the invariants that survive all prompt choices.

Keeping these notions separate is essential. They are related, but they are not the same as full weight training, and they are not the same as CoT-based expressiveness either.
