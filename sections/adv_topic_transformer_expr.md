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

## 1. Notions of Expressiveness for Transformers

For a fixed sequence length $n$, a transformer can be viewed as a function
$$
f_{\theta,n} : \mathcal{X}^n \to \mathcal{Y}^n
$$
for sequence-to-sequence tasks, or as a function $f_{\theta,n}:\mathcal{X}^n\to\mathcal{Y}$ for classification or next-token prediction. Expressiveness can then be defined in the usual approximation-theoretic way:
$$
\inf_{\theta\in\Theta} d(f_{\theta,n}, f_n),
$$
where $d$ may be a uniform norm over embedded sequences, an $L^p$ norm, sequence-level error, language-recognition error, or empirical error on a finite benchmark.

For transformers, however, the subscript $n$ matters. Some theorems fix $n$ and allow the parameters or width to depend on $n$. Others ask for one architecture family that works uniformly over all lengths. Some analyze real-valued embeddings on compact domains, while others analyze discrete strings over an alphabet $\Sigma$ and ask which formal language $L\subseteq\Sigma^\ast$ can be recognized. These are different problems. A universal approximation theorem on $[0,1]^{n\times d}$ does not imply that the same finite-precision transformer recognizes parity for every input length, and a circuit upper bound for fixed-depth finite-precision recognition does not contradict a Turing-completeness theorem with idealized precision or unbounded decoding.

A useful resource description for a transformer expressiveness result is
$$
\mathcal{R} = (L,H,d,n,b,T,P),
$$
where $L$ is depth, $H$ is number of heads, $d$ is embedding width, $n$ is input context length, $b$ is numerical precision, $T$ is the number of autoregressive or looped inference steps, and $P$ is the number of extra prompt, padding, or scratchpad tokens. Most theorems become much clearer when stated as claims about how the function class changes as one of these resources grows.

There are four notions that should be kept separate. Approximation-theoretic expressiveness asks whether a transformer can approximate continuous functions on fixed-length embedded inputs. Formal-language expressiveness asks which string languages can be recognized uniformly across lengths. Prompt-space expressiveness asks what can be induced when the backbone is frozen but prompt tokens vary. Inference-time expressiveness asks what happens when the model can generate intermediate tokens, loop, or use additional positions as workspace.

This section uses "no-CoT" to mean that the model gives its answer after one bounded-depth forward pass over the input. It uses "CoT" or "scratchpad" to mean that the model is allowed to produce intermediate tokens and condition later predictions on them. That distinction is central: no-CoT transformers are often parallel circuits; CoT transformers have a growing sequential computation trace.

## 2. Attention as a Computational Primitive

A single attention head maps token representations $X\in\mathbb{R}^{n\times d}$ to queries, keys, and values,
$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
$$
and then computes
$$
\operatorname{Attn}(X)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V,
$$
where $M$ is a mask encoding causal or bidirectional access. Multi-head attention repeats this operation with several projections and combines the outputs. A transformer block then applies tokenwise feed-forward networks, residual connections, and normalization.

This formula explains both the power and the limitations of the architecture. Attention is a content-addressed read: a token can retrieve information from other positions based on query-key similarity. This makes lookup, copying, and associative recall natural. At the same time, one bounded-depth attention stack performs a bounded number of global aggregation and routing operations. Without recurrence, extra decoding steps, or growing depth, it does not process the input as a sequential machine with one state update per symbol.

Positional information is therefore not a minor detail. Without positional encodings, a bidirectional transformer layer is naturally permutation equivariant: permuting the input positions permutes the hidden states in the same way. With absolute, relative, rotary, or learned positional encodings, the model can distinguish order and distance. Formal-language and long-context results often change when position embeddings are added, removed, extrapolated beyond training length, or represented with finite precision.

The feed-forward sublayers also matter. Attention moves and averages information across positions; the MLP transforms each position's representation. In many constructive expressiveness proofs, attention is used to route or aggregate, while the MLP implements local nonlinear computations. In mechanistic terms, this is why key-value lookup, induction-head circuits, and MLP associative memories are complementary rather than redundant.

## 3. Universal Approximation and Weight-Space Expressiveness

The strongest positive approximation result is that transformers are universal approximators of fixed-length sequence functions under suitable assumptions. [Yun, Bhojanapalli, Rawat, Reddi, and Kumar 2020](https://arxiv.org/abs/1912.10077) show that transformers without positional encodings can approximate continuous permutation-equivariant sequence-to-sequence functions with compact support, and that adding positional encodings lets transformers approximate arbitrary continuous sequence-to-sequence functions on compact domains. This is the transformer analogue of a universal approximation theorem for MLPs: it says the architecture is rich enough when the length is fixed and the weights are allowed to vary.

The caveat is the same as for MLPs, but stronger. Universality is qualitative. It does not say that the required depth, width, number of heads, or parameter magnitudes are reasonable; it does not say that gradient descent finds the construction; and it does not say that the same parameters work uniformly for all lengths. A fixed-length theorem is still useful, but it should not be read as a theorem about length extrapolation or exact algorithmic reasoning.

At the discrete-computation end, [Perez, Barcelo, and Marinkovic 2021](https://www.jmlr.org/papers/v22/20-302.html) prove that attention-based transformer models can be Turing complete under idealized assumptions, including hard attention and unbounded or exact internal representations. [Bhattamishra, Patel, and Goyal 2020](https://aclanthology.org/2020.conll-1.37/) give related Turing-completeness constructions and study which architectural components are needed. These results are important because they show that attention is not intrinsically incapable of general computation.

The Turing-completeness results also illustrate why resource assumptions cannot be ignored. Infinite precision, exact comparisons, or unbounded internal computation can hide memory in real-valued states. Real deployed transformers use finite precision, finite context windows, and finite depth. Therefore a Turing-completeness theorem and a finite-precision lower bound are not contradictory; they are statements about different computational models.

This section's takeaway is that trainable transformer weights are expressive enough to represent very broad function classes in idealized settings. The sharper questions are quantitative and resource-sensitive: how many layers and heads are needed, how much precision is assumed, whether parameters grow with input length, and whether the computation is one pass or many passes.

## 4. No-CoT Transformers as Bounded Parallel Computation

The "no-CoT as a bounded circuit" viewpoint studies a transformer that reads the input and immediately produces an answer after a bounded number of layers. In that setting, the computation is highly parallel: every layer updates all positions at once, and a constant-depth model performs only a constant number of global communication rounds. This is why circuit classes such as $\mathsf{AC}^0$ and $\mathsf{TC}^0$ appear throughout the literature.

[Hahn 2020](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00306/43545/Theoretical-Limitations-of-Self-Attention-in) is a canonical early limitation result. Hahn showed that, under soft and hard attention settings, fixed-size self-attention models cannot robustly model some periodic finite-state languages or hierarchical languages unless the number of layers or heads grows with input length. The result is not that practical LLMs cannot parse anything; rather, it isolates what is missing from a bounded self-attention computation when it is asked to implement uniform sequential or stack-like behavior across unbounded lengths.

Circuit-complexity analyses sharpen this picture. [Hao, Angluin, and Frank 2022](https://aclanthology.org/2022.tacl-1.46/) study hard-attention transformer encoders as language recognizers and show that unique-hard-attention variants lie within $\mathsf{AC}^0$, while average-hard-attention variants can recognize some languages outside $\mathsf{AC}^0$, including $\mathsf{MAJORITY}$ and $\mathsf{DYCK}$-1 in their setting. [Merrill, Sabharwal, and Smith 2022](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00493/112604/Saturated-Transformers-are-Constant-Depth) analyze saturated transformers and relate them to constant-depth threshold circuits, giving a $\mathsf{TC}^0$ style upper-bound picture.

Logic characterizations make the same idea more structural. [Chiang, Cholak, and Pillay 2023](https://proceedings.mlr.press/v202/chiang23a.html) identify a first-order logic with counting quantifiers that gives both upper and lower bounds for transformer encoders under fixed-precision assumptions. [Merrill and Sabharwal 2023](https://openreview.net/forum?id=uR8TtWCIsr) show that log-precision transformers can be expressed in first-order logic with majority quantifiers. These results explain why small changes in precision can matter: a single head's ability to attend broadly and average information depends on how finely attention weights can be represented.

The bounded-circuit lens is useful because it predicts separator tasks. Parity is not in uniform $\mathsf{AC}^0$; graph connectivity is not expected to be in small constant-depth threshold classes; exact iteration and circuit-value problems are inherently sequential in ways a bounded parallel pass cannot easily reproduce. These tasks are therefore not arbitrary puzzles. They are probes of whether a transformer has access to sequential computation, high-precision counting, or extra workspace.

## 5. Formal Languages, Logic, and Counting

Formal-language expressiveness asks which languages $L\subseteq\Sigma^\ast$ a transformer recognizer can accept uniformly over all input lengths. This is different from fitting a finite dataset of strings up to some maximum length. The mature survey reference is [Strobl, Merrill, Weiss, Chiang, and Angluin 2024](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00663/120983/What-Formal-Languages-Can-Transformers-Express-A), which emphasizes that the field contains many compatible results under different assumptions about attention, masking, precision, positional encodings, and depth.

A clean exact frontier comes from masked hard-attention transformers. [Yang, Chiang, and Angluin 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/13d7f172259b11b230cc5da8768abc5f-Abstract-Conference.html) prove that masked hard-attention transformers with strict masking and no position embeddings recognize exactly the star-free languages, equivalently the languages definable in linear temporal logic. Adding positional information, changing masking, or changing depth changes the language class. This is a good example of why "the expressiveness of transformers" is not a single object.

RASP gives another way to reason about formal sequence computations. [Weiss, Goldberg, and Yahav 2021](https://proceedings.mlr.press/v139/weiss21a.html) introduce the Restricted Access Sequence Processing Language as a programming language for transformer-like computation. RASP programs can express tasks such as histograms, sorting, and Dyck-language variants, and the program structure gives a rough account of how many layers and heads are needed. [Lindner et al. 2023](https://arxiv.org/abs/2301.05062) extend this idea with Tracr, which compiles RASP-like programs into transformer weights and provides ground-truth models for interpretability experiments.

Counting deserves its own treatment. A simple counting task asks for
$$
c_a(x_1,\ldots,x_n)=\sum_{i=1}^n \mathbf{1}\{x_i=a\}.
$$
This looks easy, but exact counting over large vocabularies and long lengths requires stable aggregation and representation of integer-valued information. [Yehudai, Kaplan, Ghandeharioun, Geva, and Globerson 2024](https://arxiv.org/abs/2407.15160) study when transformers can count occurrences and identify a bottleneck governed by embedding dimension, context length, and vocabulary size. In the favorable regime, a transformer can maintain count-like information; in crowded embedding regimes, interference and numerical instability make exact counting much harder.

The formal-language lesson is not that transformers cannot handle language. It is that bounded-depth attention has specific algebraic and circuit-theoretic limits. Star-free languages, regular languages, parity, majority, Dyck languages, and context-free languages each test different capabilities: order sensitivity, finite-state tracking, modular counting, threshold aggregation, stack-like nesting, and unbounded hierarchical memory. A good survey keeps those examples separate because they witness different resource bottlenecks.

## 6. Long Context, NIAH, and Retrieval Expressiveness

Long context is often marketed as if a larger context window automatically gives stronger reasoning. Expressiveness theory suggests a more careful interpretation: a long context increases the memory available to the model, but the model must still retrieve, aggregate, compare, and reason over that memory. The context window is a storage resource, not by itself an algorithm.

Needle-in-a-haystack tests are the simplest long-context probe. A fact or key-value pair is planted somewhere in a long distractor document, and the model is asked to recover it. In functional terms, the benchmark approximates
$$
F(h_1,\ldots,h_n,q)=v_j
\quad\text{where}\quad
j=\arg\max_i \operatorname{sim}(q,k_i),
$$
with many irrelevant $h_i$ acting as distractors. Success on this test demonstrates a form of retrieval expressiveness: the model can route a query to a relevant location in a long sequence.

NIAH is useful, but it is weak as a complete long-context test. It usually does not require combining many distant facts, resolving contradictions, counting, building a latent state, or following a multi-step dependency chain. [Liu et al. 2024](https://aclanthology.org/2024.tacl-1.9/) show the "lost in the middle" effect: language models often perform best when relevant information appears near the beginning or end of the context and worse when it appears in the middle. This indicates that nominal context length and effective context use are different quantities.

Broader benchmarks try to close this gap. [LongBench](https://arxiv.org/abs/2308.14508) evaluates long-context understanding across single-document QA, multi-document QA, summarization, few-shot learning, synthetic tasks, and code completion. [RULER](https://arxiv.org/abs/2404.06654) was motivated by the observation that simple NIAH retrieval is too shallow, and adds tasks that stress multi-hop tracing, aggregation, and variable context length. These benchmarks are empirical, not theorems, but they align with the theory: retrieval, aggregation, and serial reasoning are distinct capabilities.

Associative recall is the cleaner theoretical cousin of NIAH. The input contains key-value pairs
$$
(k_1,v_1),\ldots,(k_m,v_m),q,
$$
and the desired output is $v_j$ for the key $k_j$ matching query $q$. Attention is well suited to this task because query-key similarity can select a position and copy its value. But robust associative recall over long contexts still depends on position encodings, key collisions, distractor distribution, numerical precision, and whether the model has enough heads or layers to disambiguate the query.

Mechanistic work gives concrete circuits for this behavior. [Olsson et al. 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) identify induction heads that implement a pattern like $[A][B]\ldots[A]\to[B]$, supporting copy-and-continue behavior and many in-context learning phenomena. [Nichani, Lee, and Bietti 2024](https://openreview.net/forum?id=PtYojIoW0u) study factual recall through associative memories and show how shallow transformers can trade off attention matrices and MLPs as storage mechanisms. These results connect long-context benchmarks to actual internal algorithms: the model needs not just a large window, but circuits that use it.

## 7. In-Context Learning and Prompt-Space Expressiveness

In-context learning asks whether a fixed transformer can use examples inside the context to infer a task at test time. A typical prompt has the form
$$
(x_1,y_1),\ldots,(x_k,y_k),x_{\mathrm{query}},
$$
and the model should output something close to $f(x_{\mathrm{query}})$ for the latent function $f$ that generated the examples. This is not ordinary supervised training, because $\theta$ is not updated during the prompt; the learning algorithm, if any, must be implemented inside the forward computation.

[Garg, Tsipras, Liang, and Valiant 2022](https://openreview.net/forum?id=flNZJ2eOet) study this question by training transformers to in-context learn simple function classes such as linear functions, sparse linear functions, two-layer neural networks, and decision trees. Their results show that transformers can be trained to implement useful in-context prediction rules. [von Oswald et al. 2023](https://proceedings.mlr.press/v202/von-oswald23a.html) give a more mechanistic construction: a linear self-attention layer can implement an update resembling gradient descent on a regression loss, supporting the view of transformers as implicit optimizers in some regimes.

Prompt tuning and prefix tuning ask a nearby but different question. Instead of using ordinary examples in the context, we prepend trainable vectors or tokens while freezing the backbone. [Lester, Al-Rfou, and Constant 2021](https://aclanthology.org/2021.emnlp-main.243/) show empirically that soft prompt tuning becomes more competitive as model scale increases. Theoretical work then asks what functions can be induced by changing only those prompt parameters.

The prompt-space theory has both positive and negative sides. [Wang, Chauhan, Wang, and Hsieh 2023](https://arxiv.org/abs/2305.18787) prove universality results for prompt tuning in stylized transformer settings and also prove limitations for finite-depth fixed-weight models. [Petrov, Torr, and Bibi 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/18a367688479c1b08b001584218a4443-Abstract-Conference.html) show that context-based fine-tuning methods can be less expressive than full fine-tuning under their assumptions because they cannot freely change relative attention patterns over content tokens. In a different positive direction, [Petrov, Torr, and Bibi 2024](https://proceedings.mlr.press/v235/petrov24a.html) show that prompting or prefixing certain pretrained transformer constructions can yield universal approximation.

The synthesis is that prompt expressiveness is conditional. If the frozen backbone already contains the right computational primitives, prompt tokens can select, compose, or steer them. If the task requires an attention pattern or algorithm not present in the backbone and not realizable through the allowed prompt interface, prompting is limited. This is why "CoT prompting works" should not be confused with "any algorithm can be added by a clever sentence."

## 8. Chain of Thought and Scratchpad Expressiveness

Chain of thought changes the computation graph. Without CoT, a decoder-only transformer maps the input context $x$ to an answer token after one forward pass. With CoT, it generates a sequence
$$
z_1,\ldots,z_T
$$
and each step computes from the growing context $(x,z_1,\ldots,z_{t-1})$. Thus $T$ intermediate tokens are not just text; they are $T$ rounds of additional sequential computation and $T$ additional pieces of externalized state.

Empirically, [Wei et al. 2022](https://arxiv.org/abs/2201.11903) popularized chain-of-thought prompting as a way to improve arithmetic, commonsense, and symbolic reasoning in large language models. From an expressiveness perspective, the more important question is whether intermediate generation strictly increases the class of computations a transformer can realize. The answer is yes under several formal models.

[Merrill and Sabharwal 2023](https://arxiv.org/abs/2310.07923) give a clean complexity-theoretic account. They show that the amount of intermediate generation matters: logarithmic-length CoT only modestly extends a standard decoder-only transformer, linear-length CoT can recognize all regular languages under their projected-pre-norm assumptions, linear CoT remains within context-sensitive languages, and polynomial-length CoT with generalized pre-norm characterizes polynomial-time computation in their setting. The theorem should be read as a resource hierarchy: more scratchpad length gives more sequential computation.

[Li, Liu, Zhou, and Ma 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3309b4112c9f04a993f2bbdd0274bba1-Abstract-Conference.html) make the serial-computation intuition especially explicit. They show that constant-depth constant-bit transformers without CoT are bounded by weak parallel classes, while with $T$ CoT steps they can solve problems computable by Boolean circuits of size $T$ under their construction. Their separator tasks include permutation composition, iterated squaring, and circuit value problems, all of which require repeated dependency propagation that a one-pass low-depth model struggles to emulate.

This gives a precise meaning to "CoT parameter expressiveness." If the transformer weights can be chosen or trained to use scratchpad tokens as a work tape, then the architecture plus decoding procedure is strictly stronger than the same architecture forced to answer immediately. The extra power comes from the autoregressive loop, not from the words "think step by step" as natural language.

"CoT prompt expressiveness" is a different question. A frozen pretrained model may improve when prompted to write intermediate reasoning because the prompt changes the distribution of computations the model performs, gives it more tokens for intermediate state, and matches patterns seen in training. But the formal prompt-space limitations from Section 7 still apply. A prompt can expose or organize a latent algorithm; it does not guarantee that a missing algorithm appears.

## 9. Other Forms of Test-Time Computation

Explicit CoT is only one way to add inference-time computation. A looped transformer applies the same or related transformer block repeatedly to a representation, adding depth without adding many new parameters. The [Universal Transformer](https://arxiv.org/abs/1807.03819) introduced this recurrence-like idea in a neural sequence model, and [Giannou et al. 2023](https://proceedings.mlr.press/v202/giannou23a.html) show that looped transformers can be programmed as general-purpose computers by using the input as instructions and memory.

Looping differs from CoT in where the intermediate state lives. CoT writes state into visible tokens, which can be inspected, supervised, or consumed by later decoding steps. Looping keeps more of the computation latent inside hidden states, which can be cheaper in tokens and may better match iterative algorithms, but it is less directly auditable. Both mechanisms increase effective computation depth.

Padding and pause tokens provide another resource: extra positions. In a one-pass model, adding positions can give the network more places to store intermediate features or aggregate information in parallel. [Merrill and Sabharwal 2025](https://arxiv.org/abs/2505.18948) analyze padding tokens as parallelizable test-time compute and characterize how padding plus looping expands the reachable circuit classes in their hard-attention setting. [London and Kanade 2025](https://arxiv.org/abs/2505.21024) prove that pause tokens can strictly increase the expressiveness of constant-depth transformers under bounded- and log-precision assumptions.

These mechanisms should be compared as different resource budgets. CoT increases sequential decoding time and produces an external trace. Looping increases recurrent depth over a fixed or slowly changing representation. Padding and pause tokens increase workspace positions and can sometimes remain parallel. Long context increases storage for external information. None of these resources is identical to the others, and each one helps different task families.

## 10. Learnability, Length Generalization, and Mechanistic Realization

Representability does not imply learnability. A transformer may have weights that implement a correct algorithm, while gradient descent learns a shallow heuristic that works only on the training length or distribution. This is especially visible on arithmetic, parity, Dyck languages, sorting, and long-context retrieval tasks. The model may fit all training examples while failing to extrapolate because the learned computation is not the length-uniform algorithm.

RASP-style analyses make length generalization more concrete. If a task has a short RASP program that works for all lengths, it is plausible that a transformer can represent a uniform solution with a small number of layers and heads. [Zhou et al. 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/45ed1a72597594c097152ef9cc187762-Abstract-Conference.html) use this idea to study which algorithms transformers learn under length extrapolation, arguing that short RASP-like solutions correlate with better length generalization and that scratchpad formats can help or hurt depending on whether they simplify the underlying program.

Mechanistic interpretability gives a complementary view. Instead of asking only whether a function is representable, it asks which circuit inside the network implements it. Induction heads, associative memories, copy circuits, and RASP-compiled models provide examples where a behavior can be tied to a concrete internal mechanism. This matters because two models may have the same training accuracy but different circuits, and only one may extrapolate.

Comparisons with neighboring architectures help isolate which limitations are transformer-specific. Finite-precision RNNs and LSTMs have their own formal hierarchies; [Weiss, Goldberg, and Yahav 2018](https://aclanthology.org/P18-2117/) show that practical finite-precision recurrent models differ in their ability to implement counting behavior, and [Merrill et al. 2020](https://aclanthology.org/2020.acl-main.43/) develop a hierarchy of RNN architectures. Linear-attention transformers can also be implemented recurrently, as in [Katharopoulos et al. 2020](https://arxiv.org/abs/2006.16236), showing that "transformer versus RNN" is not a single binary distinction. The relevant comparison is which architecture gives the right memory, routing, and compute pattern for the target task.

## 11. Designing Expressiveness Experiments

A good transformer expressiveness experiment should identify the resource being tested. If the goal is retrieval, use NIAH-like or key-value tasks and vary context length, needle position, distractor similarity, and number of needles. If the goal is aggregation, use counting, majority, frequency, or histogram tasks and vary vocabulary size and embedding dimension. If the goal is formal language recognition, use parity, regular languages, star-free languages, Dyck variants, and context-free subclasses with training lengths separated from test lengths.

If the goal is serial reasoning, use separator tasks such as permutation composition, iterated squaring, graph connectivity, circuit value, multi-step arithmetic, or automaton simulation. These tasks should be tested in no-CoT, CoT, looped, padded, and prompt-only regimes under matched parameter budgets. Otherwise it is unclear whether the gain comes from the transformer weights, the scratchpad length, the extra positions, the number of forward passes, or the training distribution.

For prompt-space expressiveness, separate textual prompting, soft prompt tuning, prefix tuning, in-context examples, and full fine-tuning. A frozen-backbone prompt result should not be compared directly to a trainable-weight CoT theorem. A clean experiment records which parameters are trainable, how many prompt tokens are allowed, whether the prompt length grows with input length, and whether the backbone saw similar tasks in pretraining.

For long-context experiments, report more than maximum context length. Measure accuracy as a function of relevant-information position, context length, number of distractors, number of relevant facts, and whether the answer requires retrieval, comparison, aggregation, or multi-hop reasoning. A model that passes a one-needle retrieval test may still fail counting, contradiction resolution, or state tracking over the same window.

## 12. Takeaway and Open Problems

The main synthesis is that a transformer's expressiveness is determined by the source of computation it is allowed to use. With trainable weights and fixed length, transformers have strong universal approximation results. With arbitrary precision or idealized attention, they can be Turing complete. With fixed depth, finite precision, and no intermediate generation, they often behave like bounded parallel circuits. With CoT, looping, padding, pause tokens, or long contexts, the resource model changes and the reachable computations can expand sharply.

This is why one should not summarize the field as "transformers are weak" or "transformers are universal." Both statements are true in different regimes. The precise question is: fixed or growing length, finite or arbitrary precision, trainable or frozen weights, one pass or many passes, visible or latent scratchpad, and how much workspace?

The most important open problem is to characterize a model close to a practical decoder-only LLM: softmax attention, realistic positional encodings such as RoPE, layer normalization, residual streams, finite precision, finite context windows, and autoregressive decoding. A second open problem is the representability-learnability gap: when does gradient descent actually find the length-uniform algorithm rather than a shortcut? A third is prompt capacity: how much genuinely new computation can be induced through textual prompts, soft prompts, or prefixes for a fixed pretrained model? A fourth is the comparison among CoT, loops, padding, pause tokens, and external tools as different kinds of test-time compute.

The practical lesson for model evaluation is to choose separator tasks that match the claimed capability. NIAH tests retrieval; associative recall tests key-value binding; formal languages test state tracking and hierarchy; counting tests exact aggregation; circuit value and permutation composition test serial computation; in-context learning tests whether the context can act as data for a latent learning algorithm. Together, these give a much sharper picture of transformer expressiveness than a single benchmark score or context-window number.
