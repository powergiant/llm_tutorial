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

For a fixed sequence length $n$, a transformer can be viewed as a function
$$
f_{\theta,n} : \mathcal{X}^n \to \mathcal{Y}^n
$$
for sequence-to-sequence tasks, or as a function $f_{\theta,n}:\mathcal{X}^n\to\mathcal{Y}$ for classification or next-token prediction. Expressiveness can then be defined in the usual approximation-theoretic way:
$$
\inf_{\theta\in\Theta} d(f_{\theta,n}, f_n),
$$
where $d$ may be a uniform norm over embedded sequences, an $L^p$ norm, sequence-level error, language-recognition error, or empirical error on a finite benchmark. For transformers, however, the subscript $n$ matters. Some theorems fix $n$ and allow the parameters or width to depend on $n$, while others ask for one architecture family that works uniformly over all lengths. Some analyze real-valued embeddings on compact domains, while others analyze discrete strings over an alphabet $\Sigma$ and ask which formal language $L\subseteq\Sigma^\ast$ can be recognized. These are different problems, so a useful resource description is
$$
\mathcal{R} = (L,H,d,n,b,T,P),
$$
where $L$ is depth, $H$ is number of heads, $d$ is embedding width, $n$ is input context length, $b$ is numerical precision, $T$ is the number of autoregressive or looped inference steps, and $P$ is the number of extra prompt, padding, or scratchpad tokens.

A single attention head maps token representations $X\in\mathbb{R}^{n\times d}$ to queries, keys, and values,
$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
$$
and computes
$$
\operatorname{Attn}(X)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V,
$$
where $M$ is a mask encoding causal or bidirectional access. This formula explains both the power and the limitation of the architecture. Attention is a content-addressed read, so lookup, copying, and associative recall are natural, but one bounded-depth attention stack still performs only a bounded number of global communication rounds. Positional information is therefore decisive: without positional encodings, a bidirectional transformer layer is naturally permutation equivariant, while with absolute, relative, rotary, or learned positional encodings, the model can distinguish order and distance.

At the positive end, [Yun, Bhojanapalli, Rawat, Reddi, and Kumar 2020](https://arxiv.org/abs/1912.10077) show that transformers are universal approximators of fixed-length sequence functions under suitable assumptions, and [Perez, Barcelo, and Marinkovic 2021](https://www.jmlr.org/papers/v22/20-302.html) together with [Bhattamishra, Patel, and Goyal 2020](https://aclanthology.org/2020.conll-1.37/) give Turing-completeness results under idealized assumptions. The point is not that every transformer is simultaneously a bounded circuit and a universal machine, but that different assumptions on sequence length, positional information, masking, numerical precision, and inference procedure lead to different complexity-theoretic regimes, ranging from low-depth circuit classes such as $\mathsf{AC}^0$ and $\mathsf{TC}^0$ to polynomial-time computation and, in more idealized settings, Turing completeness.

## 2. No-CoT Transformers as Bounded Parallel Computation

The "no-CoT as a bounded circuit" viewpoint studies a transformer that reads the input and immediately produces an answer after a bounded number of layers. In that setting, the computation is highly parallel: every layer updates all positions at once, and a constant-depth model performs only a constant number of global communication rounds. This is why circuit classes such as $\mathsf{AC}^0$ and $\mathsf{TC}^0$ appear throughout the literature.

[Hahn 2020](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00306/43545/Theoretical-Limitations-of-Self-Attention-in) is a canonical early limitation result. Hahn showed that, under soft and hard attention settings, fixed-size self-attention models cannot robustly model some periodic finite-state languages or hierarchical languages unless the number of layers or heads grows with input length. Circuit-complexity analyses sharpen this picture. [Hao, Angluin, and Frank 2022](https://aclanthology.org/2022.tacl-1.46/) study hard-attention transformer encoders as language recognizers and show that unique-hard-attention variants lie within $\mathsf{AC}^0$, while average-hard-attention variants can recognize some languages outside $\mathsf{AC}^0$, including $\mathsf{MAJORITY}$ and $\mathsf{DYCK}$-1 in their setting. [Merrill, Sabharwal, and Smith 2022](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00493/112604/Saturated-Transformers-are-Constant-Depth) analyze saturated transformers and relate them to constant-depth threshold circuits, giving a $\mathsf{TC}^0$ style upper-bound picture.

Logic characterizations make the same idea more structural. [Chiang, Cholak, and Pillay 2023](https://proceedings.mlr.press/v202/chiang23a.html) identify a first-order logic with counting quantifiers that gives both upper and lower bounds for transformer encoders under fixed-precision assumptions, and [Merrill and Sabharwal 2023](https://openreview.net/forum?id=uR8TtWCIsr) show that log-precision transformers can be expressed in first-order logic with majority quantifiers. These results explain why parity, exact counting, graph connectivity, and related tasks become standard separator problems: they probe exactly the limitations of a bounded parallel computation without extra sequential workspace.

## 3. Chain of Thought as a General Computation

Chain of thought changes the computation graph. Without CoT, a decoder-only transformer maps the input context $x$ to an answer token after one forward pass. With CoT, it generates a sequence
$$
z_1,\ldots,z_T
$$
and each step computes from the growing context $(x,z_1,\ldots,z_{t-1})$. Thus $T$ intermediate tokens are not just text; they are $T$ rounds of additional sequential computation and $T$ additional pieces of externalized state. This is why CoT can move a transformer beyond the bounded-parallel regime.

It is useful to distinguish parameter expressiveness from prompt expressiveness. If the transformer weights can be chosen or trained to use scratchpad tokens as a work tape, then the architecture plus decoding procedure is strictly stronger than the same architecture forced to answer immediately. [Merrill and Sabharwal 2023](https://arxiv.org/abs/2310.07923) give a clean complexity-theoretic account: logarithmic-length CoT only modestly extends a standard decoder-only transformer, linear-length CoT can recognize all regular languages under their projected-pre-norm assumptions, linear CoT remains within context-sensitive languages, and polynomial-length CoT with generalized pre-norm characterizes polynomial-time computation in their setting. [Li, Liu, Zhou, and Ma 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3309b4112c9f04a993f2bbdd0274bba1-Abstract-Conference.html) make the serial-computation intuition explicit by showing that constant-depth constant-bit transformers without CoT are bounded by weak parallel classes, while with $T$ CoT steps they can solve problems computable by Boolean circuits of size $T$ under their construction.

Empirically, [Wei et al. 2022](https://arxiv.org/abs/2201.11903) popularized CoT prompting as a way to improve reasoning behavior in large language models, but prompt-space CoT is a weaker and different notion. A frozen pretrained model may improve when prompted to write intermediate reasoning because the prompt changes the distribution of computations the model performs, gives it more tokens for intermediate state, and matches patterns seen in training. That is not the same as saying that prompting freely adds a new algorithm; it is better viewed as steering or eliciting computations that the backbone can already implement unless the prompt is allowed to be a powerful learned prefix.

## 4. Case Studies

The cleanest non-CoT case studies come from formal languages, logic, counting, associative recall, and long-context retrieval. Formal-language expressiveness asks which languages $L\subseteq\Sigma^\ast$ a transformer recognizer can accept uniformly over all input lengths. [Strobl, Merrill, Weiss, Chiang, and Angluin 2024](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00663/120983/What-Formal-Languages-Can-Transformers-Express-A) survey this line. A particularly sharp result is [Yang, Chiang, and Angluin 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/13d7f172259b11b230cc5da8768abc5f-Abstract-Conference.html), which proves that masked hard-attention transformers with strict masking and no position embeddings recognize exactly the star-free languages. Counting gives another canonical test family. A simple task such as
$$
c_a(x_1,\ldots,x_n)=\sum_{i=1}^n \mathbf{1}\{x_i=a\}
$$
already stresses exact aggregation over long sequences, and [Yehudai, Kaplan, Ghandeharioun, Geva, and Globerson 2024](https://arxiv.org/abs/2407.15160) study when transformers can count occurrences under dimension and context constraints.

Associative recall and mechanistic circuits give a second family of examples. The model sees key-value pairs
$$
(k_1,v_1),\ldots,(k_m,v_m),q
$$
and should output the value $v_j$ corresponding to the query key. Attention is naturally suited to this task because query-key similarity can select a position and copy its value, but success still depends on positional encoding, key collisions, distractor structure, and numerical precision. [Olsson et al. 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) identify induction-head circuits for copy-and-continue behavior, and [Nichani, Lee, and Bietti 2024](https://openreview.net/forum?id=PtYojIoW0u) study factual recall through associative memories.

Long context raises an expressiveness question because the model must preserve useful positional information as the sequence becomes long, so failures of positional encoding or attention scaling can already break simple retrieval; Needle-in-a-haystack (NIAH) is the standard basic probe for this. A separate but related issue is length generalization: a model trained on shorter sequences may be expressive enough in principle yet still fail on longer ones because it learned a shortcut tied to the training length rather than the underlying algorithm. [Weiss, Goldberg, and Yahav 2021](https://proceedings.mlr.press/v139/weiss21a.html) and [Lindner et al. 2023](https://arxiv.org/abs/2301.05062) use RASP-style program descriptions and compilation to make such questions concrete, [Zhou et al. 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/45ed1a72597594c097152ef9cc187762-Abstract-Conference.html) relate short RASP-like solutions to better length generalization, and empirical benchmarks such as [LongBench](https://arxiv.org/abs/2308.14508) and [RULER](https://arxiv.org/abs/2404.06654) separate simple retrieval from harder aggregation and multi-hop reasoning.

In the CoT regime, the same kinds of tasks become separator problems for sequential workspace rather than bounded parallelism. The papers above already point to the canonical examples: permutation composition, iterated squaring, circuit value, regular-language recognition with generated scratchpad, and other tasks whose main difficulty is repeated dependency propagation across many steps. These examples are useful precisely because they let one see when extra visible workspace changes the reachable complexity class and when it merely changes prompting style.

## 5. Alternative Architectures

Explicit CoT is not the only way to add inference-time computation. A first family of alternatives stays close to transformers themselves: looped transformers and architectures that rely more heavily on persistent hidden states. The [Universal Transformer](https://arxiv.org/abs/1807.03819) introduced a recurrence-like transformer architecture, and [Giannou et al. 2023](https://proceedings.mlr.press/v202/giannou23a.html) show that looped transformers can be programmed as general-purpose computers by using the input as instructions and memory. Looping differs from CoT because the intermediate state remains latent in hidden states rather than being written into visible tokens.

A second family compares transformers with recurrent models and recurrence-like variants. [Weiss, Goldberg, and Yahav 2018](https://aclanthology.org/P18-2117/) show that practical finite-precision recurrent models have their own formal hierarchy, [Merrill et al. 2020](https://aclanthology.org/2020.acl-main.43/) develop a hierarchy of RNN architectures, and [Katharopoulos et al. 2020](https://arxiv.org/abs/2006.16236) show that linear-attention transformers can be implemented recurrently. This is the right conceptual place to compare transformers with classical RNN-style systems and with newer recurrent or state-space variants such as Mamba and RWKV, or with more specialized recurrence-like proposals such as TTT.

A third family adds external memory or auxiliary state. Padding and pause tokens provide extra workspace positions, and [Merrill and Sabharwal 2025](https://arxiv.org/abs/2505.18948) together with [London and Kanade 2025](https://arxiv.org/abs/2505.21024) show that such resources can strictly increase expressiveness in formal settings. More practically, transformers with memory and retrieval-augmented generation (RAG) add external storage rather than relying only on the visible prompt or the hidden state. The common theme across all these alternatives is that hidden state, recurrence, or memory can play the role that explicit CoT otherwise plays as test-time computational state.

## 6. In-Context Learning and Prompt-Space Expressiveness

In-context learning asks whether a fixed transformer can use examples inside the context to infer a task at test time. A typical prompt has the form
$$
(x_1,y_1),\ldots,(x_k,y_k),x_{\mathrm{query}},
$$
and the model should output something close to $f(x_{\mathrm{query}})$ for the latent function $f$ that generated the examples. This is not ordinary supervised training, because $\theta$ is not updated during the prompt; the learning algorithm, if any, must be implemented inside the forward computation. [Garg, Tsipras, Liang, and Valiant 2022](https://openreview.net/forum?id=flNZJ2eOet) study this by training transformers to in-context learn simple function classes such as linear functions and decision trees, and [von Oswald et al. 2023](https://proceedings.mlr.press/v202/von-oswald23a.html) show that linear self-attention can implement an update resembling gradient descent on a regression loss.

Prompt tuning and prefix tuning ask a related but distinct question. Instead of using ordinary examples in the context, we prepend trainable vectors or tokens while freezing the backbone. [Lester, Al-Rfou, and Constant 2021](https://aclanthology.org/2021.emnlp-main.243/) show empirically that soft prompt tuning becomes more competitive as model scale increases. Theoretical work then studies what functions can be induced by changing only those prompt parameters. [Wang, Chauhan, Wang, and Hsieh 2023](https://arxiv.org/abs/2305.18787) prove universality results for prompt tuning in stylized transformer settings and also prove limitations for finite-depth fixed-weight models. [Petrov, Torr, and Bibi 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/18a367688479c1b08b001584218a4443-Abstract-Conference.html) show that context-based fine-tuning methods can be less expressive than full fine-tuning under their assumptions because they cannot freely change relative attention patterns over content tokens, while [Petrov, Torr, and Bibi 2024](https://proceedings.mlr.press/v235/petrov24a.html) give positive universality results for certain pretrained transformer constructions.

The synthesis is that prompt-space expressiveness is conditional. If the frozen backbone already contains the right computational primitives, prompt tokens can select, compose, or steer them. If the task requires an attention pattern or algorithm not present in the backbone and not realizable through the allowed prompt interface, prompting is limited. This is why one must keep apart trainable-weight expressiveness, no-CoT bounded-circuit expressiveness, CoT as sequential workspace, and prompt-space control of a frozen model: they are related, but they are not the same notion.
