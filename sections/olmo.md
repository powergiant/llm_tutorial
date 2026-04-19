# OLMo Reproduction Survey

This note mirrors the structure of [nemotron.md](./nemotron.md) and surveys the public OLMo artifacts available as of 2026-04-17. The practical question is the same: how close can an external team get to reproducing a general LLM in the OLMo line?

## Short Answer

- `OLMo` is one of the strongest public reproduction targets in the LLM ecosystem because AI2 has consistently released not only weights, but also training data, code, recipes, logs, intermediate checkpoints, and evaluation tooling. [1][2][3][5]
- `OLMo 2` is the cleanest dense reproduction target today. AI2 explicitly states that OLMo 2 releases full training data, code and recipes, logs, and thousands of intermediate checkpoints. [9][10]
- `OLMo 3` is the most ambitious fully open target. AI2 describes it as releasing the entire model flow, including every stage, checkpoint, data point, and dependency used to build the family. [13][18][19]
- `OLMoE` is the open sparse-MoE branch of the family. It is useful if the goal is to reproduce an MoE training stack rather than a dense autoregressive stack. [6][7][8]

## OLMo Product Series

The table below lists the distinct OLMo language-model series I could verify from AI2's public pages.

| Series | First public release date | Main difference |
| --- | --- | --- |
| `OLMo` | February 1, 2024 [1] | The original dense, fully open family. AI2 released 1B and 7B models together with Dolma, training code, checkpoints, logs, and evaluation artifacts. |
| `OLMoE` | September 4, 2024 [6][7] | The sparse Mixture-of-Experts branch of OLMo. The flagship release has 7B total parameters with about 1B active per token and is positioned as a fully open MoE model. |
| `OLMo 2` | November 26, 2024 [9] | The second dense generation with a stronger architecture and training recipe, new data mixtures, explicit two-stage pretraining, and more competitive instruct models. |
| `OLMo 3` | November 20, 2025 [18][19] | The third generation, organized as a full model flow rather than only final checkpoints. It emphasizes long-context reasoning, tool use, RL pathways, and open post-training flows. |

### Difference Inside OLMo 2

`OLMo 2` is a family rather than a single checkpoint. Its main public sizes differ as follows. [9][10][11]

| OLMo 2 tier | First public release date | Difference |
| --- | --- | --- |
| `7B` | November 26, 2024 [9] | The initial flagship dense base and instruct size for the OLMo 2 release. |
| `13B` | November 26, 2024 [9] | A larger dense model trained on more total tokens and used to demonstrate stronger compute efficiency. |
| `32B` | March 13, 2025 [11] | The largest dense OLMo 2 model and, according to AI2, the first fully open model to outperform GPT-3.5 Turbo and GPT-4o mini on their cited benchmark suite. |
| `1B` | Present on the official OLMo 2 release page by April 2025. I did not verify a separate standalone blog announcement earlier than the official release page. [13] | The smallest OLMo 2 member, useful for recipe iteration and lower-cost experimentation. |

### Difference Inside OLMo 3

`OLMo 3` is organized less as a single model line and more as an open model flow. [18][19]

| OLMo 3 path | First public release date | Difference |
| --- | --- | --- |
| `Base` | November 20, 2025 [18] | The from-scratch pretrained foundation model, released at 7B and 32B. |
| `Instruct` | November 20, 2025 [18] | The chat and tool-use path derived from the same base model through SFT, DPO, and RLVR. |
| `Think` | November 20, 2025 [18] | The reasoning-focused path optimized for longer chain-of-thought style reasoning and agentic tasks. |
| `RL Zero` | November 20, 2025 [18] | A fully open reinforcement-learning path built directly on top of OLMo 3 base models for RL research. |
| `OLMo 3.1` | December 12, 2025 update on the OLMo 3 launch post [18] | A stronger continuation of the OLMo 3 post-training path, especially at 32B, rather than a separate from-scratch pretraining family. |

## What AI2 Has Released

- model weights and checkpoints for multiple OLMo families on Hugging Face [1][6][9][13][20]
- technical reports and papers describing architecture, training, and evaluation choices [2][5][7][10][19]
- open pretraining data, including Dolma, OLMo-Mix-1124, and Dolmino-Mix-1124 [4][5][16][17]
- open training code through `OLMo`, later `OLMo-core`, plus configuration and recipe artifacts [3][12]
- open post-training code and data pathways through `open-instruct` and OLMo family model cards [15][21]
- open evaluation tooling through `olmes`, and earlier evaluation artifacts through Catwalk and Paloma referenced in the original OLMo release [1][14]
- training logs and many intermediate checkpoints, which are unusually important for real reproducibility [1][10]

## What Is Still Missing Or Still On Us

- matching AI2's exact hardware environment, cluster topology, storage performance, and orchestration stack
- exact internal experiment history outside the released logs and checkpoints, including failed runs and discarded ablations
- reproducing the same wall-clock efficiency and infrastructure behavior on a different cluster
- exact deployment stack choices for production serving, latency targets, and kernel-level inference tuning
- in some later post-training and evaluation flows, reconstructing every judge model, API dependency, and tool-chain assumption exactly

These are mostly operational gaps, not fundamental openness gaps. Compared with many other open model families, OLMo leaves relatively little hidden.

## Full Training Stack

| Training stack aspect | Release status | What is public | What you would still need to do |
| --- | --- | --- | --- |
| Data sourcing, licensing, collection, cleaning, deduplication, filtering, and mixture design | Open to partial | AI2 released Dolma, documented its construction in detail, and later released OLMo-Mix-1124 and Dolmino-Mix-1124 for OLMo 2. OLMo 3 also exposes its pretraining and post-training collections through the model-flow release. [4][5][16][17][18][20] | Assemble the exact release-specific mix you want to reproduce and verify you are using the right dataset version and sampling schedule for that family. |
| Tokenizer design and vocabulary construction | Partial to open | Tokenizer artifacts are public with checkpoints and model cards, and the codebase is open. [3][9][13][21] | Reconstruct tokenizer training from scratch only if you need exact tokenizer creation provenance rather than reuse of the released tokenizer. |
| Data formatting, document packing, sharding, and dataloading | Open to partial | Training code, configs, and later OLMo-core abstractions are public. [3][12] | Match the exact release-era packing, sharding, and cache behavior if you want bit-for-bit or curve-for-curve fidelity. |
| Model architecture design | Open | OLMo, OLMo 2, OLMoE, and OLMo 3 all publish architecture details in papers, model cards, and code. [2][7][10][19][21] | Mostly choose the right family and checkpoint lineage rather than reverse-engineering hidden architecture decisions. |
| Parameter scale, context length, and compute budget | Open to partial | AI2 publishes sizes, token counts, and major scaling decisions across families. [1][10][11][19][21] | Turn those public disclosures into a real compute plan on your own hardware budget. |
| Distributed training strategy, parallelism, and fault tolerance | Partial | OLMo-core and OLMo repos expose the training framework and distributed abstractions. [3][12] | Match the exact topology, failure recovery settings, and job-level behavior on your cluster. |
| Training infrastructure, hardware, networking, storage, and orchestration | Partial | AI2 discusses infrastructure choices and exposes the software layer through open repos. OLMo-core explicitly targets large-scale modern hardware. [11][12] | Provide your own cluster and accept that exact system performance will differ. |
| Optimization setup: objectives, optimizer, scheduler, batch size, precision, initialization, regularization, and stabilization | Open to partial | OLMo family reports disclose much more than typical open releases, especially for OLMo 2 and OLMo 3. [2][10][19] | Pull the exact config set for the family and stage you want, then validate that your implementation follows the intended schedule and stability knobs. |
| Continued pretraining or domain adaptation | Open | OLMo 2 openly documents two-stage pretraining with OLMo-Mix-1124 followed by Dolmino-Mix-1124; OLMo 3 exposes staged model flow and mid-training. [9][10][16][17][18][19] | Reproduce the exact stage boundaries, token counts, and dataset transitions. |
| Supervised fine-tuning | Open | AI2 publishes post-training paths, instruct checkpoints, and uses open-instruct for the code path. [1][9][13][15][18][21] | Re-run the exact stage recipe and select the intended released checkpoint. |
| Preference tuning, reward modeling, RLHF/RLAIF, or other post-training methods | Open to partial | OLMo 2 openly incorporates DPO and RLVR in its instruct path, and OLMo 3 exposes explicit RL Zero and Think pathways. [10][15][18][19][21] | Reproduce the exact rollout settings, reward definitions, and later-stage continuation schedules. |
| Safety tuning, alignment, refusals, policy shaping, and red teaming | Partial | OLMo exposes post-training data and code, but safety is embedded within broader post-training releases rather than isolated as a dedicated safety stack. [15][18][19] | Decide whether you want to reproduce AI2's released behavior exactly or add your own safety policy layer. |
| Evaluation: pretraining metrics, downstream benchmarks, reasoning, coding, safety, robustness, and long-context evaluation | Open | AI2 releases evaluation code and benchmark suites through `olmes`, and earlier OLMo releases referenced open evaluation tooling such as Catwalk and Paloma. [1][14][19] | Pin exact benchmark versions, judge settings, and external dependencies when reproducing later results. |
| Inference-time optimization, serving assumptions, quantization, and deployment constraints | Partial | Model cards, Transformers support, and open repositories make inference straightforward. [3][13][21] | Build your own serving stack and tune for your own latency and cost constraints. |
| Reproducibility, experiment tracking, ablations, and monitoring | Open to partial | OLMo is unusually strong here because AI2 releases logs, many checkpoints, code, and evaluation tools. [1][3][10][14] | Even with this openness, you still do not inherit AI2's full internal experiment database or production monitoring environment. |

## Aspect-By-Aspect Reproduction Notes

### 1. Data sourcing, licensing, collection, cleaning, deduplication, filtering, and mixture design

This aspect defines what raw text enters training and how it is transformed into a usable corpus.

- What it needs: source lists, licenses, collection procedures, cleaning rules, deduplication method, quality filters, and final mixture weights.
- What AI2 opened: Dolma, its paper, the data curation toolkit, and later OLMo-Mix-1124 and Dolmino-Mix-1124 for OLMo 2, plus OLMo 3 data collections. [4][5][16][17][18][20]
- Is it sufficient: largely yes.
- Why not fully: the main remaining work is operational. You still need to select the exact dataset version and stage schedule for the model family you want to reproduce.

### 2. Tokenizer design and vocabulary construction

This aspect defines how text is split into tokens before training.

- What it needs: tokenizer files, normalization rules, vocabulary size, special tokens, and ideally tokenizer training provenance.
- What AI2 opened: tokenizer artifacts with model releases and open codebases. [3][9][13][21]
- Is it sufficient: mostly yes for model reproduction, less so for tokenizer-training archaeology.
- Why not fully: in practice you can reuse the released tokenizer, but reconstructing tokenizer creation from first principles may still require additional digging through configs and release artifacts.

### 3. Data formatting, document packing, sharding, and dataloading

This aspect turns curated corpora into the exact training examples and batches seen by the model.

- What it needs: serialization format, sample boundaries, packing rules, shard layout, sampling scheme, and dataloader implementation.
- What AI2 opened: the OLMo and OLMo-core training stacks. [3][12]
- Is it sufficient: close to yes.
- Why not fully: the remaining gap is exact operational fidelity across code versions and hardware environments, not secrecy.

### 4. Model architecture design

This aspect defines the neural network itself.

- What it needs: layer layout, attention or MoE design, positional encoding, normalization, activations, and tensor shapes.
- What AI2 opened: papers, code, and model cards for dense and MoE families. [2][7][10][19][21]
- Is it sufficient: yes for practical reproduction.
- Why not fully: only because implementation details can vary by release branch and code version; the architecture itself is not hidden.

### 5. Parameter scale, context length, and compute budget

This aspect determines how large the model is and how much training compute it consumes.

- What it needs: parameter counts, context length, token budget, and training schedule.
- What AI2 opened: family sizes, token counts, and scaling disclosures across OLMo, OLMo 2, OLMoE, and OLMo 3. [1][7][10][11][19][21]
- Is it sufficient: mostly yes.
- Why not fully: public numbers still have to be translated into an actual procurement and scheduling plan on your hardware.

### 6. Distributed training strategy, parallelism, and fault tolerance

This aspect determines how training is spread across many accelerators and how jobs recover from failure.

- What it needs: data, tensor, pipeline, and expert parallel settings as applicable, plus checkpointing and recovery strategy.
- What AI2 opened: open training repos and, for newer releases, OLMo-core. [3][12]
- Is it sufficient: partially.
- Why not: code is open, but exact cluster topology and job orchestration are still environment-specific.

### 7. Training infrastructure, hardware, networking, storage, and orchestration

This aspect covers the physical and systems layer on which the run is executed.

- What it needs: GPU type, interconnect, storage bandwidth, scheduler, containers, and orchestration scripts.
- What AI2 opened: the software stack and many training assumptions through code and reports. [11][12]
- Is it sufficient: no.
- Why not: AI2 cannot release your cluster for you. This is the clearest remaining reproduction gap.

### 8. Optimization setup: objectives, optimizer, scheduler, batch size, precision, initialization, regularization, and stabilization

This aspect determines how the model learns during pretraining and post-training.

- What it needs: loss definitions, optimizer choice, LR schedule, batch size, precision mode, clipping, and stability settings.
- What AI2 opened: unusually rich optimization detail in OLMo family reports, especially OLMo 2 and OLMo 3. [2][10][19]
- Is it sufficient: mostly yes.
- Why not fully: some exact run-level knobs still depend on selecting the intended config revision and trainer version.

### 9. Continued pretraining or domain adaptation

This aspect covers additional large-scale training after the first pretraining stage.

- What it needs: stage boundaries, stage-specific data mixtures, stopping rules, and checkpoint lineage.
- What AI2 opened: OLMo 2's two-stage pretraining and OLMo 3's staged model flow. [9][10][16][17][18][19]
- Is it sufficient: yes for practical reproduction.
- Why not fully: you still need to execute the exact stage schedule on your infrastructure.

### 10. Supervised fine-tuning

This aspect teaches the model to follow instructions and preferred response formats.

- What it needs: prompt-response data, templates, training objective, curriculum, and checkpoint selection criteria.
- What AI2 opened: instruct checkpoints, post-training data pathways, and open-instruct. [1][9][13][15][18][21]
- Is it sufficient: mostly yes.
- Why not fully: exact release matching still requires selecting the exact recipe branch and post-training stage.

### 11. Preference tuning, reward modeling, RLHF/RLAIF, or other post-training methods

This aspect improves behavior beyond plain SFT by using preferences, rewards, or reinforcement learning.

- What it needs: preference data or reward setup, rollout environment, optimization algorithm, and filtering logic.
- What AI2 opened: OLMo 2's DPO and RLVR path, open-instruct, and OLMo 3's RL Zero and Think pathways. [10][15][18][19]
- Is it sufficient: stronger than most open families, but still partial.
- Why not: RL systems are sensitive to infrastructure and run settings, so artifact openness does not automatically guarantee curve-matching reproduction.

### 12. Safety tuning, alignment, refusals, policy shaping, and red teaming

This aspect constrains the model to behave safely and within product policy.

- What it needs: safety datasets, policy targets, refusal criteria, and evaluation procedures.
- What AI2 opened: open post-training flows and model cards, but not a standalone safety release analogous to a dedicated safety benchmark suite plus policy pack in every generation. [15][18][19]
- Is it sufficient: partial.
- Why not: OLMo is very open, but the safety layer is distributed across the broader post-training stack rather than packaged as a single dedicated reproducibility artifact.

### 13. Evaluation: pretraining metrics, downstream benchmarks, reasoning, coding, safety, robustness, and long-context evaluation

This aspect determines whether the model is good enough to release.

- What it needs: benchmark suites, prompts, scoring logic, contamination control, and regression tracking.
- What AI2 opened: OLMES plus earlier open evaluation tooling, and published benchmark suites in papers. [1][14][19]
- Is it sufficient: yes for serious external reproduction.
- Why not fully: later eval flows may still depend on external APIs or pinned benchmark versions that you need to control carefully.

### 14. Inference-time optimization, serving assumptions, quantization, and deployment constraints

This aspect determines how the released model is run efficiently after training.

- What it needs: serving backend, kernel choices, memory assumptions, and latency-throughput targets.
- What AI2 opened: model cards, standard Transformers compatibility, and open code. [3][13][21]
- Is it sufficient: enough to use and study the models, not enough to inherit AI2's serving stack.
- Why not: deployment is still your responsibility.

### 15. Reproducibility, experiment tracking, ablations, and monitoring

This aspect makes the entire pipeline auditable and repeatable.

- What it needs: exact configs, seeds, logs, checkpoint lineage, evaluation outputs, and ablation records.
- What AI2 opened: this is where OLMo is strongest. The family releases logs, many checkpoints, code, datasets, and evaluation tooling. [1][3][10][14][19]
- Is it sufficient: closer to yes than almost any comparable family.
- Why not fully: even here, you still do not get every internal experiment that never became part of a release.

## Practical Conclusion

If the goal is to reproduce a general LLM from scratch as faithfully as possible, `OLMo` is one of the best public targets available.

If you want the cleanest dense end-to-end reproduction target, start with `OLMo 2`: its artifacts are mature, explicitly documented, and tied to a detailed technical report. If you want the most advanced fully open target, study `OLMo 3`: it extends openness beyond final checkpoints to the full model flow. If you want an open sparse model, use `OLMoE`.

Compared with the Nemotron survey, the main difference is simple: in OLMo, the missing pieces are mostly operational rather than secret. That is the core reason OLMo is such a strong reproduction target.

## References

1. [OLMo: Open Language Model](https://allenai.org/blog/olmo-open-language-model-87ccfc95f580)
2. [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)
3. [allenai/OLMo](https://github.com/allenai/OLMo)
4. [Ai2 Dolma: 3 trillion token open corpus for language model pretraining](https://allenai.org/blog/dolma-3-trillion-tokens-open-llm-corpus-9a0ff4b8da64)
5. [Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research](https://arxiv.org/abs/2402.00159)
6. [OLMoE: an open, small, and state-of-the-art mixture-of-experts model](https://allenai.org/blog/olmoe-an-open-small-and-state-of-the-art-mixture-of-experts-model-c258432d0514)
7. [OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2409.02060)
8. [allenai/OLMoE](https://github.com/allenai/OLMoE)
9. [OLMo 2: The best fully open language model to date](https://allenai.org/blog/olmo2)
10. [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656)
11. [OLMo 2 32B: First fully open model to outperform GPT 3.5 and GPT 4o mini](https://allenai.org/blog/olmo2-32b)
12. [allenai/OLMo-core](https://github.com/allenai/OLMo-core)
13. [Olmo](https://allenai.org/olmo)
14. [allenai/olmes](https://github.com/allenai/olmes)
15. [allenai/open-instruct](https://github.com/allenai/open-instruct)
16. [allenai/olmo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124)
17. [allenai/dolmino-mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124)
18. [Olmo 3: Charting a path through the model flow to lead open-source AI](https://allenai.org/blog/olmo3)
19. [Olmo 3](https://arxiv.org/abs/2512.13961)
20. [Olmo 3 collection](https://huggingface.co/collections/allenai/olmo-3)
21. [Olmo 3 7B model card](https://huggingface.co/allenai/Olmo-3-1025-7B)
