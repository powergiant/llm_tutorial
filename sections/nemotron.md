# Nemotron Reproduction Survey

This note surveys the public Nemotron artifacts available as of 2026-04-17 and asks a practical question: how close can an external team get to reproducing a general LLM in the Nemotron line?

## Short Answer

- `Nemotron 3` is the strongest reproduction target. NVIDIA now states that Nemotron ships open weights, training data, and recipes, and the Nemotron 3 pages explicitly link checkpoints, datasets, technical reports, and a developer repository with reproducible pipelines. [1][2][3]
- `Nemotron-4` is only partially reproducible. NVIDIA released weights, model cards, a technical report, and alignment tooling, but the public material is less complete than the newer Nemotron 3 release. [4][5][6]
- `Llama-Nemotron` models are not fully reproducible from scratch because they are derivatives of Meta Llama checkpoints. They are useful for post-training study, but they are not a clean from-scratch open-base reproduction target. [7][8]

## Nemotron Product Series

The table below lists the distinct Nemotron-branded LLM series I could verify from NVIDIA's public pages.

| Series | First public release date | Main difference |
| --- | --- | --- |
| `Nemotron-4` | June 14, 2024 [5][17] | A 340B Transformer family centered on synthetic-data generation for downstream LLM training. It includes Base, Instruct, and Reward variants and was the first major open Nemotron family with a full synthetic-data-generation story. |
| `Nemotron-H` | March 21, 2025 [18] | A hybrid Mamba-Transformer family focused on higher accuracy at the same inference cost. It includes base, instruct, and VLM variants and emphasizes inference efficiency rather than only scale. |
| `Llama-Nemotron` | March 2025 [19] | A reasoning-focused family built by post-training Meta Llama models, offered in Nano, Super, and Ultra sizes. Its main difference is that it is derived from Llama rather than trained as a fully independent Nemotron base model. |
| `Nemotron Nano 2` | August 18, 2025 [20] | A smaller hybrid reasoning family with 128K context and major data release around pretraining. It is more compact than Nemotron-4 and is positioned around efficient open reasoning models. |
| `Nemotron 3` | December 15, 2025 [2] | NVIDIA's more fully open agentic family with open weights, data, and recipes. It is organized into Nano, Super, and Ultra tiers for efficiency, collaborative-agent workloads, and top-end reasoning respectively. |
| `Nemotron-Cascade` | December 15, 2025 [21] | A post-training series built with Cascade RL for general-purpose reasoning. Its main difference is that it is an RL recipe and model line layered on top of pretrained Nemotron bases rather than a new base pretraining family. |
| `Nemotron-Cascade 2` | March 16, 2026 [22] | A second-generation Cascade RL series with broader domain coverage and multi-domain on-policy distillation. It improves reasoning and agentic performance over the first Cascade release. |

### Difference Inside Nemotron 3

`Nemotron 3` itself has three product tiers. NVIDIA describes them as follows. [2][23]

| Nemotron 3 tier | Public release date | Difference |
| --- | --- | --- |
| `Nano` | December 15, 2025 [2] | The smallest and most cost-efficient tier, aimed at strong reasoning and agentic use with lower inference cost. |
| `Super` | March 10, 2026 [23] | A 120B total / 12B active hybrid LatentMoE model optimized for collaborative agents, long context, and high-volume workloads; first in the series with LatentMoE, MTP, and NVFP4 pretraining. |
| `Ultra` | Announced as part of the family on December 15, 2025; I did not verify a separate standalone release page in the sources above. [2] | The largest tier, positioned for the highest reasoning accuracy in the family. |

## What NVIDIA Has Released

- model weights and checkpoints for multiple Nemotron families on Hugging Face [1][2][4][7][8]
- technical reports and whitepapers describing model design and training at a high level [1][2][5]
- pretraining, post-training, safety, and RL datasets under the Nemotron umbrella [1][2][9][10]
- a public Nemotron developer repository with training recipes and end-to-end reference pipelines [3]
- open-source training and post-training tooling across NeMo, NeMo Curator, NeMo RL, NeMo Gym, and NeMo Evaluator [6][11][12][13][14][15][16]

## What Is Still Missing Or Still On Us

- matching NVIDIA's exact hardware environment, cluster topology, storage system, and job orchestration
- exact internal experiment history, failed runs, hyperparameter sweeps, and ablation trail
- any non-redistributable data or preprocessing details not included in the published datasets
- exact tokenizer construction procedure for every generation; tokenizer artifacts are public with checkpoints, but tokenizer training pipelines are not consistently documented
- exact data mixtures and filtering thresholds for every historical Nemotron generation
- exact deployment stack choices and inference kernels used in NVIDIA's internal production settings

The last four points are partly an inference from the absence of direct public artifacts in the cited sources.

## Full Training Stack

| Training stack aspect | Release status | What is public | What you would still need to do |
| --- | --- | --- | --- |
| Data sourcing, licensing, collection, cleaning, deduplication, filtering, and mixture design | Partial to open | Nemotron 3 publishes major pretraining datasets and NVIDIA also released Nemotron-CC plus the Curator toolkit used for large-scale curation. Nemotron-4 discloses corpus scope but not a comparably complete raw-data release. [1][2][4][9][11][12] | Verify redistribution coverage, reconstruct any unreleased sources, and decide whether to exactly mirror NVIDIA's blends or build a compatible substitute. |
| Tokenizer design and vocabulary construction | Partial | Tokenizer artifacts are bundled with released checkpoints and model repositories on Hugging Face. [2][4][7][8] | Rebuild tokenizer training if the exact corpus, normalization rules, and vocabulary training recipe are not public. |
| Data formatting, document packing, sharding, and dataloading | Partial | The Nemotron developer repo claims complete reproducible pipelines, and NeMo provides large-scale training infrastructure. [3][13] | Recreate the exact packing, sharding, cache layout, and dataloader behavior used in NVIDIA's production runs. |
| Model architecture design | Open | Nemotron-4 model cards and Nemotron 3 reports disclose the architecture family, context length, attention scheme, and major design choices. [2][4][5] | Implement the exact architecture variant if a report omits minor details, or reuse NVIDIA's released recipes where available. |
| Parameter scale, context length, and compute budget | Partial | Public model cards and reports disclose parameter counts, active parameter counts, sequence length, and broad training scale. [2][4][5][8] | Convert those disclosures into a concrete budget, schedule, and hardware procurement plan. |
| Distributed training strategy, parallelism, and fault tolerance | Partial | NeMo is NVIDIA's open training framework for large-scale multi-GPU and multi-node training, and Nemotron recipes are published in the Nemotron repo. [3][13] | Match the exact parallelism topology, checkpointing cadence, recovery strategy, and cluster-level reliability behavior. |
| Training infrastructure, hardware, networking, storage, and orchestration | Partial | NVIDIA discloses hardware expectations in model cards and exposes the software stack around NeMo. [1][4][13] | Provide your own cluster, schedulers, interconnect assumptions, storage bandwidth, and job orchestration. |
| Optimization setup: objectives, optimizer, scheduler, batch size, precision, initialization, regularization, and stabilization | Partial to open | Nemotron-4 publicly discloses batch size, sequence length, and training phases; Nemotron 3 reports disclose precision and major optimization choices at a high level. [2][4][5] | Fill in any omitted optimizer hyperparameters and stabilization tricks, then validate they reproduce the same loss curve. |
| Continued pretraining or domain adaptation | Open for some generations, partial overall | Nemotron-4 explicitly describes an 8T pretrain followed by 1T continued pretraining, and Llama-Nemotron CPT models are published as continual-pretraining derivatives. [4][8] | Recreate the exact CPT mixture and schedule when NVIDIA does not release every input shard. |
| Supervised fine-tuning | Open | NVIDIA publishes post-training datasets and open alignment/training repos; Nemotron 4 Instruct and Nemotron 3 SFT data are public. [2][6][7][14] | Reproduce exact prompt formatting, curriculum, and run selection. |
| Preference tuning, reward modeling, RLHF/RLAIF, or other post-training methods | Open to partial | NVIDIA released NeMo-Aligner, NeMo RL, NeMo Gym, reward-related models, and Nemotron RL datasets. [1][2][6][14][15] | Rebuild the exact reward shaping, environment mix, rollout settings, and filtering logic for the target model generation. |
| Safety tuning, alignment, refusals, policy shaping, and red teaming | Partial to open | NVIDIA publishes Nemotron safety datasets and alignment tooling. [1][6][14][15] | Recreate the exact safety policies, policy weights, and red-team protocols used internally. |
| Evaluation: pretraining metrics, downstream benchmarks, reasoning, coding, safety, robustness, and long-context evaluation | Open | Model cards report benchmark results and NeMo Evaluator is open. Nemotron pages also list benchmark families used in release materials. [1][2][4][16] | Reproduce the benchmark harness versions, prompts, judge models, and contamination controls. |
| Inference-time optimization, serving assumptions, quantization, and deployment constraints | Open to partial | NVIDIA publishes quantized checkpoints, TensorRT-LLM support, and deployment cookbooks through the Nemotron developer page and repo. [1][2][3] | Match the exact runtime kernels, serving topology, and latency-throughput tuning used by NVIDIA. |
| Reproducibility, experiment tracking, ablations, and monitoring | Partial | The Nemotron repo, reports, and NeMo Evaluator materially improve reproducibility. [2][3][16] | Rebuild internal tracking, run provenance, monitoring, and ablation history; that operational layer is not fully public. |

## Aspect-By-Aspect Reproduction Notes

### 1. Data sourcing, licensing, collection, cleaning, deduplication, filtering, and mixture design

This aspect defines what raw text enters training and how it is transformed into a usable corpus.

- What it needs: source lists, licenses, crawl or collection procedures, cleaning rules, deduplication method, quality filters, safety filters, and final mixture weights.
- What NVIDIA opened: Nemotron pretraining datasets, Nemotron-CC, and NeMo Curator. Nemotron 3 also links released pretraining datasets and states that the data they can redistribute is open. [1][2][9][10][11][12]
- Is it sufficient: not fully.
- Why not: released datasets help a lot, but exact historical mixtures, unreleased sources, and every threshold and filtering decision are not fully specified for all Nemotron generations, especially older ones.

### 2. Tokenizer design and vocabulary construction

This aspect defines how text is split into tokens before training.

- What it needs: tokenizer files, normalization rules, vocabulary size, special tokens, training corpus, and tokenizer training recipe.
- What NVIDIA opened: tokenizer artifacts are distributed with checkpoints and model repositories. [2][4][7][8]
- Is it sufficient: partially.
- Why not: using the published tokenizer is enough to run the released models, but it is not always enough to independently reproduce tokenizer training from scratch.

### 3. Data formatting, document packing, sharding, and dataloading

This aspect turns curated corpora into the exact training examples and batches seen by the model.

- What it needs: serialization format, sample boundaries, packing rules, truncation rules, shard layout, sampling scheme, and dataloader implementation.
- What NVIDIA opened: the Nemotron developer repository and NeMo framework provide recipe and infrastructure support. [3][13]
- Is it sufficient: partially.
- Why not: public recipes indicate the overall path, but exact packing heuristics, cache behavior, and production dataloader details are not fully described in the surveyed public sources.

### 4. Model architecture design

This aspect defines the neural network itself.

- What it needs: layer layout, attention or state-space blocks, MoE design if any, positional encoding, normalization, activation functions, and exact tensor shapes.
- What NVIDIA opened: model cards and technical reports disclose major architecture decisions for Nemotron-4 and Nemotron 3. [2][4][5]
- Is it sufficient: mostly yes.
- Why not: the high-level design is public, but some implementation details may still need to be inferred from released code or checkpoints.

### 5. Parameter scale, context length, and compute budget

This aspect determines how large the model is and how much training compute it consumes.

- What it needs: parameter counts, active parameter counts for sparse models, context length, token budget, hardware assumptions, and planned training duration.
- What NVIDIA opened: model cards and reports provide parameter counts, context windows, and broad token counts. [2][4][5][8]
- Is it sufficient: partially.
- Why not: these disclosures are enough for planning, but not enough to reconstruct the full cost model or exact compute allocation.

### 6. Distributed training strategy, parallelism, and fault tolerance

This aspect determines how training is spread across many accelerators and how jobs recover from failure.

- What it needs: tensor, pipeline, data, and expert parallel settings; checkpointing cadence; restart behavior; and cluster recovery policy.
- What NVIDIA opened: NeMo and published Nemotron recipes expose the general software stack for large-scale training. [3][13]
- Is it sufficient: partially.
- Why not: public tooling shows how to train at scale, but not necessarily the exact parallelism and fault-tolerance choices used for a specific internal run.

### 7. Training infrastructure, hardware, networking, storage, and orchestration

This aspect covers the physical and systems layer on which the run is executed.

- What it needs: GPU type, node shape, interconnect, storage bandwidth, scheduler, containers, and orchestration scripts.
- What NVIDIA opened: model cards mention hardware expectations, and NeMo exposes the software framework. [1][4][13]
- Is it sufficient: no.
- Why not: NVIDIA does not publish its full internal cluster environment, so any external reproduction has to provide an equivalent but not identical infrastructure stack.

### 8. Optimization setup: objectives, optimizer, scheduler, batch size, precision, initialization, regularization, and stabilization

This aspect determines how the model learns during pretraining and post-training.

- What it needs: loss definitions, optimizer choice, LR schedule, batch size, precision mode, initialization details, clipping, regularization, and stabilization tricks.
- What NVIDIA opened: Nemotron-4 discloses batch size, sequence length, and training phases; Nemotron 3 reports disclose major precision and optimization choices at a higher level. [2][4][5]
- Is it sufficient: partially.
- Why not: the public material gives the outline, but exact hyperparameters and small stability-critical details are still incomplete in the surveyed sources.

### 9. Continued pretraining or domain adaptation

This aspect covers additional large-scale training after the first pretraining stage.

- What it needs: CPT data mixture, objective, schedule, stopping rule, and checkpoint lineage.
- What NVIDIA opened: Nemotron-4 explicitly states that it used 8T initial pretraining plus 1T continued pretraining, and Llama-Nemotron CPT derivatives are public. [4][8]
- Is it sufficient: partially.
- Why not: the existence of CPT is public, but the exact data shards and full CPT recipe are not always released.

### 10. Supervised fine-tuning

This aspect teaches the model to follow instructions and preferred response formats.

- What it needs: prompt-response data, formatting templates, training objective, curriculum, and checkpoint selection criteria.
- What NVIDIA opened: post-training datasets and open alignment or training repositories are available, including Nemotron 3 SFT data and Nemotron-4 instruct artifacts. [2][6][7][14]
- Is it sufficient: close, but still partial.
- Why not: available datasets and tooling make reproduction plausible, but exact prompt templating and run-selection details still need reconstruction.

### 11. Preference tuning, reward modeling, RLHF/RLAIF, or other post-training methods

This aspect improves behavior beyond plain SFT by using preferences, rewards, or reinforcement learning.

- What it needs: preference data or reward data, reward model recipe, rollout environments, sampling policy, optimization algorithm, and filtering logic.
- What NVIDIA opened: NeMo-Aligner, NeMo RL, NeMo Gym, reward-related models, and Nemotron RL datasets. [1][2][6][14][15]
- Is it sufficient: partially.
- Why not: the ingredients are increasingly open, especially for Nemotron 3, but exact environment mixes, reward shaping, and rollout settings remain generation-specific and not fully public.

### 12. Safety tuning, alignment, refusals, policy shaping, and red teaming

This aspect constrains the model to behave safely and within product policy.

- What it needs: safety datasets, refusal criteria, policy taxonomy, adversarial evaluation, and balancing between helpfulness and strictness.
- What NVIDIA opened: Nemotron safety datasets and open alignment tooling. [1][6][14][15]
- Is it sufficient: partially.
- Why not: the public materials support safety training, but exact internal policies, weightings, and red-team procedures are not fully disclosed.

### 13. Evaluation: pretraining metrics, downstream benchmarks, reasoning, coding, safety, robustness, and long-context evaluation

This aspect determines whether the model is good enough to release.

- What it needs: benchmark suites, evaluation prompts, scoring logic, judge setup if used, contamination controls, and regression tracking.
- What NVIDIA opened: model cards report evaluation results and NeMo Evaluator is public. [1][2][4][16]
- Is it sufficient: mostly for external benchmarking, not fully for exact reproduction.
- Why not: benchmark names and results are public, but exact harness versions, judge prompts, and contamination procedures are not always fully pinned down.

### 14. Inference-time optimization, serving assumptions, quantization, and deployment constraints

This aspect determines how the released model is run efficiently after training.

- What it needs: quantization recipe, serving backend, kernel choices, parallel serving topology, memory assumptions, and latency-throughput targets.
- What NVIDIA opened: quantized checkpoints, TensorRT-LLM support, and deployment cookbooks through the Nemotron pages and repo. [1][2][3]
- Is it sufficient: mostly for deployment, not fully for exact reproduction.
- Why not: an external team can serve the models, but matching NVIDIA's exact runtime behavior still depends on deployment-specific engineering choices.

### 15. Reproducibility, experiment tracking, ablations, and monitoring

This aspect makes the entire pipeline auditable and repeatable.

- What it needs: exact configs, seeds, run metadata, experiment logs, checkpoint lineage, and ablation records.
- What NVIDIA opened: reports, the Nemotron repository, and NeMo Evaluator improve reproducibility. [2][3][16]
- Is it sufficient: no.
- Why not: external teams do not get NVIDIA's full internal experiment database, failed-run history, or complete ablation trail, which matters when trying to match a release exactly.

## Practical Conclusion

If the goal is to reproduce a general LLM in the Nemotron line as closely as possible, target `Nemotron 3`, not `Nemotron-4` and not `Llama-Nemotron`.

`Nemotron 3` is the first release where NVIDIA publicly presents the combination that reproduction actually needs: weights, datasets, reports, and training recipes. `Nemotron-4` released important components, but not the same level of end-to-end openness. `Llama-Nemotron` is valuable for post-training study, but it inherits a closed dependency on the parent Llama base.

## References

1. [NVIDIA Nemotron developer page](https://developer.nvidia.com/nemotron)
2. [NVIDIA Nemotron 3 family page](https://research.nvidia.com/labs/nemotron/Nemotron-3/)
3. [NVIDIA-NeMo/Nemotron](https://github.com/NVIDIA-NeMo/Nemotron)
4. [nvidia/Nemotron-4-340B-Base model card](https://huggingface.co/nvidia/Nemotron-4-340B-Base)
5. [Nemotron-4 340B technical report page](https://research.nvidia.com/publication/2024-06_nemotron-4-340b)
6. [NVIDIA/NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner)
7. [Llama-Nemotron post-training dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset)
8. [Llama-3.1-Nemotron-Ultra-253B-CPT-v1 model card](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-CPT-v1)
9. [Nemotron pretraining dataset sample](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample)
10. [Nemotron-CC announcement](https://developer.nvidia.com/blog/announcing-nemotron-cc-a-trillion-token-english-language-dataset-for-llm-pretraining/)
11. [NVIDIA-NeMo/Curator](https://github.com/NVIDIA-NeMo/Curator)
12. [Building Nemotron-CC with NeMo Curator](https://developer.nvidia.com/blog/building-nemotron-cc-a-high-quality-trillion-token-dataset-for-llm-pretraining-from-common-crawl-using-nvidia-nemo-curator/)
13. [NVIDIA-NeMo/NeMo](https://github.com/NVIDIA-NeMo/NeMo)
14. [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL)
15. [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym)
16. [NeMo Evaluator](https://developer.nvidia.com/nemo-evaluator)
17. [NVIDIA blog: Nemotron-4 synthetic data generation](https://blogs.nvidia.com/blog/nemotron-4-synthetic-data-generation-llm-training/)
18. [Nemotron-H family page](https://research.nvidia.com/labs/adlr/nemotronh/)
19. [NVIDIA blog: Llama Nemotron reasoning models](https://developer.nvidia.com/blog/build-enterprise-ai-agents-with-advanced-open-nvidia-llama-nemotron-reasoning-models/)
20. [NVIDIA Nemotron Nano 2 family page](https://research.nvidia.com/labs/adlr/NVIDIA-Nemotron-Nano-2/)
21. [Nemotron-Cascade family page](https://research.nvidia.com/labs/nemotron/nemotron-cascade/)
22. [Nemotron-Cascade 2 family page](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/)
23. [NVIDIA Nemotron 3 Super page](https://research.nvidia.com/labs/nemotron/Nemotron-3-Super/)
