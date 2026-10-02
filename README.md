# Computation, Not Memory: Chain-of-Thought and State Tracking in LLM Agents

Code for the paper *Computation, Not Memory: Chain-of-Thought and State Tracking in LLM Agents* by Pedram Mouseli and Massieh Moayedi.

We study how an LLM agent's actions depend on its reasoning when the correct action depends on a changing state. The agent works in a small file-system environment: it must move every unlocked log file to an archive, while background events toggle file locks and announce which file changed but not its new value. We separate generating reasoning before each action from retaining earlier reasoning in the prompt, test whether the content of the reasoning matters, and compare linear probes and activation patching with the agent's actions.

## Installation

```
pip install -r requirements.txt
```

We used Python 3.10. Models are loaded from the Hugging Face Hub (`Qwen/Qwen3-8B`, `Qwen/Qwen3.5-9B`, `Qwen/Qwen3-8B-Base` and other Qwen3 sizes). `vllm` is needed only to generate rollouts.

## Usage

Generate rollouts for one condition. The mode is `think` (reasoning) or `terse` (direct action), the persistence of earlier replies in the prompt is `full`, `body` or `cmd`, and the format is `chat` or `plain`:

```
python -m cotstate.gen --model Qwen/Qwen3-8B --mode think --persist cmd --format chat \
    --n 500 --max-tokens 4000 --max-model-len 32768 --out data/think_cmd_chat/transcripts.jsonl
```

Count lock-violating commands in saved rollouts:

```
python -m cotstate.audit_behav think_cmd=data/think_cmd_chat --out results/violations.json --csv results/violations.csv
```

Each script documents its arguments with `--help`.

## Code

| Module | Purpose |
|---|---|
| `env.py` | Environment: files, locks, commands and background events |
| `gen.py` | Rollouts for one condition (vLLM) |
| `score_actions.py`, `audit_behav.py`, `rescore_all.py`, `gen_vs_persist.py` | Lock violations, and the generation and retention contrasts |
| `task_outcomes.py`, `goal_analysis.py`, `supp_stats.py`, `reasoning_stats.py`, `run_configs.py` | Task success and other per-run summaries |
| `fixed_history2.py`, `fixed_history_ctrl.py` | Fixed-history arms: no reasoning, own reasoning, state table, filler and other-history reasoning |
| `cf_action.py`, `cf_swap.py`, `cf_edit.py` | Counterfactual pairs, reasoning swap and targeted edit |
| `acts.py`, `probe.py`, `probe_vs_tfidf.py`, `within_condition.py`, `oracle.py`, `textbase.py` | Activations, linear probes and text baselines |
| `intervene.py`, `patching.py`, `patching2.py`, `patching_action.py`, `patching_matched.py` | Text corruption and activation patching |
| `steer_null.py`, `steer_null_stats.py` | Steering with a random-direction null |
| `stats.py`, `summary_stats.py` | Statistical tests with episode-level resampling |

## Citation

<!-- ```bibtex
@misc{mouseli2026computation,
  title  = {Computation, Not Memory: Chain-of-Thought and State Tracking in {LLM} Agents},
  author = {Mouseli, Pedram and Moayedi, Messiah},
  year   = {2026}
} -->
```
