# HF Jobs training

Train mjlab-microduck on Hugging Face managed GPUs. Authentication uses the
cached Hub token (`hf auth login`) or `HF_TOKEN`; the submitter calls the
`huggingface_hub` Python API directly.

## One-time setup

```bash
hf auth login

# Optional: only needed when keeping the task's default W&B logger.
wandb login
```

W&B is optional. Without it, pass both `--no-wandb` (do not forward a secret)
and `--agent.logger tensorboard` (do not start a W&B-backed training run).

## Submit a run

Add `--hf-jobs` to an otherwise normal training command:

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.max_iterations 4000 \
  --hf-jobs
```

The prompt selects the personal or organization namespace that owns the job,
private source/checkpoint dataset, and billing. By default, each launch writes
below a unique `training/runs/<run-name>` path in that dataset, so training
never creates a model repository. Use `--namespace <name>` to skip the prompt;
non-interactive runs default to the personal namespace. Without `--hf-jobs`,
training remains local.

Useful submission flags:

- `--namespace <name>` — select the account or organization.
- `--flavor l4x1` — choose hardware (`l4x1` is the default).
- `--timeout 12h` — set the job time limit.
- `--detach` — return after submission instead of streaming logs.
- `--dry-run` — build the source snapshot and print the job spec only.
- `--run-name <tag>` — choose the readable checkpoint-path name.
- `--hf-resume-checkpoint <path>` — resume from a checkpoint in the mounted
  private source dataset.
- `--hf-bc-expert-checkpoint <path>` / `--hf-bc-anchor-checkpoint <path>` —
  use licensed behavior-cloning teachers from that same dataset instead of
  resolving the task defaults through W&B.
- `--no-wandb` — do not require or forward a W&B API key; also select a
  non-W&B logger in the training arguments.
- `--uv-cache` — opt in to a persistent `uv` cache bucket. It is off by
  default because the bucket's FUSE/copy path can be slower than downloading
  wheels on each job. `--uv-cache-bucket` overrides its location.

## Fine-tune an existing checkpoint

Put each `.pt` seed in the source dataset, then pass its path relative to the
dataset's `/src` mount:

```bash
hf upload BWJ2310/mjlab-microduck-src ./walk.pt checkpoints/walk.pt \
  --type dataset

uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.max_iterations 1500 \
  --agent.logger tensorboard \
  --hf-resume-checkpoint checkpoints/walk.pt \
  --namespace BWJ2310 \
  --run-name xc330-walk \
  --no-wandb --detach --hf-jobs
```

`--hf-resume-checkpoint` copies `/src/<path>` to the registered task's local
log tree and injects the matching `--agent.resume`, `--agent.load-run`, and
`--agent.load-checkpoint` options. Do not also pass those options or a W&B
checkpoint source. This restores the complete RSL-RL state: actor, critic, PPO
optimizer, normalizer, iteration, and curriculum counters. Task-specific
auxiliary optimizers are separate state and must document their own resume
contract. Use one private training-artifacts dataset. Pass
the previous output checkpoint to `--hf-resume-checkpoint` when continuing;
every launch writes to a new generated per-run path so it cannot overwrite an
earlier run. Intermediate checkpoints are training artifacts and never create
model repositories. The source/checkpoint dataset is created on the first run.
Only the final, validated ten-policy bundle belongs in the single public model
repository
[`BWJ2310/bruzmicroduck-xc330-custom-weight-policies`](https://huggingface.co/BWJ2310/bruzmicroduck-xc330-custom-weight-policies).
Cross-task transfer is a warm start instead; prefix the command with
`MICRODUCK_WARM_START=1` only when that is intentional.

## Outputs and job management

The submitter snapshots tracked and nonignored untracked worktree files into
the private source dataset. The job runs `uv sync`, trains, and uploads new
`model_*.pt` checkpoints plus dumped configs every 60 seconds and once more on
exit. It does not auto-export the newest checkpoint: evaluate and select a
checkpoint first, then export that selection explicitly for the final bundle.

The job ID, job URL, and exact private dataset checkpoint path are printed at
submission. Manage detached jobs with:

```bash
hf jobs inspect <job-id> --namespace <name>
hf jobs logs <job-id> --namespace <name> --follow
hf jobs cancel <job-id> --namespace <name>
```
