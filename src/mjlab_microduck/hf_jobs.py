"""Submit a mjlab-microduck training run as a Hugging Face Job.

Invoked by the --hf-jobs interception (see train_hook.py):

    uv run train Mjlab-Velocity-Flat-MicroDuck \
        --env.scene.num-envs 4096 --agent.max_iterations 4000 --hf-jobs

Anything that isn't an --hf-* / submission flag is forwarded verbatim to
`uv run train` inside the job.

Auth: the cached HF token from `hf auth login` / HF_TOKEN env. The account's
orgs are listed at submission and you pick the namespace to run under
(personal or org) — repos, the optional uv-cache bucket and the job itself
all live in that namespace. Pass --namespace to skip the prompt (automation).

Everything goes through the huggingface_hub Python API (Jobs API, hub >= 1.x)
— the standalone `hf` CLI is NOT required.

Source: a snapshot of tracked files (committed or not) is uploaded to a
private HF dataset repo and mounted read-only inside the job. Checkpoints are
pushed by a watcher running alongside training (scripts/hf/uploader.py), under
a per-launch path in that same dataset by default. Training artifacts never
create model repositories; the only model repository is the validated bundle.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from netrc import netrc
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, Volume, get_token

DEFAULT_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
DEFAULT_FLAVOR = "l4x1"
DEFAULT_TIMEOUT = "12h"

_RESUME_RUN_DIR = "hf_seed"
# Deliberately does not match model_*.pt: the checkpoint uploader must only see
# checkpoints produced by this job, not its seed.
_RESUME_FILENAME = "model.pt"
_RESUME_CONFLICT_OPTIONS = (
    "--agent.resume",
    "--agent.load-run",
    "--agent.load-checkpoint",
    "--wandb-run-path",
    "--wandb-checkpoint-name",
)
_RESUME_TRAIN_ARGS = (
    "--agent.resume",
    "True",
    "--agent.load-run",
    rf"^{_RESUME_RUN_DIR}$",
    "--agent.load-checkpoint",
    r"^model\.pt$",
)
_BC_EXPERT_ENV = "MICRODUCK_BC_EXPERT_CHECKPOINT"
_BC_ANCHOR_ENV = "MICRODUCK_BC_ANCHOR_CHECKPOINT"

# Bootstrap script run inside the container. `$VAR` is expanded by the
# container shell from the job's env vars.
BOOTSTRAP = r"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -qq -y --no-install-recommends git curl ca-certificates xz-utils >/dev/null
# Pinned uv: the cache bucket persists across jobs, and a floating "latest" uv
# reading entries written by an older uv corrupts installs (seen 2026-07-21:
# bam's built-wheel cache entry from a 0.9.x-era job made 0.11.30 fail with
# "The wheel is invalid: Missing .dist-info directory").
curl -LsSf https://astral.sh/uv/0.11.30/install.sh | sh >/dev/null
export PATH="/root/.local/bin:$PATH"
# HF Jobs met le cache uv et /work sur des FS différents -> uv ne peut pas
# hardlink et son fallback corrompt les wheels construits (bam -> "Missing
# .dist-info directory"). copy = install fiable (le remède que uv suggère).
export UV_LINK_MODE=copy

mkdir -p /work && cd /work
echo "[bootstrap] extracting source $SRC_TARBALL"
tar -xzf "/src/$SRC_TARBALL"

echo "[bootstrap] uv sync"
# Self-heal a poisoned persistent cache: a bad entry fails sync
# deterministically, so nuke the cache and rebuild it once before giving up.
uv sync --no-progress || {
    echo "[bootstrap] uv sync failed — cleaning uv cache and retrying"
    uv cache clean || true
    uv sync --no-progress
}

# TRAIN_ARGS is produced with shlex.quote(). Expanding that scalar directly
# would pass its quote characters literally, so rebuild the original argv.
mapfile -d '' -t TRAIN_ARGV < <(uv run python - <<'PY'
import os
import shlex
import sys
for arg in shlex.split(os.environ["TRAIN_ARGS"]):
    sys.stdout.buffer.write(arg.encode() + b"\0")
PY
)
if [ "${#TRAIN_ARGV[@]}" -eq 0 ]; then
    echo "[bootstrap] TRAIN_ARGS decoded to an empty argv" >&2
    exit 2
fi

# mjlab's train CLI treats --agent.load-checkpoint as a regex below
# logs/rsl_rl/<experiment>/<run>/; it does not accept an absolute checkpoint
# path. Stage the requested read-only dataset file in that layout before the
# uploader starts. model.pt intentionally does not match model_*.pt.
if [ -n "${HF_RESUME_CHECKPOINT:-}" ]; then
    uv run python - <<'PY'
from mjlab_microduck.hf_jobs import _stage_resume_checkpoint
_stage_resume_checkpoint()
PY
fi

echo "[bootstrap] launching checkpoint uploader"
mkdir -p logs/rsl_rl
nohup uv run python scripts/hf/uploader.py > /tmp/uploader.log 2>&1 &
UPLOADER_PID=$!

printf '[bootstrap] starting training: uv run train'
printf ' %q' "${TRAIN_ARGV[@]}"
printf '\n'
set +e
uv run train "${TRAIN_ARGV[@]}"
TRAIN_RC=$?
set -e

echo "[bootstrap] training exited with code $TRAIN_RC, final upload pass"
# Stop and reap the watcher before the synchronous pass so two Hub commits
# cannot race.  A completed Job must mean its final checkpoints reached the
# Hub; retry transient commit failures, then surface a persistent failure.
kill $UPLOADER_PID 2>/dev/null || true
wait $UPLOADER_PID 2>/dev/null || true
UPLOAD_RC=1
for attempt in 1 2 3; do
    if CKPT_ONE_SHOT=1 uv run python scripts/hf/uploader.py; then
        UPLOAD_RC=0
        break
    fi
    echo "[bootstrap] final upload failed (attempt $attempt/3)"
    [ "$attempt" -lt 3 ] && sleep 5
done
if [ "$UPLOAD_RC" -ne 0 ]; then
    echo "[bootstrap] final checkpoint upload failed permanently" >&2
    [ "$TRAIN_RC" -eq 0 ] && TRAIN_RC=1
fi

exit $TRAIN_RC
"""


def _option_present(argv: list[str], option: str) -> bool:
    """Return whether ``option`` occurs in split or ``--opt=value`` form."""
    return any(arg == option or arg.startswith(option + "=") for arg in argv)


def _option_value(argv: list[str], option: str) -> str | None:
    """Return the last value supplied for a CLI option, if any."""
    value: str | None = None
    for i, arg in enumerate(argv):
        if arg.startswith(option + "="):
            value = arg.split("=", 1)[1]
        elif arg == option:
            if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                raise ValueError(f"{option} requires a value")
            value = argv[i + 1]
    return value


def _safe_dataset_file(value: str, option: str) -> str:
    """Validate and normalize a file path relative to the /src mount."""
    path = PurePosixPath(value)
    if not value or value == "." or path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"{option} must be a relative file path below /src "
            "(for example checkpoints/walk.pt)"
        )
    return path.as_posix()


def _safe_run_path_component(value: str) -> str:
    """Turn a display-oriented run name into one readable Hub path component."""
    component = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not component:
        raise ValueError("--run-name must contain at least one letter or number")
    if len(component) > 96:
        digest = hashlib.sha256(value.encode()).hexdigest()[:12]
        component = f"{component[:83].rstrip('-')}-{digest}"
    return component


def _safe_experiment_name(value: str) -> str:
    """Validate an experiment path relative to the job's local log root."""
    path = PurePosixPath(value)
    if not value or value == "." or path.is_absolute() or ".." in path.parts:
        raise ValueError(
            "--agent.experiment-name must stay below logs/rsl_rl "
            "(absolute paths and '..' are not allowed with --hf-resume-checkpoint)"
        )
    return path.as_posix()


def _prepare_resume_training_args(
    checkpoint: str | None, train_args: list[str]
) -> tuple[str | None, list[str]]:
    """Validate a mounted seed and add the exact mjlab local-resume flags."""
    if checkpoint is None:
        return None, train_args

    checkpoint = _safe_dataset_file(checkpoint, "--hf-resume-checkpoint")
    conflicts = [
        option
        for option in _RESUME_CONFLICT_OPTIONS
        if _option_present(train_args, option)
    ]
    if conflicts:
        joined = ", ".join(conflicts)
        raise ValueError(
            f"--hf-resume-checkpoint owns the resume source; remove conflicting option(s): {joined}"
        )
    return checkpoint, [*train_args, *_RESUME_TRAIN_ARGS]


def _effective_experiment_name(train_argv: list[str]) -> str:
    """Resolve the log root used by mjlab for this exact train invocation."""
    if not train_argv:
        raise ValueError("TRAIN_ARGS does not contain a task id")

    explicit = _option_value(train_argv[1:], "--agent.experiment-name")
    if explicit is not None:
        return explicit

    # Importing mjlab loads task plugins, including this repository's tasks.
    import mjlab.tasks  # noqa: F401
    from mjlab.tasks.registry import load_rl_cfg

    return load_rl_cfg(train_argv[0]).experiment_name


def _stage_resume_checkpoint(
    *,
    checkpoint: str | None = None,
    train_args: str | None = None,
    dataset_root: Path = Path("/src"),
    work_root: Path = Path("."),
) -> Path:
    """Copy an HF-mounted checkpoint into mjlab's expected local log layout.

    ``checkpoint`` and ``train_args`` default to the environment injected into
    the Job. Explicit parameters keep the filesystem behavior easy to test.
    """
    raw_checkpoint = (
        os.environ.get("HF_RESUME_CHECKPOINT") if checkpoint is None else checkpoint
    )
    if raw_checkpoint is None:
        raise ValueError("HF_RESUME_CHECKPOINT is not set")
    relative = _safe_dataset_file(raw_checkpoint, "--hf-resume-checkpoint")

    raw_train_args = os.environ.get("TRAIN_ARGS") if train_args is None else train_args
    if not raw_train_args:
        raise ValueError("TRAIN_ARGS is not set")
    train_argv = shlex.split(raw_train_args)
    experiment_name = _safe_experiment_name(_effective_experiment_name(train_argv))

    dataset_root = dataset_root.resolve()
    source = (dataset_root / relative).resolve()
    try:
        source.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(
            f"resume checkpoint resolves outside {dataset_root}: {relative}"
        ) from exc
    if not source.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {source}")

    log_root = (work_root / "logs" / "rsl_rl").resolve()
    destination = (
        log_root / experiment_name / _RESUME_RUN_DIR / _RESUME_FILENAME
    ).resolve()
    try:
        destination.relative_to(log_root)
    except ValueError as exc:  # Defensive if an existing log component is a symlink.
        raise ValueError(
            f"resume staging destination resolves outside {log_root}: {experiment_name}"
        ) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o644)
    print(f"[bootstrap] staged resume checkpoint {source} -> {destination}", flush=True)
    return destination


def _wandb_api_key() -> str | None:
    """Best-effort lookup of the user's wandb API key.

    Order: WANDB_API_KEY env -> ~/.netrc (machine api.wandb.ai).
    """
    if k := os.environ.get("WANDB_API_KEY"):
        return k
    try:
        n = netrc(str(Path.home() / ".netrc"))
        auth = n.authenticators("api.wandb.ai")
        if auth and auth[2]:
            return auth[2]
    except (FileNotFoundError, OSError):
        pass
    return None


def _repo_root() -> Path:
    """Repo root of the CURRENT directory — worktree-aware.

    Resolving from cwd ensures a run from a worktree snapshots that worktree.
    """
    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
    return Path(out.decode().strip())


def _build_tarball(repo_root: Path, out_path: Path) -> str:
    """Create a tarball of HEAD + uncommitted tracked changes. Returns short SHA."""
    sha = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root)
        .decode()
        .strip()
    )

    # Use `git ls-files` so we include tracked-but-modified files (working tree
    # state) but skip ignored junk (.venv, logs, *.onnx, wandb/, etc.).
    files = (
        subprocess.check_output(
            ["git", "ls-files", "-co", "--exclude-standard"], cwd=repo_root
        )
        .decode()
        .splitlines()
    )

    with tarfile.open(out_path, "w:gz") as tar:
        for rel in files:
            p = repo_root / rel
            if p.exists() and p.is_file():
                tar.add(p, arcname=rel)
    return sha


def _pick_namespace(api: HfApi, preset: str | None) -> str:
    """Choose the namespace (personal account or org) the job runs under.

    Interactive prompt unless --namespace was given or there is nothing to
    choose. Non-tty (scripts, CI) falls back to the personal account.
    """
    info = api.whoami()
    user = info.get("name") or info.get("email")
    if not user:
        raise RuntimeError(
            "Could not determine HF username. Run `hf auth login` first."
        )
    orgs = [o["name"] for o in info.get("orgs", []) if o.get("name")]

    if preset is not None:
        if preset not in (user, *orgs):
            raise RuntimeError(
                f"--namespace {preset!r} is neither your account ({user}) "
                f"nor one of your orgs ({', '.join(orgs) or 'none'})."
            )
        return preset

    if not orgs:
        return user

    if not sys.stdin.isatty():
        print(f"[hf] non-interactive, defaulting to personal namespace: {user}")
        return user

    choices = [user, *orgs]
    print("[hf] run under which namespace?")
    print(f"  1) {user} (personal)")
    for i, org in enumerate(orgs, start=2):
        print(f"  {i}) {org}")
    while True:
        raw = input(f"Choice [1-{len(choices)}, default 1]: ").strip()
        if raw == "":
            return user
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        if raw in choices:
            return raw
        print(f"  invalid choice: {raw!r}")


def _await_scheduling(
    api: HfApi, job_id: str, namespace: str, budget_s: float = 1200.0
) -> tuple[str, str | None]:
    """Poll until the job leaves SCHEDULING (image pull / queue / mounts).

    This phase legitimately takes minutes (the pytorch image pull alone is
    ~5 min on a cold GPU node) and is also where volume-mount failures
    surface, ~7 min in ("init container exhausted retries"). Returns
    (stage, message) at the first non-SCHEDULING stage, or the last observed
    one when the budget runs out (a long queue is not an error — the caller's
    streaming loop keeps supervising).
    """
    deadline = time.monotonic() + budget_s
    last_note = time.monotonic()
    stage, message = "", None
    while time.monotonic() < deadline:
        status = api.inspect_job(job_id=job_id, namespace=namespace).status
        stage, message = status.stage, status.message
        if stage and stage != "SCHEDULING":
            return stage, message
        if time.monotonic() - last_note > 60:
            print("[job] still scheduling (queue / image pull / volume mounts)...")
            last_note = time.monotonic()
        time.sleep(10)
    return stage, message


def submit(argv: list[str]) -> int:
    """Parse submission args from ``argv`` and launch the HF job."""
    ap = argparse.ArgumentParser(
        prog="train --hf-jobs",
        description="Submit a microduck training run to HF Jobs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("task", help="mjlab task id, e.g. Mjlab-Velocity-Flat-MicroDuck")
    ap.add_argument("--flavor", default=DEFAULT_FLAVOR, help="HF Jobs hardware flavor")
    ap.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image to run in")
    ap.add_argument("--timeout", default=DEFAULT_TIMEOUT, help="Job max duration")
    ap.add_argument(
        "--namespace",
        default=None,
        help="HF namespace (your username or an org) to run under; skips the prompt.",
    )
    ap.add_argument(
        "--run-name",
        default=None,
        help="Short tag for this run; defaults to task+timestamp",
    )
    ap.add_argument(
        "--detach",
        action="store_true",
        help="Submit and return immediately (do not stream logs).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Build tarball and print the job spec without submitting.",
    )
    ap.add_argument(
        "--src-repo",
        default=None,
        help="HF dataset repo for source tarballs. Defaults to <namespace>/mjlab-microduck-src",
    )
    ap.add_argument(
        "--hf-resume-checkpoint",
        default=None,
        metavar="PATH",
        help=(
            "Resume from PATH in the source dataset mounted at /src "
            "(for example checkpoints/walk.pt). Injects the mjlab resume flags."
        ),
    )
    ap.add_argument(
        "--hf-bc-expert-checkpoint",
        default=None,
        metavar="PATH",
        help=(
            "Use PATH from the /src source dataset as the fallen-state behavior-"
            "cloning expert instead of downloading the configured wandb checkpoint."
        ),
    )
    ap.add_argument(
        "--hf-bc-anchor-checkpoint",
        default=None,
        metavar="PATH",
        help=(
            "Use PATH from the /src source dataset as the upright behavior-cloning "
            "anchor instead of downloading the configured wandb checkpoint."
        ),
    )
    ap.add_argument(
        "--uv-cache-bucket",
        default=None,
        help="HF bucket used as UV_CACHE_DIR to persist wheels across runs. "
        "Defaults to <namespace>/mjlab-uv-cache. Requires --uv-cache.",
    )
    ap.add_argument(
        "--uv-cache",
        action="store_true",
        help="Mount a persistent uv cache bucket (OFF by default: the FUSE "
        "bucket mount does not support hardlinks, so `uv sync` falls back "
        "to full-copying ~6 GB of unpacked packages through the network "
        "mount — far slower than just re-downloading wheels from PyPI, "
        "which HF's datacenter bandwidth handles in ~1 min; it also "
        "poisons across uv versions: a 0.9.x-era entry made 0.11.30 die "
        "with 'wheel is invalid: Missing .dist-info', 2026-07-21).",
    )
    ap.add_argument(
        "--no-wandb",
        action="store_true",
        help="Do not forward a wandb API key (training will fail if wandb is enabled).",
    )
    args, train_args = ap.parse_known_args(argv)

    try:
        resume_checkpoint, train_args = _prepare_resume_training_args(
            args.hf_resume_checkpoint, train_args
        )
        bc_expert_checkpoint = (
            _safe_dataset_file(
                args.hf_bc_expert_checkpoint, "--hf-bc-expert-checkpoint"
            )
            if args.hf_bc_expert_checkpoint is not None
            else None
        )
        bc_anchor_checkpoint = (
            _safe_dataset_file(
                args.hf_bc_anchor_checkpoint, "--hf-bc-anchor-checkpoint"
            )
            if args.hf_bc_anchor_checkpoint is not None
            else None
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    api = HfApi()
    try:
        namespace = _pick_namespace(api, args.namespace)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"[hf] namespace: {namespace}")

    token = get_token()
    if not token:
        print("error: no cached HF token. Run `hf auth login`.", file=sys.stderr)
        return 1

    repo_root = _repo_root()
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    unique_stamp = f"{stamp}-{uuid.uuid4().hex[:8]}"
    src_repo = args.src_repo or f"{namespace}/mjlab-microduck-src"
    try:
        checkpoint_path_prefix = "training/runs/" + _safe_run_path_component(
            f"{args.run_name or args.task}-{unique_stamp}"
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    env: dict[str, str] = {
        # Disarms the --hf-jobs interception inside the job, so the job's own
        # `uv run train` can only ever train locally (see train_hook.py).
        "MICRODUCK_IN_HF_JOB": "1",
        "CKPT_REPO": src_repo,
        "CKPT_PATH_PREFIX": checkpoint_path_prefix,
        "TRAIN_ARGS": " ".join(shlex.quote(a) for a in [args.task, *train_args]),
    }
    if resume_checkpoint is not None:
        env["HF_RESUME_CHECKPOINT"] = resume_checkpoint
    if bc_expert_checkpoint is not None:
        env[_BC_EXPERT_ENV] = f"/src/{bc_expert_checkpoint}"
    if bc_anchor_checkpoint is not None:
        env[_BC_ANCHOR_ENV] = f"/src/{bc_anchor_checkpoint}"
    # Warm start (tasks/mdp.py Patch 5): curricula restart at 0 after the
    # checkpoint load. Forwarded so `MICRODUCK_WARM_START=1 uv run train ... --hf-jobs`
    # behaves like the local run.
    if os.environ.get("MICRODUCK_WARM_START"):
        env["MICRODUCK_WARM_START"] = os.environ["MICRODUCK_WARM_START"]
    secrets: dict[str, str] = {"HF_TOKEN": token}

    # Forward wandb credentials (env var, then ~/.netrc)
    if not args.no_wandb:
        wb_key = _wandb_api_key()
        if not wb_key:
            print(
                "[wandb] ✗ no API key found (checked $WANDB_API_KEY and ~/.netrc).\n"
                "        Run `wandb login` locally, or pass --no-wandb to skip.",
                file=sys.stderr,
            )
            return 1
        secrets["WANDB_API_KEY"] = wb_key
        src = "env" if os.environ.get("WANDB_API_KEY") else "~/.netrc"
        print(f"[wandb] forwarding API key from {src}")
        for k in ("WANDB_PROJECT", "WANDB_ENTITY"):
            if os.environ.get(k):
                env[k] = os.environ[k]

    volumes = [
        Volume(type="dataset", source=src_repo, mount_path="/src", read_only=True)
    ]

    # Persistent uv cache — opt-in via --uv-cache (see the flag's help text:
    # cross-filesystem installs from the FUSE bucket are slower than fresh
    # PyPI downloads, and a stale entry deterministically killed uv sync on
    # 2026-07-21).
    cache_bucket: str | None = None
    if args.uv_cache:
        cache_bucket = args.uv_cache_bucket or f"{namespace}/mjlab-uv-cache"
        volumes.append(
            Volume(type="bucket", source=cache_bucket, mount_path="/uv-cache")
        )
        env["UV_CACHE_DIR"] = "/uv-cache"
        print(f"[uv-cache] using bucket {cache_bucket}")

    # 1. Build tarball
    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / f"src-{unique_stamp}.tar.gz"
        print(f"[src] building tarball -> {tar_path.name} (from {repo_root})")
        sha = _build_tarball(repo_root, tar_path)
        size_mb = tar_path.stat().st_size / 1e6
        print(f"[src] HEAD={sha}, {size_mb:.1f} MB")
        env["SRC_TARBALL"] = tar_path.name
        env["GIT_SHA"] = sha

        if args.dry_run:
            print("[dry-run] would submit job:")
            print(f"  namespace: {namespace}")
            print(f"  image:     {args.image}")
            print(f"  flavor:    {args.flavor}, timeout: {args.timeout}")
            print(
                f"  volumes:   {[f'{v.type}:{v.source} -> {v.mount_path}' for v in volumes]}"
            )
            print(f"  env:       { {k: v for k, v in env.items()} }")
            print(f"  secrets:   { {k: '***' for k in secrets} }")
            print(
                "  checkpoints: "
                f"https://huggingface.co/datasets/{src_repo}/tree/main/"
                f"{checkpoint_path_prefix}"
            )
            return 0

        # 2. Upload tarball + pre-create repos/bucket
        api.create_repo(src_repo, repo_type="dataset", private=True, exist_ok=True)
        print(f"[src] uploading to dataset {src_repo}")
        api.upload_file(
            path_or_fileobj=str(tar_path),
            path_in_repo=tar_path.name,
            repo_id=src_repo,
            repo_type="dataset",
        )
        if cache_bucket is not None:
            api.create_bucket(cache_bucket, private=True, exist_ok=True)

        # 3. Submit. GPU-node volume mounts fail transiently ("init container
        # exhausted retries", surfacing ~7 min into SCHEDULING — observed
        # 2026-07 while an identical probe job mounted fine at the same time).
        # Supervise the scheduling phase and resubmit on a mount failure.
        checkpoint_destination = (
            f"https://huggingface.co/datasets/{src_repo}/tree/main/"
            f"{checkpoint_path_prefix}"
        )
        print(f"[ckpt] checkpoints -> {checkpoint_destination}")
        print(
            f"[job] submitting (namespace={namespace}, flavor={args.flavor}, timeout={args.timeout})"
        )
        job = None
        stage, message = "", None
        for attempt in range(3):
            if attempt:
                print(
                    f"[job] ✗ volume mount failed (flaky node) — resubmitting ({attempt + 1}/3)"
                )
                time.sleep(10)
            try:
                job = api.run_job(
                    image=args.image,
                    command=["bash", "-c", BOOTSTRAP],
                    env=env,
                    secrets=secrets,
                    flavor=args.flavor,
                    timeout=args.timeout,
                    volumes=volumes,
                    namespace=namespace,
                )
            except Exception as e:
                msg = str(e)
                if "402" in msg or "Payment Required" in msg or "credit" in msg.lower():
                    print(
                        "\n[job] ✗ Hugging Face Jobs billing error.\n"
                        f"    The namespace {namespace!r} has insufficient Jobs credits.\n"
                        "    → Add credits:   https://huggingface.co/settings/billing\n"
                        "    → Or get HF Pro: https://huggingface.co/settings/billing/subscription",
                        file=sys.stderr,
                    )
                    return 2
                if (
                    "403" in msg
                    or "Forbidden" in msg
                    or "required permissions" in msg.lower()
                ):
                    print(
                        "\n[job] ✗ Hugging Face Jobs permission error (403).\n"
                        "    Your HF token authenticates fine but is NOT allowed to use the Jobs API\n"
                        f"    for namespace {namespace!r}. This is a token-scope problem, not billing.\n"
                        "    → Create/edit a fine-grained token WITH the Jobs permission enabled:\n"
                        "        https://huggingface.co/settings/tokens\n"
                        "      (fine-grained → under your user AND/OR the org, tick the 'Jobs' permissions),\n"
                        "      then re-login locally:  hf auth login\n"
                        '    → Verify:  python -c "from huggingface_hub import HfApi; "\n'
                        "               \"print(list(HfApi().list_jobs(namespace='<ns>')))\"  (must not 403)\n"
                        "    (If Jobs are enabled but still blocked, the namespace may also need an HF\n"
                        "     plan/credits that include Jobs — see the billing link above.)",
                        file=sys.stderr,
                    )
                    return 3
                raise
            print(f"[job] id:  {job.id}")
            if getattr(job, "url", None):
                print(f"[job] url: {job.url}")
            if args.detach:
                print(
                    "[job] --detach: not supervising startup — check the URL above; "
                    "transient 'Volume mount failed' errors need a manual resubmit."
                )
                return 0
            stage, message = _await_scheduling(api, job.id, namespace)
            if stage == "ERROR" and "mount" in (message or "").lower():
                continue  # flaky node — resubmit
            break

    assert job is not None
    if stage == "ERROR":
        print(f"[job] ✗ failed to start: {message}", file=sys.stderr)
        return 1

    # Supervise to completion: stream logs, re-attach if the stream drops
    # (it returns empty while the container is still starting), and report
    # the terminal status.
    print("[job] streaming logs (Ctrl-C detaches; the job keeps running)")
    try:
        while True:
            try:
                for line in api.fetch_job_logs(
                    job_id=job.id, namespace=namespace, follow=True
                ):
                    print(line)
            except Exception as e:
                print(f"[job] log stream dropped ({e}); re-attaching")
            status = api.inspect_job(job_id=job.id, namespace=namespace).status
            if status.stage == "COMPLETED":
                print("[job] ✓ completed")
                return 0
            if status.stage in ("ERROR", "DELETED", "CANCELED"):
                print(f"[job] ✗ {status.stage}: {status.message}", file=sys.stderr)
                return 1
            time.sleep(10)
    except KeyboardInterrupt:
        print(
            f"\n[job] detached. Job {job.id} is still running: {getattr(job, 'url', job.id)}"
        )
        return 0
