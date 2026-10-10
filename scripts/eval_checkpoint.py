"""Run a fixed-seed, headless rollout of one rsl-rl checkpoint.

Example::

    uv run scripts/eval_checkpoint.py Mjlab-Velocity-Flat-MicroDuck \
      --checkpoint-file logs/rsl_rl/run/model_3000.pt \
      --output eval/walk.json --num-envs 64 --steps 500 --seed 0 \
      --action-scale 0.9
"""

from mjlab_microduck.evaluate import main

if __name__ == "__main__":
    main()
