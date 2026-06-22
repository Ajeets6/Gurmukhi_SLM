import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import shlex
    import subprocess
    import sys
    from pathlib import Path

    import marimo as mo

    return Path, mo, os, shlex, subprocess, sys


@app.cell
def _(mo):
    mo.md(
        """
        # Gurmukhi SLM Training Runner

        This notebook wraps `Gur_slm.py` for cloud notebook environments.

        Marimo notebooks do not use IPython `!python` syntax reliably, so this
        notebook runs commands through Python `subprocess`.
        """
    )
    return


@app.cell
def _(Path):
    PROJECT_ROOT = Path.cwd()
    TRAIN_SCRIPT = PROJECT_ROOT / "Gur_slm.py"
    DATA_PATH = PROJECT_ROOT / "datasets" / "cleaned.tsv"
    TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "hf_bpe24k_tokenizer.json"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    return CHECKPOINT_DIR, DATA_PATH, PROJECT_ROOT, TOKENIZER_PATH, TRAIN_SCRIPT


@app.cell
def _(DATA_PATH, TOKENIZER_PATH, TRAIN_SCRIPT, mo):
    path_status = {
        "Gur_slm.py": TRAIN_SCRIPT.exists(),
        "datasets/cleaned.tsv": DATA_PATH.exists(),
        "tokenizer/hf_bpe24k_tokenizer.json": TOKENIZER_PATH.exists(),
    }

    mo.md(
        "\n".join(
            [
                "## Required Files",
                "",
                *[
                    f"- `{name}`: {'found' if exists else 'missing'}"
                    for name, exists in path_status.items()
                ],
            ]
        )
    )
    return (path_status,)


@app.cell
def _(subprocess, sys):
    def run_command(command: list[str], env: dict[str, str] | None = None) -> int:
        print("$ " + " ".join(command))
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")

        return process.wait()


    def python_command(*args: str) -> list[str]:
        return [sys.executable, *args]


    return python_command, run_command


@app.cell
def _(mo):
    mo.md("## Environment Check")
    return


@app.cell
def _(python_command, run_command):
    env_check_status = run_command(
        python_command(
            "-c",
            (
                "import torch, pandas, tokenizers; "
                "print('torch', torch.__version__); "
                "print('cuda_available', torch.cuda.is_available()); "
                "print('cuda_device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
            ),
        )
    )
    env_check_status
    return


@app.cell
def _(mo):
    mo.md("## Training Controls")
    return


@app.cell
def _(mo):
    profile = mo.ui.dropdown(
        options=["tiny", "small", "base", "large"],
        value="base",
        label="Model profile",
    )
    batch_size = mo.ui.number(value=64, start=1, stop=256, step=1, label="Batch size")
    epochs = mo.ui.number(value=5, start=1, stop=100, step=1, label="Epochs")
    grad_accum_steps = mo.ui.number(value=1, start=1, stop=64, step=1, label="Grad accumulation")
    num_workers = mo.ui.number(value=4, start=0, stop=32, step=1, label="DataLoader workers")
    max_rows = mo.ui.number(value=0, start=0, stop=2_000_000, step=1024, label="Max rows; 0 = full corpus")
    max_steps = mo.ui.number(value=0, start=0, stop=1_000_000, step=100, label="Max steps/epoch; 0 = full epoch")
    amp = mo.ui.dropdown(options=["bf16", "fp16", "none"], value="bf16", label="AMP")

    mo.vstack(
        [
            mo.hstack([profile, amp]),
            mo.hstack([batch_size, epochs, grad_accum_steps, num_workers]),
            mo.hstack([max_rows, max_steps]),
        ]
    )
    return amp, batch_size, epochs, grad_accum_steps, max_rows, max_steps, num_workers, profile


@app.cell
def _(
    amp,
    batch_size,
    epochs,
    grad_accum_steps,
    max_rows,
    max_steps,
    num_workers,
    profile,
):
    def optional_int_flag(name: str, value: int) -> list[str]:
        return [name, str(value)] if value and value > 0 else []

    selected_train_command = [
        "Gur_slm.py",
        "--train",
        "--device",
        "cuda",
        "--profile",
        profile.value,
        "--batch-size",
        str(batch_size.value),
        "--epochs",
        str(epochs.value),
        "--grad-accum-steps",
        str(grad_accum_steps.value),
        "--num-workers",
        str(num_workers.value),
        "--amp",
        amp.value,
        "--output-dir",
        f"checkpoints/{profile.value}_full",
        *optional_int_flag("--max-rows", int(max_rows.value)),
        *optional_int_flag("--max-steps", int(max_steps.value)),
    ]

    selected_train_command
    return (selected_train_command,)


@app.cell
def _(mo, selected_train_command):
    mo.md(
        "Selected training command:\n\n"
        + "```bash\npython "
        + " ".join(selected_train_command)
        + "\n```"
    )
    return


@app.cell
def _(mo):
    dry_run_button = mo.ui.run_button(label="Run dry run")
    smoke_train_button = mo.ui.run_button(label="Run smoke train")
    full_train_button = mo.ui.run_button(label="Run selected full training command")

    mo.hstack([dry_run_button, smoke_train_button, full_train_button])
    return dry_run_button, full_train_button, smoke_train_button


@app.cell
def _(dry_run_button, python_command, run_command):
    if dry_run_button.value:
        dry_run_status = run_command(
            python_command(
                "Gur_slm.py",
                "--dry-run",
                "--device",
                "cuda",
                "--profile",
                "small",
                "--batch-size",
                "4",
                "--dry-run-rows",
                "128",
                "--amp",
                "bf16",
            )
        )
    else:
        dry_run_status = None

    dry_run_status
    return


@app.cell
def _(python_command, run_command, smoke_train_button):
    if smoke_train_button.value:
        smoke_train_status = run_command(
            python_command(
                "Gur_slm.py",
                "--train",
                "--device",
                "cuda",
                "--profile",
                "small",
                "--batch-size",
                "4",
                "--max-rows",
                "4096",
                "--max-steps",
                "500",
                "--epochs",
                "1",
                "--amp",
                "bf16",
                "--output-dir",
                "checkpoints/smoke_cloud",
                "--log-every",
                "25",
            )
        )
    else:
        smoke_train_status = None

    smoke_train_status
    return


@app.cell
def _(full_train_button, python_command, run_command, selected_train_command):
    if full_train_button.value:
        full_train_status = run_command(python_command(*selected_train_command))
    else:
        full_train_status = None

    full_train_status
    return


@app.cell
def _(CHECKPOINT_DIR, mo):
    mo.md("## Checkpoints")
    checkpoint_files = sorted(CHECKPOINT_DIR.glob("**/*.pt")) if CHECKPOINT_DIR.exists() else []
    checkpoint_files[-20:]
    return


if __name__ == "__main__":
    app.run()
