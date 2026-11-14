"""download.sh and pretrain.sh, the two scripts of the pretraining tutorial: the commands they compose, what they
refuse, and that neither is tied to one cluster. Every external program is a stub on PATH that records its arguments."""

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def stubs(tmp_path):
    """A bin/ of stand-ins (hf, accelerate, nvidia-smi) that append their arguments to calls.log."""
    bin_dir, log = tmp_path / "bin", tmp_path / "calls.log"
    bin_dir.mkdir()
    for name in ("hf", "accelerate"):
        (bin_dir / name).write_text(f'#!/bin/bash\necho "{name} $*" >> "{log}"\n')
    (bin_dir / "nvidia-smi").write_text('#!/bin/bash\nfor i in 0 1 2 3; do echo "GPU $i: stub"; done\n')
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    return bin_dir, log


def run(script: str, args: list[str], env_extra: dict, stubs) -> subprocess.CompletedProcess:
    bin_dir, _ = stubs
    keep = {k: v for k, v in os.environ.items() if k in ("HOME", "TMPDIR", "LANG")}
    env = {**keep, "PATH": f"{bin_dir}:/usr/bin:/bin:/sbin:/usr/sbin", **env_extra}
    return subprocess.run(["bash", str(REPO / script), *args], capture_output=True, text=True, env=env)


def patched_checkout(tmp_path: Path) -> Path:
    """A folder that looks like a patched reve_eeg checkout to pretrain.sh (the marker patch 8 leaves)."""
    reve = tmp_path / "reve_eeg"
    (reve / "src" / "utils").mkdir(parents=True)
    (reve / "src" / "utils" / "data_loading.py").write_text("    return _zarr_train_loader(args)  # PATCH 8\n")
    return reve


def test_download_fetches_the_tagged_store_and_prints_the_path(tmp_path, stubs):
    r = run("download.sh", [str(tmp_path / "data")], {}, stubs)
    assert r.returncode == 0, r.stderr
    calls = stubs[1].read_text().splitlines()
    assert calls[0] == "hf auth whoami"
    assert calls[1] == f"hf download EEGDash/eegdash-corpus-v0 --repo-type dataset --revision v1.0 --local-dir {tmp_path}/data/eegdash-corpus-v0"
    assert r.stdout.strip().splitlines()[-1] == f"DATA={tmp_path}/data/eegdash-corpus-v0/eegdash-corpus-v0.zarr"


def test_download_refuses_without_a_directory_or_a_login(tmp_path, stubs):
    assert run("download.sh", [], {}, stubs).returncode != 0
    (stubs[0] / "hf").write_text("#!/bin/bash\nexit 1\n")
    r = run("download.sh", [str(tmp_path / "data")], {}, stubs)
    assert r.returncode == 1 and "hf auth login" in r.stderr and not (tmp_path / "data").exists()


@pytest.mark.skipif(shutil.which("sha256sum") is None, reason="sha256sum not installed")
def test_verify_checks_the_list_and_fails_on_a_changed_file(tmp_path, stubs):
    dest = tmp_path / "data" / "eegdash-corpus-v0"
    (dest / "eegdash-corpus-v0.zarr").mkdir(parents=True)
    names = [f"eegdash-corpus-v0.zarr/c{i}" for i in range(3)]
    for n in names:
        (dest / n).write_bytes(n.encode())
    sums = "".join(f"{hashlib.sha256(n.encode()).hexdigest()}  {n}\n" for n in names)
    (dest / "eegdash-corpus-v0-SHA256SUMS").write_text(sums)
    ok = run("download.sh", [str(tmp_path / "data")], {"VERIFY": "1", "JOBS": "2"}, stubs)
    assert ok.returncode == 0 and "checksums OK" in ok.stdout, ok.stderr
    (dest / names[1]).write_bytes(b"changed")
    bad = run("download.sh", [str(tmp_path / "data")], {"VERIFY": "1"}, stubs)
    assert bad.returncode != 0 and "checksums OK" not in bad.stdout


def launch_line(log: Path) -> str:
    lines = [c for c in log.read_text().splitlines() if c.startswith("accelerate launch")]
    assert len(lines) == 1
    return lines[0]


def test_pretrain_on_one_machine_uses_its_gpus_and_the_papers_settings(tmp_path, stubs):
    env = {"DATA": "/d/eegdash-corpus-v0.zarr", "CKPT": str(tmp_path / "ckpt"), "REVE_EEG": str(patched_checkout(tmp_path))}
    r = run("pretrain.sh", [], env, stubs)
    assert r.returncode == 0, r.stderr
    line = launch_line(stubs[1])
    assert "--num_processes 4 --num_machines 1 " in line and "--machine_rank" not in line and "--multi_gpu" not in line
    for setting in ("encoder=base", "data.path=/d/eegdash-corpus-v0.zarr", "trainer.epochs=17", "trainer.batch_size=300",
                    "trainer.lr=2.4e-4", "seed=2", "name=reve-base-v0-seed2", "checkpointing.load_last_state=False"):
        assert setting in line, setting


def test_pretrain_on_several_machines_passes_the_rendezvous(tmp_path, stubs):
    env = {"DATA": "/d/x.zarr", "CKPT": str(tmp_path / "ckpt"), "REVE_EEG": str(patched_checkout(tmp_path)), "NNODES": "4"}
    for missing in ({"NODE_RANK": "2"}, {"MAIN_HOST": "n01"}):
        r = run("pretrain.sh", [], {**env, **missing}, stubs)
        assert r.returncode != 0 and "several machines" in r.stderr and not stubs[1].exists()
    r = run("pretrain.sh", [], {**env, "NODE_RANK": "2", "MAIN_HOST": "n01", "SEED": "3"}, stubs)
    assert r.returncode == 0, r.stderr
    line = launch_line(stubs[1])
    assert "--num_processes 16 --num_machines 4 " in line and "--multi_gpu" in line
    assert "--machine_rank 2 --main_process_ip n01 --main_process_port 29500" in line
    assert "trainer.n_nodes=4" in line and "seed=3" in line and "name=reve-base-v0-seed3" in line


def test_pretrain_resumes_a_run_that_saved_a_state(tmp_path, stubs):
    ckpt = tmp_path / "ckpt"
    (ckpt / "states" / "reve-base-v0-seed2" / "last").mkdir(parents=True)
    env = {"DATA": "/d/x.zarr", "CKPT": str(ckpt), "REVE_EEG": str(patched_checkout(tmp_path))}
    assert run("pretrain.sh", [], env, stubs).returncode == 0
    assert "checkpointing.load_last_state=True" in launch_line(stubs[1])


def test_pretrain_refuses_a_missing_store_and_an_unpatched_checkout(tmp_path, stubs):
    reve = patched_checkout(tmp_path)
    assert run("pretrain.sh", [], {"CKPT": "/c", "REVE_EEG": str(reve)}, stubs).returncode != 0
    (reve / "src" / "utils" / "data_loading.py").write_text("# as cloned\n")
    r = run("pretrain.sh", [], {"DATA": "/d/x.zarr", "CKPT": "/c", "REVE_EEG": str(reve)}, stubs)
    assert r.returncode == 1 and "apply_patches.sh" in r.stderr and not stubs[1].exists()


def test_neither_script_is_tied_to_a_scheduler_or_a_site():
    """The tutorial must run on a workstation or any cluster: no scheduler directive, no account, no site path."""
    for name in ("download.sh", "pretrain.sh"):
        text = (REPO / name).read_text()
        assert not re.search(r"#SBATCH|\bsrun\b|\bsbatch\b|\bqsub\b|SLURM_|--account|--partition|/expanse|/work/hdd", text), name


def test_the_short_requirements_are_the_full_environments_pins():
    def pins(name: str) -> dict:
        lines = [ln.strip() for ln in (REPO / name).read_text().splitlines() if "==" in ln and not ln.startswith("#")]
        return {ln.split("==")[0].lower().replace("_", "-"): ln.split("==")[1].split("+")[0] for ln in lines}

    short, full = pins("requirements-pretraining.txt"), pins("requirements.txt")
    assert {"torch", "zarr", "accelerate", "hydra-core", "transformers", "huggingface-hub"} <= set(short)
    assert {k: full.get(k) for k in short} == short
