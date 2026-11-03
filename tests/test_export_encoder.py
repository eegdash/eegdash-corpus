"""A reve_eeg encoder state dict converts into braindecode's REVE with only the head missing; the Hub format
(save_pretrained / from_pretrained) round-trips and the model card carries licence and SHA-256."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


def _encoder_state(reve_src, depth, path):
    from models.encoder import get_reve_model

    cfg = SimpleNamespace(encoder=SimpleNamespace(
        transformer=SimpleNamespace(embed_dim=512, depth=depth, heads=8, head_dim=64, mlp_dim_ratio=2.66, use_geglu=True),
        freqs=4, noise_ratio=0.0025, patch_size=200, patch_overlap=20))
    torch.save(get_reve_model(cfg).state_dict(), path)
    return path


@pytest.fixture(scope="module")
def local_encoder_state(reve_src, tmp_path_factory):
    return _encoder_state(reve_src, 22, tmp_path_factory.mktemp("enc") / "encoder.pth")


def test_export_matches_braindecode_keys(local_encoder_state, scripts_on_path, tmp_path):
    out = tmp_path / "reve_braindecode.pth"
    r = subprocess.run([sys.executable, str(scripts_on_path / "export_encoder.py"),
                        "--encoder", str(local_encoder_state), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "head only: True" in r.stdout and "unexpected=0" in r.stdout
    sd = torch.load(out, map_location="cpu")
    assert any(k.startswith("transformer.layers.21.") for k in sd)  # depth 22 made it through
    assert "depth 22" in r.stdout


def test_export_infers_small_depth(reve_src, scripts_on_path, tmp_path):
    """A REVE-Small snapshot (encoder=small, 4 layers) exports without flags: the size is read off the state dict."""
    enc = _encoder_state(reve_src, 4, tmp_path / "encoder_epoch_5.pth")
    out = tmp_path / "small.pth"
    r = subprocess.run([sys.executable, str(scripts_on_path / "export_encoder.py"), "--encoder", str(enc), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "encoder size: embed_dim 512 depth 4 heads 8" in r.stdout and "head only: True" in r.stdout
    sd = torch.load(out, map_location="cpu")
    assert not any(k.startswith("transformer.layers.4.") for k in sd)


def test_random_init_export_is_a_loadable_small_floor(scripts_on_path, tmp_path):
    """--random-init --depth 4 writes an untrained REVE-Small in the export format; finetune_reve/openeegbench load it like a snapshot."""
    out = tmp_path / "random_init_small.pth"
    r = subprocess.run([sys.executable, str(scripts_on_path / "export_encoder.py"), "--random-init", "--depth", "4", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "random init: embed_dim 512 depth 4 heads 8" in r.stdout
    sd = torch.load(out, map_location="cpu")
    assert any(k.startswith("transformer.layers.3.") for k in sd) and not any(k.startswith("transformer.layers.4.") for k in sd)
    bad = subprocess.run([sys.executable, str(scripts_on_path / "export_encoder.py"), "--out", str(out)], capture_output=True, text=True)
    assert bad.returncode != 0 and "exactly one" in bad.stderr


CHS = [{"ch_name": n} for n in ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3", "CZ", "C4", "T8", "P7", "P3", "PZ", "P4", "P8", "O1", "O2"]]


def test_hub_format_round_trips_offline(tmp_path):
    """What --push-to-hub uploads (config.json + weights) rebuilds the same model with REVE.from_pretrained, size included."""
    import json

    pytest.importorskip("huggingface_hub")
    from braindecode.models import REVE

    torch.manual_seed(0)
    model = REVE(n_outputs=2, n_chans=19, n_times=2000, sfreq=200, embed_dim=64, depth=1, heads=2, chs_info=CHS)
    model.save_pretrained(tmp_path)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert (cfg["depth"], cfg["embed_dim"], cfg["heads"]) == (1, 64, 2)
    a, b = model.state_dict(), REVE.from_pretrained(str(tmp_path)).state_dict()
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def test_model_card_names_licence_hash_and_run(scripts_on_path):
    from export_encoder import model_card

    sha = "ab" * 32
    card = model_card("example-org/reve-base-run3", sha, run="base-run3", epoch="final", arch=dict(embed_dim=512, depth=22, heads=8))
    assert len(card) >= 200 and card.startswith("---\nlicense: cc-by-nc-sa-4.0\nlibrary_name: braindecode\n")
    assert sha in card and "base-run3" in card and "REVE.from_pretrained(\"example-org/reve-base-run3\")" in card
    assert "eegdash-corpus-v0" in card and "- eeg" in card
    assert "license: mit" in model_card("x/y", sha, license="mit", note="custom note")
    assert "custom note" in model_card("x/y", sha, note="custom note")


def test_push_flags_are_checked_before_any_upload(scripts_on_path, tmp_path):
    out = tmp_path / "x.pth"
    for extra in (["--random-init", "--push-to-hub", "a/b"], ["--random-init", "--verify"]):
        r = subprocess.run([sys.executable, str(scripts_on_path / "export_encoder.py"), "--out", str(out), *extra], capture_output=True, text=True)
        assert r.returncode != 0 and not out.exists()
