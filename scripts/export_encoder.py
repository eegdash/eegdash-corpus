"""Turn reve_eeg's `encoder.pth` (saved at the end of train.py, or an `encoder_epoch_<k>.pth` snapshot) into a
braindecode REVE state dict that finetune_reve.py --checkpoint and OpenEEGBench can load.

Key names: reve_eeg's local REVE class loads the Hugging Face `brain-bzh/reve-base` state dict with
strict=True (src/models/encoder.py), and braindecode's REVE.from_pretrained loads the same HF repo, so
the two key sets are expected to match except for braindecode's classification head (`final_layer.*`).
This script verifies that instead of assuming it.

The model size (embed dim, depth, heads) is read off the state dict, so Small (depth 4) and Base (depth 22)
exports need no flags; --embed-dim/--depth/--heads override the inference.

  python scripts/export_encoder.py --encoder runs/reve-base-v0-run1/encoder.pth --out runs/reve-base-v0-run1/reve_braindecode.pth
  python scripts/export_encoder.py --random-init --depth 4 --out random_init_small.pth     # untrained floor, same format
  python scripts/export_encoder.py --encoder encoder.pth --out reve_braindecode.pth --push-to-hub <user>/reve-base-run3 \
         --run base-run3 --epoch final --verify     # also publish it (private) with a model card; --verify reloads it from the Hub

--push-to-hub writes config.json + weights with braindecode's PyTorchModelHubMixin.push_to_hub (private repo), then
replaces the auto-generated README.md with a model card (licence, how to load, SHA-256 of the --out file, run/epoch).
"""
import argparse, hashlib, re, torch
from pathlib import Path

DEFAULT_NOTE = ("REVE-Base encoder trained from scratch on the `reve-public-nc` corpus (named `eegdash-corpus-v0` in the "
                "submitted paper): 66 public datasets of the REVE release (65 through EEGDash plus I-CARE), 45,549 hours "
                "of EEG. Pass --card-note for a camera-ready run on `eegdash-corpus-v0` (77 datasets, without I-CARE).")

def arch_from_state_dict(sd: dict, head_dim: int = 64) -> dict:
    """embed_dim from the patch embedding, depth from the highest transformer layer index, heads from to_qkv."""
    embed_dim = sd["to_patch_embedding.0.weight"].shape[0]
    layers = [int(m[1]) for k in sd for m in [re.match(r"transformer\.layers\.(\d+)\.", k)] if m]
    depth = max(layers) + 1
    heads = sd["transformer.layers.0.0.to_qkv.weight"].shape[0] // (3 * head_dim)
    return dict(embed_dim=embed_dim, depth=depth, heads=heads)

def sha256_of(path: str | Path) -> str:
    """Full hex SHA-256 of a file; the card quotes it so a download can be matched to the benchmark rows' `weights`."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()

def model_card(repo: str, sha256: str, license: str = "cc-by-nc-sa-4.0", note: str | None = None,
               run: str | None = None, epoch: str | None = None, arch: dict | None = None) -> str:
    """README.md for the Hub repo: YAML front matter, what the weights are, how to load them, their identity."""
    arch_line = f"- size: embed_dim {arch['embed_dim']}, depth {arch['depth']}, heads {arch['heads']}\n" if arch else ""
    run_line = f"- training run: `{run}`" + (f", epoch `{epoch}`" if epoch else "") + "\n" if run or epoch else ""
    return (f"---\nlicense: {license}\nlibrary_name: braindecode\ntags:\n- eeg\n- reve\n---\n\n"
            f"# {repo.split('/')[-1]}\n\n{note or DEFAULT_NOTE}\n\n"
            "The weights are braindecode's `REVE` class; the classification head (`final_layer`) is untrained, "
            "so pass `n_outputs` for your task and train the head (or fine-tune) before use.\n\n"
            f"```python\nfrom braindecode.models import REVE\nmodel = REVE.from_pretrained(\"{repo}\")\n```\n\n"
            f"- exported state dict SHA-256: `{sha256}`\n{arch_line}{run_line}"
            f"- licence: {license}\n")

def push(model, repo: str, card: str, message: str) -> None:
    """config.json + weights via the mixin, then our card over the generated README.md."""
    from huggingface_hub import upload_file
    model.push_to_hub(repo, private=True, commit_message=message)
    upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo, commit_message="Model card")
    print(f"pushed {repo} (private)")

def verify(model, repo: str) -> None:
    """Reload from the Hub and compare every tensor: proves the repo loads with REVE.from_pretrained alone."""
    from braindecode.models import REVE
    a, b = model.state_dict(), REVE.from_pretrained(repo).state_dict()
    bad = sorted(set(a) ^ set(b)) + [k for k in set(a) & set(b) if not torch.equal(a[k], b[k])]
    if bad:
        raise SystemExit(f"round trip from {repo} differs on {len(bad)} tensor(s): {bad[:5]}")
    print(f"verified {repo}: {len(a)} tensors identical")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=None, help="reve_eeg encoder state dict (encoder.pth / encoder_epoch_<k>.pth)")
    ap.add_argument("--random-init", action="store_true", help="no weights: an untrained REVE of the given size (floor for benchmarks)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--embed-dim", type=int, default=None); ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--push-to-hub", default=None, metavar="REPO", help="also publish the model (private) to this Hub repo")
    ap.add_argument("--card-license", default="cc-by-nc-sa-4.0"); ap.add_argument("--card-note", default=None, help="what the weights are (card body)")
    ap.add_argument("--run", default=None); ap.add_argument("--epoch", default=None, help="named in the card and commit message")
    ap.add_argument("--verify", action="store_true", help="after --push-to-hub, reload from the Hub and compare state dicts")
    a = ap.parse_args()
    if bool(a.encoder) == a.random_init:
        raise SystemExit("give exactly one of --encoder or --random-init")
    if a.push_to_hub and a.random_init:
        raise SystemExit("--push-to-hub publishes trained weights; not with --random-init")
    if a.verify and not a.push_to_hub:
        raise SystemExit("--verify checks a --push-to-hub upload; give both")
    from braindecode.models import REVE
    if a.random_init:
        torch.manual_seed(0)
        arch = dict(embed_dim=a.embed_dim or 512, depth=a.depth or 22, heads=a.heads or 8)
        model = REVE(n_outputs=2, n_chans=19, n_times=2000, sfreq=200, **arch,
                     chs_info=[{"ch_name": n} for n in ["FP1","FP2","F7","F3","FZ","F4","F8","T7","C3","CZ","C4","T8","P7","P3","PZ","P4","P8","O1","O2"]])
        print(f"random init: embed_dim {arch['embed_dim']} depth {arch['depth']} heads {arch['heads']} (seed 0)")
        torch.save(model.state_dict(), a.out); print("wrote", a.out); return
    sd = torch.load(a.encoder, map_location="cpu")
    sd = {k.replace("encoder.", "", 1) if k.startswith("encoder.") else k: v for k, v in sd.items()}
    sd.pop("cls_query_token", None)
    arch = arch_from_state_dict(sd)
    for k in ("embed_dim", "depth", "heads"):
        if getattr(a, k) is not None:
            arch[k] = getattr(a, k)
    print(f"encoder size: embed_dim {arch['embed_dim']} depth {arch['depth']} heads {arch['heads']}")
    model = REVE(n_outputs=2, n_chans=19, n_times=2000, sfreq=200, **arch,
                 chs_info=[{"ch_name": n} for n in ["FP1","FP2","F7","F3","FZ","F4","F8","T7","C3","CZ","C4","T8","P7","P3","PZ","P4","P8","O1","O2"]])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    head_only = all(k.startswith("final_layer") for k in missing)
    print(f"missing={len(missing)} (head only: {head_only})  unexpected={len(unexpected)}")
    if unexpected or not head_only:
        print("  missing:", missing[:10]); print("  unexpected:", unexpected[:10])
        raise SystemExit("Key mismatch between reve_eeg encoder and braindecode REVE; inspect before exporting.")
    torch.save(model.state_dict(), a.out)
    print("wrote", a.out)
    if a.push_to_hub:
        sha = sha256_of(a.out)
        tag = " ".join(x for x in (a.run, a.epoch and f"epoch {a.epoch}") if x)
        push(model, a.push_to_hub, model_card(a.push_to_hub, sha, a.card_license, a.card_note, a.run, a.epoch, arch),
             f"REVE encoder export {tag} sha256 {sha[:16]}".replace("  ", " "))
        if a.verify:
            verify(model, a.push_to_hub)

if __name__ == "__main__":
    main()
