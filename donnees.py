"""Listes d'épisodes, encodeur gelé E_h et génération des démonstrations (sections 4 et 5.2).

Données au format zarr (un groupe par jeu, Blosc sans perte), placées dans
$ROBOSIMON_DONNEES (par défaut /temporary/$USER/donnees sur le PPTI). Les plongements E_h
sont précalculés une fois, en demi-précision, dans un fichier .npy à côté.
Usage : python donnees.py <g> <L> [<L> ...]
"""
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import torchvision
import zarr

from simon_sim import SimonEnv, plan_expert

RACINE = Path(__file__).parent
LISTES = RACINE / "listes"
DOSSIER = Path(os.environ.get("ROBOSIMON_DONNEES", RACINE / "donnees"))
N_TRAIN, N_VAL, N_TEST = 500, 30, 100
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def listes(L):
    """Listes figées (séquence, graine d'environnement) : entraînement, validation, test.
    Graines disjointes entre les trois listes."""
    f = LISTES / f"listes_L{L}.json"
    if f.exists():
        return json.loads(f.read_text())
    rng = np.random.default_rng(1000 + L)
    out = {}
    for nom, n, base in (("train", N_TRAIN, 0), ("val", N_VAL, 100_000), ("test", N_TEST, 200_000)):
        out[nom] = [{"sequence": rng.integers(0, 4, L).tolist(), "graine": L * 1_000_000 + base + i}
                    for i in range(n)]
    LISTES.mkdir(exist_ok=True)
    f.write_text(json.dumps(out))
    return out


class Encodeur(torch.nn.Module):
    """ResNet-18 ImageNet gelé ; carte finale regroupée en grille g x g puis aplatie."""

    def __init__(self, g=1):
        super().__init__()
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        self.tronc = torch.nn.Sequential(*list(r.children())[:-2]).eval()
        self.g = g
        self.register_buffer("moy", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("ect", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images_uint8):            # (B, H, W, 3) uint8
        x = torch.as_tensor(np.asarray(images_uint8)).to(self.moy.device)
        x = (x.permute(0, 3, 1, 2).float() / 255 - self.moy) / self.ect
        f = self.tronc(x)
        if f.device.type == "mps":              # MPS : regroupement non divisible non géré
            f = f.cpu()
        return torch.nn.functional.adaptive_avg_pool2d(f, self.g).flatten(1).to(self.moy.device)


def _episode(ep):
    """Joue l'expert sur un épisode (dans un processus de travail) et renvoie ses tableaux."""
    global _ENV
    if "_ENV" not in globals():
        _ENV = SimonEnv()
    env = _ENV
    obs, info = env.reset(ep["graine"], {"sequence": ep["sequence"]})
    plan, decisions = plan_expert(env, np.random.default_rng(ep["graine"] + 10_000_000))
    out = {k: [] for k in ("images", "etat", "action", "phase", "led", "pression")}
    fini, nb = False, 0
    while not fini:
        a = plan[env.t]
        out["images"].append(obs["image"])
        out["etat"].append(obs["etat"])
        out["action"].append(a.astype(np.float32))
        out["phase"].append(info["phase"])
        out["led"].append(info["led"])
        obs, _, fini, _, info = env.step(a)
        nouv = info["pressions"][nb:]
        out["pression"].append(nouv[0][1] if nouv else -1)
        nb = len(info["pressions"])
    assert info["resultat"] == "succes", (ep, info)
    out = {k: np.array(v) for k, v in out.items()}
    out["decision"] = np.zeros(len(out["etat"]), dtype=np.int8)
    out["decision"][decisions] = 1
    return out


def generer(L, partie="train", n_proc=18):
    """Joue l'expert sur une liste et écrit le groupe zarr du jeu."""
    chemin = DOSSIER / f"L{L}_{partie}.zarr"
    if (chemin / "fini").exists():
        return chemin
    eps = listes(L)[partie]
    racine = zarr.open_group(chemin, mode="w")
    blosc = zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
    tab = {"images": racine.create_array("images", shape=(0, 240, 320, 3), chunks=(16, 240, 320, 3),
                                         dtype="uint8", compressors=blosc)}
    for k, forme, dt in (("etat", (3,), "float32"), ("action", (3,), "float32"), ("phase", (), "int8"),
                         ("led", (5,), "uint8"), ("pression", (), "int8"), ("decision", (), "int8")):
        nom = k if k in ("etat", "action") else f"annotations/{k}"
        tab[k] = racine.create_array(nom, shape=(0, *forme), chunks=(4096, *forme), dtype=dt)
    debut, t0 = [0], time.time()
    tampon = {k: [] for k in tab}
    with get_context("spawn").Pool(n_proc) as pool:
        for i, out in enumerate(pool.imap(_episode, eps, chunksize=2)):
            for k in tab:
                tampon[k].append(out[k])
            debut.append(debut[-1] + len(out["etat"]))
            if len(tampon["etat"]) == 25 or i == len(eps) - 1:
                for k in tab:
                    tab[k].append(np.concatenate(tampon[k]))
                    tampon[k] = []
                print(f"L={L} {partie} {i + 1}/{len(eps)} {time.time() - t0:.0f}s", flush=True)
    racine.create_array("debut_episode", data=np.array(debut, dtype=np.int64))
    seqs = np.array([ep["sequence"] + [-1] * (4 - L) for ep in eps], dtype=np.int8)
    racine.create_array("annotations/sequence", data=seqs)
    racine.create_array("graine", data=np.array([ep["graine"] for ep in eps], dtype=np.int64))
    (chemin / "fini").touch()
    return chemin


def precalculer(L, g, partie="train"):
    """Plongements E_h (fp16) de toutes les images d'un jeu."""
    f = DOSSIER / f"L{L}_{partie}_g{g}.npy"
    if f.exists():
        return f
    images = zarr.open_group(DOSSIER / f"L{L}_{partie}.zarr", mode="r")["images"]
    enc = Encodeur(g).to(DEVICE)
    sortie = np.lib.format.open_memmap(f.with_suffix(".tmp.npy"), mode="w+", dtype=np.float16,
                                       shape=(images.shape[0], 512 * g * g))
    for i in range(0, images.shape[0], 512):
        sortie[i:i + 512] = enc(images[i:i + 512]).half().cpu().numpy()
    sortie.flush()
    del sortie
    f.with_suffix(".tmp.npy").rename(f)
    return f


def charger(L, partie="train", g=None):
    """Tableaux d'un jeu (sans les images) ; plongements si g est donné."""
    r = zarr.open_group(DOSSIER / f"L{L}_{partie}.zarr", mode="r")
    d = {"etat": r["etat"][:], "action": r["action"][:], "debut_episode": r["debut_episode"][:],
         "graine": r["graine"][:]}
    for k in ("phase", "led", "pression", "decision", "sequence"):
        d[k] = r[f"annotations/{k}"][:]
    if g is not None:
        d["plongements"] = np.load(DOSSIER / f"L{L}_{partie}_g{g}.npy", mmap_mode="r")
    return d



if __name__ == "__main__":
    g = int(sys.argv[1])
    for L in map(int, sys.argv[2:]):
        for partie in ("train", "test"):
            generer(L, partie)
            precalculer(L, g, partie)
