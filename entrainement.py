"""Entraînement, sélection du point de sauvegarde et évaluation de test (sections 5.5 et 9.7).

Usage : python entrainement.py <exp>:<famille>:<variante>:<L>:<graine> [...]
Chaque condition est jouée à la suite. Les résultats (une ligne par épisode évalué) vont dans
resultats/episodes/<condition>.parquet, le journal de perte dans resultats/pertes/.
"""
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from donnees import RACINE, DEVICE, Encodeur, charger, listes
from evaluation import evaluer
from politiques import Donnees, creer

N_ITER = int(os.environ.get("ROBOSIMON_ITER", 20_000))
TAILLE_LOT = 64
LR_ACT = float(os.environ.get("ROBOSIMON_LR_ACT", 1e-4))   # 1e-5 sans plateau sur B1 en E0 (section 5.3)
G = json.loads((RACINE / "resultats" / "e0.json").read_text())["visibilite"]["g_retenu"]
CKPT = Path(os.environ.get("TMPD", RACINE)) / "ckpt"
RES = RACINE / "resultats"


def hachage_code():
    h = hashlib.sha1()
    for f in sorted(RACINE.glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:10]


def points_selection(n_iter):
    """Trois points régulièrement espacés sur la seconde moitié de l'entraînement."""
    return [int(round(n_iter * f)) for f in (2 / 3, 5 / 6, 1)]


def entrainer(famille, variante, L, graine, nom, n_iter=N_ITER):
    dossier = CKPT / nom
    dossier.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(graine)
    np.random.seed(graine)
    donnees = Donnees(charger(L, "train", G), DEVICE)
    modele = creer(famille, variante, donnees.plong.shape[1]).to(DEVICE)
    opt, sched = modele.optimiseur(LR_ACT) if famille == "ACT" else modele.optimiseur(n_iter=n_iter)
    gen = torch.Generator(device=DEVICE).manual_seed(graine)
    debut = 0
    if (dossier / "dernier.pt").exists():                    # reprise après interruption
        etat = torch.load(dossier / "dernier.pt", map_location=DEVICE)
        modele.load_state_dict(etat["modele"])
        opt.load_state_dict(etat["opt"])
        if sched:
            sched.load_state_dict(etat["sched"])
        gen.set_state(etat["gen"].cpu() if DEVICE != "cuda" else etat["gen"])
        debut = etat["iter"]
    journal = RES / "pertes" / f"{nom}.csv"
    journal.parent.mkdir(parents=True, exist_ok=True)
    if debut == 0:
        journal.write_text("iter,perte,detail\n")
    points = points_selection(n_iter)
    modele.train()
    t0, cumul = time.time(), []
    for it in range(debut + 1, n_iter + 1):
        idx = torch.randint(0, len(donnees.plong), (TAILLE_LOT,), device=DEVICE, generator=gen)
        perte, detail = modele.perte(donnees.lot(idx, modele.n_obs, passe=getattr(modele, "n_passe", 0)))
        opt.zero_grad(set_to_none=True)
        perte.backward()
        torch.nn.utils.clip_grad_norm_(modele.parameters(), 10.0)
        opt.step()
        if sched:
            sched.step()
        cumul.append(perte.item())
        if it % 200 == 0:
            with journal.open("a") as f:
                f.write(f"{it},{np.mean(cumul):.5f},\"{json.dumps(detail)}\"\n")
            cumul = []
        if it % 2000 == 0 or it in points:
            etat = {"modele": modele.state_dict(), "opt": opt.state_dict(), "iter": it,
                    "sched": sched.state_dict() if sched else None, "gen": gen.get_state()}
            torch.save(etat, dossier / "dernier.tmp")
            os.replace(dossier / "dernier.tmp", dossier / "dernier.pt")
        if it in points:
            torch.save(modele.state_dict(), dossier / f"it{it}.pt")
            print(f"{nom} it {it} {time.time() - t0:.0f}s", flush=True)
    return modele, points


def lignes(res, episodes, meta, partie, it):
    return [meta | {"partie": partie, "iter": it, "episode": i, "sequence": json.dumps(ep["sequence"]),
                    "graine_env": ep["graine"], "pressions": json.dumps(r["pressions"]),
                    "pos": json.dumps(r["pos"])} | {k: v for k, v in r.items() if k not in ("pressions", "pos")}
            for i, (r, ep) in enumerate(zip(res, episodes))]


def condition(spec):
    exp, famille, variante, L, graine, *it = spec.split(":")     # 6e champ facultatif : itérations
    L, graine = int(L), int(graine)
    n_iter = int(it[0]) if it else N_ITER
    nom = f"{exp}_{famille}_{variante}_L{L}_s{graine}" + (f"_it{n_iter}" if it else "")
    sortie = RES / "episodes" / f"{nom}.parquet"
    marque = RES / "episodes" / f"{nom}.encours"
    marque.parent.mkdir(parents=True, exist_ok=True)
    if sortie.exists():
        print("déjà fait :", nom)
        return
    try:                                                 # réservation atomique de la condition
        fd = os.open(marque, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print("en cours ailleurs :", nom)
        return
    os.write(fd, os.uname().nodename.encode())
    os.close(fd)
    t0 = time.time()
    modele, points = entrainer(famille, variante, L, graine, nom, n_iter)
    t_entr = time.time() - t0
    enc = Encodeur(G).to(DEVICE)
    meta = {"experience": exp, "famille": famille, "variante": variante, "L": L, "graine": graine, "n_iter": n_iter,
            "commit": hachage_code()}
    lst = listes(L)
    rows, scores = [], {}
    for it in points:                                   # sélection sur la validation
        modele.load_state_dict(torch.load(CKPT / nom / f"it{it}.pt", map_location=DEVICE))
        r = evaluer(modele, enc, lst["val"], graine=graine, device=DEVICE)
        rows += lignes(r, lst["val"], meta, "val", it)
        scores[it] = (np.mean([x["succes"] for x in r]), np.mean([x["prefixe"] for x in r]), it)
    meilleur = max(scores, key=scores.get)
    modele.load_state_dict(torch.load(CKPT / nom / f"it{meilleur}.pt", map_location=DEVICE))
    r = evaluer(modele, enc, lst["test"], graine=graine, device=DEVICE)
    rows += lignes(r, lst["test"], meta, "test", meilleur)
    torch.save(modele.state_dict(), CKPT / nom / "retenu.pt")
    for it in points:
        (CKPT / nom / f"it{it}.pt").unlink()
    sortie.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(sortie)
    duree = {"machine": os.uname().nodename, "entrainement_s": t_entr, "total_s": time.time() - t0, "retenu": meilleur,
             "succes_test": float(np.mean([x["succes"] for x in r]))}
    (RES / "episodes" / f"{nom}.json").write_text(json.dumps(duree))
    marque.unlink()
    print(nom, duree, flush=True)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    for spec in sys.argv[1:]:
        condition(spec)
