"""Diagnostics (sections 6.6 et 7.2) : indice d'inertie, sondes linéaires sur les jetons
mémoire, taux de suivi contrefactuel. Lit les politiques retenues, écrit
resultats/diagnostics/<condition>.json. Usage : python diagnostics.py <condition> [...]
(ex. E1_ACT_M1_L3_s0)."""
import json
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from donnees import DEVICE, Encodeur, charger, listes
from entrainement import CKPT, G, RES
from evaluation import evaluer
from politiques import Donnees, creer
from simon_sim import DEPART


@torch.no_grad()
def indice_inertie(modele, d, graine=0):
    """Boucle ouverte sur les démonstrations de test, aux instants où l'expert quitte un
    bouton pour un autre. II = part des instants où le bloc prédit est plus proche du
    maintien de la dernière action que du bloc de l'expert."""
    idx = torch.nonzero(torch.as_tensor(d["decision"]) == 1)[:, 0].to(DEVICE)
    donnees = Donnees(d, DEVICE, k_max=128)
    lot = donnees.lot(idx, modele.n_obs)
    gen = torch.Generator(device=DEVICE).manual_seed(graine)
    pred = modele.agir(lot, gen)
    expert = lot["action"]
    maintien = donnees.action[idx - 1][:, None].expand_as(expert)
    d_maintien = (pred - maintien).flatten(1).norm(dim=1)
    d_expert = (pred - expert).flatten(1).norm(dim=1)
    detail = (d_maintien < d_expert).int().tolist()
    return float(np.mean(detail)), detail


@torch.no_grad()
def sondes(modele, d):
    """Jetons mémoire au signal de départ, une ligne par épisode de test ; une régression
    logistique par position de la séquence, validation croisée en 5 plis (une ligne = un
    épisode, donc plis groupés par épisode)."""
    debut = d["debut_episode"]
    idx = [debut[e] + int(np.argmax(d["phase"][debut[e]:debut[e + 1]] == DEPART)) for e in range(len(debut) - 1)]
    donnees = Donnees(d, DEVICE, k_max=128)
    lot = donnees.lot(torch.as_tensor(idx, device=DEVICE), modele.n_obs, avec_action=False)
    modele.eval()
    jetons = modele.memoire.module(lot["hist"], lot["age"], lot["vide"], lot.get("hist_etat"))
    X = jetons.flatten(1).cpu().numpy()
    res = []
    for j in range(int((d["sequence"][0] >= 0).sum())):
        y = d["sequence"][:, j]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1))
        res.append(float(cross_val_score(clf, X, y, cv=KFold(5, shuffle=True, random_state=0)).mean()))
    return res


def suivi_contrefactuel(modele, enc, L, graine):
    """Remplace les images de la présentation par celles d'une séquence s' (s'_1 != s_1)
    et observe le premier bouton pressé."""
    rng = np.random.default_rng(100 + graine)
    eps = []
    for ep in listes(L)["test"]:
        s = ep["sequence"]
        while True:
            s2 = rng.integers(0, 4, L).tolist()
            if s2[0] != s[0]:
                break
        eps.append(ep | {"substitut": s2})
    r = evaluer(modele, enc, eps, graine=graine, device=DEVICE)
    premiere = np.array([x["premiere_pression"] for x in r])
    return {"tsc": float(np.mean(premiere == [e["substitut"][0] for e in eps])),
            "p_s1": float(np.mean(premiere == [e["sequence"][0] for e in eps])),
            "p_aucune": float(np.mean(premiere == -1)), "n": len(eps)}


def diagnostiquer(nom):
    exp, famille, variante, L, s = nom.split("_")[:5]
    L, graine = int(L[1:]), int(s[1:])
    sortie = RES / "diagnostics" / f"{nom}.json"
    if sortie.exists():
        return
    d = charger(L, "test", G)
    modele = creer(famille, variante, d["plongements"].shape[1]).to(DEVICE)
    modele.load_state_dict(torch.load(CKPT / nom / "retenu.pt", map_location=DEVICE))
    modele.eval()
    res = {"experience": exp, "famille": famille, "variante": variante, "L": L, "graine": graine}
    res["inertie"], res["inertie_detail"] = indice_inertie(modele, d, graine)
    if variante != "B0":
        res["sondes"] = sondes(modele, d)
        res |= suivi_contrefactuel(modele, Encodeur(G).to(DEVICE), L, graine)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(json.dumps(res, indent=1))
    print(nom, res, flush=True)


if __name__ == "__main__":
    for nom in sys.argv[1:]:
        diagnostiquer(nom)
