"""Tests des variantes (section 9.6), sur un petit jeu de données. Lancer sur une machine à GPU :
python test_politiques.py <L>. Écrit resultats/test_politiques.json."""
import json
import sys

import numpy as np
import torch

from donnees import DEVICE, Encodeur, charger, listes
from entrainement import G, RES
from evaluation import evaluer
from politiques import Donnees, creer, VARIANTES
from simon_sim import PRESENTATION


def sous_jeu(d, n):
    fin = d["debut_episode"][n]
    out = {k: v[:fin] for k, v in d.items() if k not in ("debut_episode", "sequence", "graine")}
    out |= {"debut_episode": d["debut_episode"][:n + 1], "sequence": d["sequence"][:n], "graine": d["graine"][:n]}
    out["plongements"] = np.asarray(d["plongements"][:fin])
    return out


def main(L, n_ep=10, n_iter=3000):
    d = sous_jeu(charger(L, "train", G), n_ep)
    donnees = Donnees(d, DEVICE)
    enc = Encodeur(G).to(DEVICE)
    eps = listes(L)["train"][:n_ep]
    res = {}
    for fam in ("ACT", "DP"):
        for var in VARIANTES:
            torch.manual_seed(0)
            m = creer(fam, var, d["plongements"].shape[1]).to(DEVICE)
            opt, sched = m.optimiseur(1e-4) if fam == "ACT" else m.optimiseur(n_iter=n_iter)
            m.train()
            for it in range(n_iter):
                idx = torch.randint(0, len(donnees.plong), (64,), device=DEVICE)
                perte, detail = m.perte(donnees.lot(idx, m.n_obs))
                opt.zero_grad()
                perte.backward()
                opt.step()
                if sched:
                    sched.step()
            m.eval()
            # sensibilité à l'historique : on remplace les images de présentation par du bruit
            idx = torch.arange(len(donnees.plong), device=DEVICE)[torch.as_tensor(d["phase"], device=DEVICE) > 3][::50]
            lot = donnees.lot(idx, m.n_obs, avec_action=False)
            g = torch.Generator(device=DEVICE)
            a1 = m.agir(lot, g.manual_seed(0))
            pres = torch.as_tensor(d["phase"], device=DEVICE)[donnees.debut[donnees.episode[idx]][:, None]
                                                                + lot["age"].mul(-30).round().long()
                                                                + donnees.t[idx][:, None]] == PRESENTATION
            lot2 = dict(lot)
            lot2["hist"] = torch.where(pres[..., None], lot["hist"][torch.randperm(len(idx))], lot["hist"])
            a2 = m.agir(lot2, g.manual_seed(0))
            ecart = float((a1 - a2).abs().max())
            succ = np.mean([r["succes"] for r in evaluer(m, enc, eps, device=DEVICE)])
            res[f"{fam}_{var}"] = {"perte_finale": perte.item(), "detail": detail, "succes_train": float(succ),
                                   "ecart_hist_presentation": ecart}
            print(fam, var, res[f"{fam}_{var}"], flush=True)
    for fam in ("ACT", "DP"):
        assert res[f"{fam}_B0"]["ecart_hist_presentation"] == 0, "B0 dépend de l'historique"
        assert res[f"{fam}_M1"]["ecart_hist_presentation"] > 1e-3, "M1 ignore la présentation"
    (RES / "test_politiques.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]))
