"""E0 : validation du banc (section 6.3). Écrit resultats/e0.json.
Usage : python e0.py expert | visibilite | fuites"""
import json
import sys
from multiprocessing import Pool

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from donnees import RACINE, DEVICE, Encodeur, charger
from simon_sim import SimonEnv, episode_expert, EXECUTION

FICHIER = RACINE / "resultats" / "e0.json"


def enregistrer(cle, valeur):
    FICHIER.parent.mkdir(exist_ok=True)
    tout = json.loads(FICHIER.read_text()) if FICHIER.exists() else {}
    tout[cle] = valeur
    FICHIER.write_text(json.dumps(tout, indent=1))
    print(cle, json.dumps(valeur, indent=1))


def sonde(X, y, groupes, n_classes, alpha=0.05):
    """Sonde linéaire, validation croisée groupée par épisode (5 plis) et test binomial
    unilatéral contre le hasard. Renvoie précision, p-valeur, absence de fuite."""
    modele = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    pred = cross_val_predict(modele, X, y, groups=groupes,
                             cv=GroupKFold(n_splits=5))
    n_ok = int((pred == y).sum())
    test = binomtest(n_ok, len(y), p=1 / n_classes, alternative="greater")
    return n_ok / len(y), test.pvalue, bool(test.pvalue >= alpha)


def _expert(args):
    L, i = args
    rng = np.random.default_rng(L * 7919 + i)
    seq = rng.integers(0, 4, L).tolist()
    *_, infos = episode_expert(5_000_000 + L * 10_000 + i, seq)
    return infos[-1]["resultat"], infos[-1]["erreur"]


def expert():
    res = {}
    with Pool(6) as p:
        for L in (2, 3, 4):
            r = p.map(_expert, [(L, i) for i in range(500)])
            ok = sum(x[0] == "succes" for x in r)
            res[f"L{L}"] = {"succes": ok / 500, "echecs": [x for x in r if x[0] != "succes"]}
    enregistrer("expert", res)


def visibilite(n_ep=600):
    """Décode l'état des LED (éteint, 4 couleurs, GO) depuis E_h pour g = 1, 2, 4.
    Une image par épisode, tirée avant l'exécution (doigt en pose de repos)."""
    rng = np.random.default_rng(0)
    env = SimonEnv()
    images, y = [], []
    for i in range(n_ep):
        seq = rng.integers(0, 4, 3).tolist()
        env.reset(9_000_000 + i, {"sequence": seq, "rendu": False})
        classe = i % 6                     # classes équilibrées
        if classe == 0:
            cands = [t for t in range(env.debut_exec) if env.etat_led(t).sum() == 0]
        elif classe == 5:
            cands = [t for t in range(env.debut_exec) if env.etat_led(t)[4]]
        else:
            cands = [t for t in range(env.debut_exec) if env.etat_led(t)[classe - 1]]
        if not cands:                     # couleur absente de la séquence : on la force
            seq[0] = classe - 1
            env.reset(9_000_000 + i, {"sequence": seq, "rendu": False})
            cands = [t for t in range(env.debut_exec) if env.etat_led(t)[classe - 1]]
        env.t = int(rng.choice(cands))
        env._maj_led()
        images.append(env._obs(True)["image"])
        y.append(classe)
    y = np.array(y)
    res = {}
    for g in (1, 2, 4):
        enc = Encodeur(g).to(DEVICE)
        X = np.concatenate([enc(np.stack(images[i:i + 128])).cpu().numpy() for i in range(0, n_ep, 128)])
        acc, _, _ = sonde(X, y, np.arange(n_ep), 6)
        res[f"g{g}"] = acc
    res["g_retenu"] = min([g for g in (1, 2, 4) if res[f"g{g}"] >= 0.99] or [4])
    enregistrer("visibilite", res)


def fuites(L=3, n_ep=400):
    """Sondes de fuite : s_1 depuis une image d'exécution tirée au hasard (une par épisode),
    depuis la première image d'exécution (doigt encore au repos : fuite de la scène seule),
    et depuis q_t au signal de départ. Contrôle positif : image du premier allumage.
    Sur une image d'exécution tirée au hasard, le doigt de l'expert est souvent sur s_1 ou
    s'y dirige : la sonde lit alors le comportement de l'agent, pas une fuite de la scène."""
    g = json.loads(FICHIER.read_text())["visibilite"]["g_retenu"]
    d = charger(L, g=g)
    rng = np.random.default_rng(1)
    X_img, X_img0, X_pres, X_etat, y = [], [], [], [], []
    for e in range(n_ep):
        a, b = d["debut_episode"][e], d["debut_episode"][e + 1]
        exe = np.arange(a, b)[d["phase"][a:b] == EXECUTION]
        X_img.append(d["plongements"][rng.choice(exe)].astype(np.float32))
        X_img0.append(d["plongements"][exe[0]].astype(np.float32))       # doigt encore au repos
        pres = np.arange(a, b)[d["phase"][a:b] == 1]
        X_pres.append(d["plongements"][pres[9]].astype(np.float32))     # contrôle positif : 1er allumage
        depart = a + np.argmax(d["phase"][a:b] == 3)
        X_etat.append(d["etat"][depart])
        y.append(d["sequence"][e, 0])
    y, grp = np.array(y), np.arange(n_ep)
    res = {}
    for nom, X in (("image", np.array(X_img)), ("image_debut_execution", np.array(X_img0)),
                   ("etat", np.array(X_etat)), ("controle_positif_presentation", np.array(X_pres))):
        acc, p, ok = sonde(X, y, grp, 4)
        res[nom] = {"precision": acc, "p_valeur": p, "absence_de_fuite": ok}
    enregistrer("fuites", res)


if __name__ == "__main__":
    {"expert": expert, "visibilite": visibilite, "fuites": fuites}[sys.argv[1]]()
