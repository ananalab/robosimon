"""Évaluation en boucle fermée, en lot (sections 3.2, 5.6 et 7.1).

Chaque environnement tourne dans son propre processus ; la politique est interrogée
en un seul appel GPU pour tous les environnements qui en ont besoin. Seules les images
utiles sont rendues : grille d'historique (t multiple de S_H) et images d'interrogation
(t multiple de H_EXEC, et t - 1 pour DP).
"""
import multiprocessing as mp

import numpy as np
import torch

from politiques import H_EXEC, K_MAX, indices_grille, normaliser, denormaliser, oracle_onehot
from simon_sim import SimonEnv, PRESENTATION, FPS, S_H, chronologie


def _travailleur(conn):
    env = SimonEnv()
    while True:
        cmd, arg = conn.recv()
        if cmd == "fin":
            return
        if cmd == "reset":
            ep, rendu = arg
            obs, info = env.reset(ep["graine"], {"sequence": ep["sequence"], "rendu": False})
            env.substitut = ep.get("substitut")
            if rendu:
                obs["image"] = _rendre(env)
            conn.send((obs, info, False))
        else:
            action, rendu = arg
            obs, _, fini, _, info = env.step(action, rendu=False)
            if rendu:
                obs["image"] = _rendre(env)
            info["contacts_rates"] = env.contacts_rates
            conn.send((obs, info, fini))


def _rendre(env):
    """Rendu de l'image courante. Pour le test de substitution contrefactuelle, les images
    de la phase de présentation sont celles de la séquence substituée s' (même graine,
    donc même scène ; seul l'état des LED diffère)."""
    if env.substitut is not None and env.phase() == PRESENTATION:
        vraie = env.led
        intervalles, _ = chronologie(env.substitut)
        env.led = np.array([any(a <= env.t < b and c == i for a, b, c in intervalles) for i in range(5)],
                           dtype=np.uint8)
        env.intervalles, sauv = intervalles, env.intervalles
        env._maj_led()
        img = env._obs(True)["image"]
        env.intervalles = sauv
        env._maj_led()
        assert (env.led == vraie).all()
        return img
    return env._obs(True)["image"]


class VecEnv:
    def __init__(self, n):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_travailleur, args=(b,), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)

    def fermer(self):
        for c in self.conns:
            c.send(("fin", None))
        for p in self.procs:
            p.join()


def _a_rendre(t, n_obs):
    return t % S_H == 0 or t % H_EXEC == 0 or (n_obs == 2 and t % H_EXEC == H_EXEC - 1)


def resumer(info, L):
    """Métriques d'un épisode à partir de l'info finale."""
    seq, press = info["sequence"], info["pressions"]
    correctes = info["progres"]
    # précision par position : la pression j est jugée si le préfixe 1..j-1 est correct
    pos = [None] * 4
    for j in range(L):
        if j < correctes:
            pos[j] = 1
        elif j == correctes and info["erreur"] in ("memoire", "repetition"):
            pos[j] = 0
    return {"succes": int(info["resultat"] == "succes"), "prefixe": correctes / L,
            "erreur": info["erreur"] or "", "pressions": [[float(t), int(b)] for t, b in press],
            "premiere_pression": int(press[0][1]) if press else -1,
            "duree": info["t"], "pos": pos}


@torch.no_grad()
def evaluer(politique, encodeur, episodes, n_env=20, graine=0, device="cuda"):
    """Joue tous les épisodes (liste de dicts sequence, graine[, substitut]) en boucle fermée.
    Renvoie une liste de dicts de métriques, dans l'ordre des épisodes."""
    politique.eval()
    n_obs = politique.n_obs
    gen = torch.Generator(device=device).manual_seed(graine)
    venv = VecEnv(min(n_env, len(episodes)))
    n = len(venv.conns)
    file = list(range(len(episodes)))
    slot = [None] * n          # état par environnement
    resultats = [None] * len(episodes)

    def demarrer(i):
        if not file:
            slot[i] = None
            return
        k = file.pop(0)
        venv.conns[i].send(("reset", (episodes[k], True)))
        slot[i] = {"ep": k, "t": 0, "emb": {}, "etats": {}, "bloc": None,
                   "sequence": episodes[k]["sequence"], "L": len(episodes[k]["sequence"])}

    def recevoir(ids):
        msgs = [venv.conns[i].recv() for i in ids]
        imgs = [m[0]["image"] for m in msgs if "image" in m[0]]
        plong = encodeur(np.stack(imgs)).float() if imgs else None
        j = 0
        for i, (obs, info, fini) in zip(ids, msgs):
            s = slot[i]
            s["etats"][s["t"]] = obs["etat"]
            if "image" in obs:
                s["emb"][s["t"]] = plong[j]
                j += 1
            s["info"], s["fini"] = info, fini

    try:
        for i in range(n):
            demarrer(i)
        recevoir([i for i in range(n) if slot[i]])
        while any(slot):
            actifs = [i for i in range(n) if slot[i]]
            # 1. interrogation de la politique pour les environnements en début de bloc
            q = [i for i in actifs if slot[i]["t"] % H_EXEC == 0]
            if q:
                lot = construire_lot(politique, [slot[i] for i in q], device)
                blocs = denormaliser(politique.agir(lot, gen)).cpu().numpy()
                for i, b in zip(q, blocs):
                    slot[i]["bloc"] = b[:H_EXEC]
            # 2. un pas de simulation pour tous
            for i in actifs:
                s = slot[i]
                a = s["bloc"][s["t"] % H_EXEC]
                s["t"] += 1
                venv.conns[i].send(("step", (a, _a_rendre(s["t"], n_obs))))
            recevoir(actifs)
            for i in actifs:
                if slot[i]["fini"]:
                    resultats[slot[i]["ep"]] = resumer(slot[i]["info"], slot[i]["L"]) | {
                        "contacts_rates": slot[i]["info"]["contacts_rates"]}
                    demarrer(i)
            nouveaux = [i for i in actifs if slot[i] and slot[i]["t"] == 0]
            if nouveaux:
                recevoir(nouveaux)
    finally:
        venv.fermer()
    return resultats


def construire_lot(politique, slots, device):
    """Entrées de la politique à partir des tampons d'historique (grille absolue)."""
    t = torch.tensor([s["t"] for s in slots], device=device)
    tau, vide = indices_grille(t, K_MAX)
    D = next(iter(slots[0]["emb"].values())).shape[-1]
    hist = torch.zeros(len(slots), K_MAX, D, device=device)
    hist_etat = torch.zeros(len(slots), K_MAX, 3, device=device)
    for b, s in enumerate(slots):
        k = (~vide[b]).nonzero()[:, 0]
        taus = tau[b, k].tolist()
        hist[b, k] = torch.stack([s["emb"][tk] for tk in taus])
        hist_etat[b, k] = torch.as_tensor(np.stack([s["etats"][tk] for tk in taus]), device=device)
    obs_t = [[max(0, s["t"] - 1), s["t"]][-politique.n_obs:] for s in slots]
    etat = torch.as_tensor(np.array([[s["etats"][u] for u in o] for s, o in zip(slots, obs_t)]), device=device)
    cur = torch.stack([torch.stack([s["emb"][u] for u in o]) for s, o in zip(slots, obs_t)])
    seq = torch.tensor([s["sequence"] + [-1] * (4 - len(s["sequence"])) for s in slots], device=device)
    return {"cur": cur, "etat": normaliser(etat.float()), "hist": hist, "hist_etat": normaliser(hist_etat),
            "age": (t[:, None] - tau).float() / FPS, "vide": vide,
            "oracle": oracle_onehot(seq, (seq >= 0).sum(1))}
