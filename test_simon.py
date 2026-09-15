"""Tests unitaires de l'environnement Simon (section 9.4). Lancer : python test_simon.py"""
import numpy as np
import mujoco
from simon_sim import (SimonEnv, chronologie, episode_expert, REPOS, T_REPOS, D_ON, D_OFF, DELAI,
                       D_GO, D_FB, COURSE, PRESENTATION, DELAI_PH, DEPART, EXECUTION, REPOS_PH)


def test_chronologie():
    for L in (2, 3, 4):
        env = SimonEnv(rendu_actif=False)
        env.reset(0, {"sequence": [0] * L, "rendu": False})
        fin_pres = T_REPOS + L * (D_ON + D_OFF)
        attendu = {0: REPOS_PH, T_REPOS: PRESENTATION, fin_pres: DELAI_PH,
                   fin_pres + DELAI: DEPART, fin_pres + DELAI + D_GO: EXECUTION}
        for t, ph in attendu.items():
            assert env.phase(t) == ph and (t == 0 or env.phase(t - 1) != ph), (L, t)
        # chaque allumage dure exactement D_ON images ; GO dure D_GO images
        allume = np.array([env.etat_led(t) for t in range(env.debut_exec + 5)])
        assert allume[:, :4].sum() == L * D_ON and allume[:, 4].sum() == D_GO
        assert chronologie([1, 2])[1] == T_REPOS + 2 * (D_ON + D_OFF) + DELAI + D_GO


def _forcer_piston(env, b, n_pas):
    """Enfonce le piston b par une force directe (sans le doigt) pendant n_pas images, puis le relâche."""
    for k in range(2 * n_pas):
        env.data.qfrc_applied[env.qadr_piston[b]] = -3.0 if k < n_pas else 0.0
        obs, r, fini, _, info = env.step(REPOS, rendu=False)
        if fini:
            return info
    return info


def test_pression_unique_et_echecs():
    env = SimonEnv(rendu_actif=False)
    env.reset(1, {"sequence": [2, 0, 1], "rendu": False})
    while env.t < env.debut_exec:
        env.step(REPOS, rendu=False)
    info = _forcer_piston(env, 2, 5)
    assert [b for _, b in info["pressions"]] == [2] and info["resultat"] is None, info
    # rétroaction : le bouton pressé est allumé pendant D_FB images
    debut = int(np.ceil(info["pressions"][0][0]))
    assert all(env.etat_led(t)[2] == 1 for t in range(debut, debut + D_FB))
    assert env.etat_led(debut + D_FB)[2] == 0
    # mauvaise pression : échec (répétition du bouton précédent)
    info = _forcer_piston(env, 2, 5)
    assert info["resultat"] == "echec" and info["erreur"] == "repetition", info
    # pression pendant la présentation : anticipation
    env.reset(1, {"sequence": [2, 0, 1], "rendu": False})
    for _ in range(T_REPOS + 3):
        env.step(REPOS, rendu=False)
    info = _forcer_piston(env, 3, 5)
    assert info["resultat"] == "echec" and info["erreur"] == "anticipation", info


def test_repos_ne_masque_aucune_led():
    """Image avec toutes les LED allumées : identique avec ou sans le doigt en pose de repos."""
    env = SimonEnv()
    env.reset(0, {"sequence": [0, 1], "rendu": False})
    for i in range(5):
        env.model.geom_rgba[env.g_led[i], 3] = 1
        env.model.geom_rgba[env.g_led[i], :3] = 1
    mujoco.mj_forward(env.model, env.data)
    avec = env._obs(True)["image"].astype(int)
    env.data.qpos[:3] = (0.5, 0.5, 0.5)   # doigt hors champ
    mujoco.mj_forward(env.model, env.data)
    sans = env._obs(True)["image"].astype(int)
    # pixels qui changent : uniquement ceux du doigt, jamais ceux des LED blanches
    blanc = (sans > 240).all(-1)
    assert blanc.sum() > 50 and (np.abs(avec - sans).sum(-1)[blanc] == 0).all()


def test_determinisme():
    a = episode_expert(7, [3, 1, 2], rendu=True)
    b = episode_expert(7, [3, 1, 2], rendu=True)
    assert all((x == y).all() for x, y in zip(a[0], b[0])) and (a[1] == b[1]).all()
    assert a[3][-1]["resultat"] == "succes"


def test_expert_immobile_avant_depart():
    _, etats, actions, infos = episode_expert(11, [1, 1, 0, 2])
    avant = [i for i, inf in enumerate(infos[:-1]) if inf["t"] < inf["debut_exec"]]
    assert np.allclose(actions[avant], REPOS) and np.abs(etats[avant] - REPOS).max() < 1e-4


if __name__ == "__main__":
    for nom, f in list(globals().items()):
        if nom.startswith("test_"):
            f()
            print("ok", nom)
