"""Visuels : figure 1 (scène et chronologie), vidéos MP4 et GIF des démonstrations et des politiques.
  python videos.py figure1
  python videos.py expert
  python videos.py politiques <condition> [<condition> ...]   (ex. E1_ACT_M1_L3_s0 ; sur la machine
                                                              qui détient le point de sauvegarde)
  python videos.py contrefactuel <condition> [<épisode>]
  python videos.py episodes <épisode> <préfixe> <condition> ...  (une vidéo par politique, même épisode)
  python videos.py composer <sortie> <vidéo1.mp4> <vidéo2.mp4> ...  (côte à côte, en local)
"""
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from simon_sim import (SimonEnv, plan_expert, chronologie, t_max, COULEURS, FPS, PRESENTATION,
                       REPOS_PH, DELAI_PH, DEPART, EXECUTION)

RACINE = Path(__file__).parent
VID = Path(os.environ.get("ROBOSIMON_VIDEOS", RACINE / "videos"))
NOM_PHASE = {REPOS_PH: "rest", PRESENTATION: "presentation", DELAI_PH: "delay", DEPART: "GO signal",
             EXECUTION: "execution"}
TXT = {"seq": "sequence:", "vue": "shown:", "presse": "pressed:", "succes": "SUCCESS", "echec": "FAILURE",
       "memoire": "memory", "repetition": "repetition", "motrice": "motor", "blocage": "stall",
       "anticipation": "early press"}
VARIANTE_EN = {"B0": "no memory", "B1": "oracle", "M1": "image history", "M1p": "history + past state"}
try:
    POLICE = ImageFont.truetype("DejaVuSans.ttf", 13)
except OSError:
    POLICE = ImageFont.load_default()


def annoter(img, titre, info, sequence, substitut=None):
    """Bandeau de quatre lignes sous l'image : politique ; temps, phase et issue ; séquence à
    reproduire (et séquence montrée en cas de substitution) ; boutons pressés."""
    h, w, _ = img.shape
    out = Image.new("RGB", (w, h + 80), (252, 252, 251))
    out.paste(Image.fromarray(img), (0, 0))
    d = ImageDraw.Draw(out)
    gris = (82, 81, 78)

    def cases(x, y, couleurs, cadres=None):
        for j, c in enumerate(couleurs):
            d.rectangle([x + 18 * j, y + 1, x + 18 * j + 13, y + 14], fill=tuple(int(255 * v) for v in COULEURS[c]),
                        outline=None if cadres is None else cadres[j], width=2)
        return x + 18 * len(couleurs)

    d.text((6, h + 3), titre, fill=(11, 11, 11), font=POLICE)
    d.text((6, h + 22), f"t = {info['t'] / FPS:4.1f} s   {NOM_PHASE[info['phase']]}", fill=gris, font=POLICE)
    if info["resultat"]:
        txt = TXT["succes"] if info["resultat"] == "succes" else f"{TXT['echec']} ({TXT[info['erreur']]})"
        d.text((w - 6 - d.textlength(txt, font=POLICE), h + 22), txt, font=POLICE,
               fill=(0, 99, 0) if info["resultat"] == "succes" else (227, 73, 72))
    d.text((6, h + 41), TXT["seq"], fill=gris, font=POLICE)
    x = cases(80, h + 41, sequence)
    if substitut:
        d.text((x + 12, h + 41), TXT["vue"], fill=gris, font=POLICE)
        cases(x + 16 + d.textlength(TXT["vue"], font=POLICE), h + 41, substitut)
    d.text((6, h + 60), TXT["presse"], fill=gris, font=POLICE)
    b = [b for _, b in info["pressions"]]
    cases(80, h + 60, b, [(0, 131, 0) if j < len(sequence) and c == sequence[j] else (227, 73, 72)
                          for j, c in enumerate(b)])
    return np.array(out)


def ecrire(images, nom, fps=FPS, gif_pas=2):
    VID.mkdir(exist_ok=True)
    imageio.mimwrite(VID / f"{nom}.mp4", images, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    # GIF : une image sur deux, résolution réduite de moitié
    petites = [np.array(Image.fromarray(im).resize((im.shape[1] // 2 * 2 * 3 // 4, im.shape[0] * 3 // 4)))
               for im in images[::gif_pas]]
    imageio.mimwrite(VID / f"{nom}.gif", petites, duration=1000 * gif_pas / fps, loop=0)
    print("écrit", VID / f"{nom}.mp4")


# -------------------------------------------------------------------- expert
def expert(seed=4200007, sequence=(2, 0, 3, 1)):
    env = SimonEnv()
    obs, info = env.reset(seed, {"sequence": list(sequence)})
    plan, _ = plan_expert(env, np.random.default_rng(seed + 10_000_000))
    images, fini = [], False
    while not fini:
        images.append(annoter(obs["image"], "Scripted expert", info, list(sequence)))
        obs, _, fini, _, info = env.step(plan[env.t])
    images += [annoter(obs["image"], "Scripted expert", info, list(sequence))] * FPS
    ecrire(images, "expert_L4")


def figure1(seed=3000011, sequence=(1, 3, 0)):
    """Figure 1 : rendus aux instants clés et chronogramme des LED d'un épisode L = 3."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    env = SimonEnv()
    obs, info = env.reset(seed, {"sequence": list(sequence)})
    plan, _ = plan_expert(env, np.random.default_rng(seed + 10_000_000))
    images, infos, fini = [obs["image"]], [info], False
    while not fini:
        obs, _, fini, _, info = env.step(plan[env.t])
        images.append(obs["image"])
        infos.append(info)
    instants = {NOM_PHASE[REPOS_PH]: 15, f"{NOM_PHASE[PRESENTATION]} (1)": 30 + 9,
                f"{NOM_PHASE[PRESENTATION]} (3)": 30 + 54 + 9, NOM_PHASE[DELAI_PH]: 150,
                NOM_PHASE[DEPART]: env.debut_exec - 3}
    press = [int(np.ceil(t)) + 2 for t, _ in infos[-1]["pressions"]]
    appui = "press"
    instants[f"{NOM_PHASE[EXECUTION]} ({appui} 1)"] = press[0]
    instants[f"{NOM_PHASE[EXECUTION]} ({appui} 3)"] = min(press[-1], len(images) - 1)
    fig = plt.figure(figsize=(11, 3.6), facecolor="white")
    for k, (titre, t) in enumerate(instants.items()):
        ax = fig.add_axes([0.01 + k * 0.141, 0.5, 0.135, 0.38])
        ax.imshow(images[t])
        ax.set_title(f"{titre}\nt = {t / FPS:.1f} s", fontsize=8, color="#52514e")
        ax.axis("off")
    ax = fig.add_axes([0.06, 0.1, 0.9, 0.34], facecolor="white")
    T = len(images)
    for i in range(5):
        on = np.array([inf["led"][i] for inf in infos])
        ax.fill_between(np.arange(T) / FPS, i - 0.38 * on, i + 0.38 * on, step="post",
                        color=COULEURS[i] if i < 4 else "#898781", linewidth=0)
    for t, b in infos[-1]["pressions"]:
        ax.plot(t / FPS, b, marker="v", color="#0b0b0b", ms=6)
    fin_pres = 30 + 27 * len(sequence)
    for x0, x1, ph in ((0, 30, REPOS_PH), (30, fin_pres, PRESENTATION), (fin_pres, fin_pres + 60, DELAI_PH),
                       (env.debut_exec, T, EXECUTION)):
        nom = NOM_PHASE[ph]
        ax.axvspan(x0 / FPS, x1 / FPS, color="#e1e0d9" if ph in (PRESENTATION, EXECUTION) else "#f3f2ee",
                   zorder=0, alpha=0.6)
        ax.text((x0 + x1) / 2 / FPS, -0.35, nom, ha="center", fontsize=8, color="#52514e")
    ax.set_yticks(range(5), ["red", "green", "blue", "yellow", "GO"], fontsize=8)
    ax.set_ylim(5.1, -0.6)
    ax.set_xlim(0, T / FPS)
    ax.set_xlabel("time (s)  —  ▼ press detected (expert)", fontsize=8, color="#52514e")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    (RACINE / "figures").mkdir(exist_ok=True)
    sortie = RACINE / "figures"
    sortie.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(sortie / f"scene_timeline.{ext}", dpi=200, bbox_inches="tight")


# ---------------------------------------------------------------- politiques
def teaser():
    """Figure d'ouverture du rapport : la présentation d'une séquence, puis l'issue du même épisode de
    test pour quatre politiques (images des vidéos, bandeau retiré)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def image(f, t=None):                          # t = None : dernière image
        r = imageio.get_reader(VID / f)
        return r.get_data(r.count_frames() - 1 if t is None else t)[:240]
    haut = [(image("act_history.mp4", t), titre) for t, titre in ((39, "1st light"), (66, "2nd light"), (93, "3rd light"),
                                                     (150, "delay"), (173, "GO signal"))]
    bas = []
    for f, titre in (("act_no_memory.mp4", "ACT, no memory\nstalls"), ("act_history.mp4", "ACT, image history\nsuccess"),
                     ("dp_history_20k.mp4", "DP, 20k iterations\nmemory error"),
                     ("dp_history_60k.mp4", "DP, 60k iterations\nsuccess")):
        bas.append((image(f), titre))
    fig = plt.figure(figsize=(10, 4.1), facecolor="white")
    for k, (im, titre) in enumerate(haut):
        ax = fig.add_axes([0.005 + k * 0.199, 0.56, 0.19, 0.36])
        ax.imshow(im)
        ax.set_title(titre, fontsize=9, color="#52514e")
        ax.axis("off")
    for k, (im, titre) in enumerate(bas):
        ax = fig.add_axes([0.005 + k * 0.249, 0.0, 0.24, 0.45])
        ax.imshow(im)
        ax.set_title(titre, fontsize=9, color="#52514e")
        ax.axis("off")
    for ext in ("pdf", "png"):
        fig.savefig(RACINE / "figures" / f"teaser.{ext}", dpi=200, bbox_inches="tight")


def collage(sortie, cases, colonnes=2, pas=2, largeur=280):
    """GIF en mosaïque de plusieurs vidéos (bandeau retiré, cases recadrées en 16:10), avec un badge
    d'issue et une étiquette par case. cases : liste de (fichier .mp4, étiquette, réussite ou None)."""
    lecteurs = [iter(imageio.get_reader(VID / f)) for f, _, _ in cases]
    recadrer = lambda im: im[20:220]                               # 320 x 200, soit 16:10
    derniere = [recadrer(next(r)) for r in lecteurs]
    hauteur = largeur * 10 // 16
    images, i, actifs = [], 0, True
    while actifs:
        if i % pas == 0:
            mosaique = Image.new("RGB", (colonnes * largeur + colonnes - 1,
                                         -(-len(cases) // colonnes) * (hauteur + 1) - 1), (255, 255, 255))
            for k, (im, (_, etiquette, ok)) in enumerate(zip(derniere, cases)):
                tuile = Image.fromarray(im).resize((largeur, hauteur))
                d = ImageDraw.Draw(tuile)
                x = 5
                if ok is not None:
                    badge = "success" if ok else "failure"
                    lb = d.textlength(badge, font=POLICE)
                    d.rectangle([x, 5, x + lb + 8, 23], fill=(0, 131, 70) if ok else (200, 50, 50))
                    d.text((x + 4, 7), badge, fill=(255, 255, 255), font=POLICE)
                    x += lb + 12
                le = d.textlength(etiquette, font=POLICE)
                d.rectangle([x, 5, x + le + 8, 23], fill=(30, 30, 30))
                d.text((x + 4, 7), etiquette, fill=(255, 255, 255), font=POLICE)
                mosaique.paste(tuile, ((k % colonnes) * (largeur + 1), (k // colonnes) * (hauteur + 1)))
            images.append(np.array(mosaique))
        i += 1
        actifs = False
        for k, r in enumerate(lecteurs):
            try:
                derniere[k] = recadrer(next(r))
                actifs = True
            except StopIteration:
                pass
    imageio.mimwrite(sortie, images, duration=1000 * pas / FPS, loop=0)


def jouer(modele, enc, ep, titre, device, substitut=None, graine=0):
    """Un épisode en boucle fermée, image par image, avec la même logique que evaluation.py
    (interrogation toutes les H_EXEC images, grille absolue d'historique)."""
    import torch
    from evaluation import construire_lot, _rendre
    from politiques import H_EXEC, denormaliser
    env = SimonEnv()
    obs, info = env.reset(ep["graine"], {"sequence": ep["sequence"], "rendu": False})
    env.substitut = substitut
    gen = torch.Generator(device=device).manual_seed(graine)
    slot = {"t": 0, "emb": {}, "etats": {}, "sequence": ep["sequence"]}
    images, fini = [], False
    while True:
        img = _rendre(env)
        slot["emb"][env.t] = enc(img[None])[0].float()
        slot["etats"][env.t] = obs["etat"]
        slot["t"] = env.t
        images.append(annoter(img, titre, info, ep["sequence"], substitut))
        if fini:
            break
        if env.t % H_EXEC == 0:
            with torch.no_grad():
                bloc = denormaliser(modele.agir(construire_lot(modele, [slot], device), gen))[0].cpu().numpy()
        obs, _, fini, _, info = env.step(bloc[env.t % H_EXEC], rendu=False)
    return images + [images[-1]] * FPS, info


def charger_modele(nom):
    import torch
    from donnees import DEVICE, Encodeur
    from entrainement import CKPT, G
    from politiques import creer
    exp, fam, var, L, s = nom.split("_")[:5]
    modele = creer(fam, var, 512 * G * G).to(DEVICE)
    modele.load_state_dict(torch.load(CKPT / nom / "retenu.pt", map_location=DEVICE))
    modele.eval()
    return modele, Encodeur(G).to(DEVICE), int(L[1:]), DEVICE


def politiques(noms, n_ep=3):
    from donnees import listes
    for nom in noms:
        modele, enc, L, dev = charger_modele(nom)
        for k, ep in enumerate(listes(L)["test"][:n_ep]):
            images, info = jouer(modele, enc, ep, nom.replace("_", " "), dev)
            ecrire(images, f"{nom}_ep{k}_{info['resultat']}")


def contrefactuel(nom, k=0):
    """Même épisode, présentation remplacée par une autre séquence (test de substitution)."""
    from donnees import listes
    modele, enc, L, dev = charger_modele(nom)
    ep = listes(L)["test"][k]
    s2 = [(c + 1) % 4 for c in ep["sequence"]]
    images, info = jouer(modele, enc, ep, titre_politique(nom), dev, substitut=s2)
    ecrire(images, f"{nom}_contrefactuel_ep{k}")


def titre_politique(nom):
    fam, var = nom.split("_")[1:3]
    k_it = nom.split("_it")[1][:-3] + "k" if "_it" in nom else "20k"
    return f"{fam}, {VARIANTE_EN[var]}, {k_it} iter."


def episodes_video(noms, k, sortie):
    """Une vidéo par politique sur l'épisode de test k (chaque machine ne détient que ses modèles)."""
    from donnees import listes
    for nom in noms:
        modele, enc, L, dev = charger_modele(nom)
        ep = listes(L)["test"][k]
        images, info = jouer(modele, enc, ep, titre_politique(nom), dev)
        ecrire(images, f"{sortie}_{nom}")


def composer(fichiers, sortie):
    """Assemble côte à côte des vidéos déjà écrites (même épisode, durées différentes)."""
    lecteurs = [iter(imageio.get_reader(VID / f)) for f in fichiers]   # lecture en flux (mémoire)
    derniere = [next(r) for r in lecteurs]
    mp4 = imageio.get_writer(VID / f"{sortie}.mp4", fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    gif, i, actifs = [], 0, True
    while actifs:
        im = np.concatenate(derniere, 1)
        mp4.append_data(im)
        if i % 3 == 0:
            gif.append(np.array(Image.fromarray(im).resize((im.shape[1] * 3 // 4, im.shape[0] * 3 // 4))))
        i += 1
        actifs = False
        for k, r in enumerate(lecteurs):
            try:
                derniere[k] = next(r)
                actifs = True
            except StopIteration:
                pass
    mp4.close()
    imageio.mimwrite(VID / f"{sortie}.gif", gif, duration=100, loop=0)
    print("écrit", VID / f"{sortie}.mp4")




if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "figure1":
        figure1()
    elif cmd == "teaser":
        teaser()
    elif cmd == "expert":
        expert()
    elif cmd == "politiques":
        politiques(sys.argv[2:])
    elif cmd == "contrefactuel":
        contrefactuel(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    elif cmd == "episodes":
        episodes_video(sys.argv[4:], int(sys.argv[2]), sys.argv[3])
    elif cmd == "composer":
        composer(sys.argv[3:], sys.argv[2])
