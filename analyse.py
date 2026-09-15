"""Analyse statistique, tableaux LaTeX et figures du rapport.
Lit uniquement resultats/ ; écrit figures/*.{pdf,png}, report/tables/*.tex (dont numbers.tex,
les valeurs citées dans le texte) et resultats/synthese.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RACINE = Path(__file__).parent
RES, FIG, TAB = RACINE / "resultats", RACINE / "figures", RACINE / "report" / "tables"
N_BOOT = 10_000
FAMILLES = ("ACT", "DP")
COUL = {"B0": "#898781", "B1": "#eb6834", "M1": "#2a78d6", "M1p": "#6250d6", "M1_60k": "#1baf7a", "M1p_60k": "#0f7a55"}
MARQ = {"B0": "s", "B1": "o", "M1": "^", "M1p": "v", "M1_60k": "D", "M1p_60k": "P"}
NOM = {"B0": "B0 (no memory)", "B1": "B1 (oracle)", "M1": "M1 (image history)", "M1p": "M1p (history + past state)",
       "M1_60k": "M1, 60k iterations", "M1p_60k": "M1p, 60k iterations"}
TXT, TXT2, MUET, GRILLE, FOND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"
ERREURS = ("memoire", "repetition", "motrice", "blocage", "anticipation")
ERREURS_EN = ("memory", "repetition", "motor", "stall", "early press")

plt.rcParams.update({"figure.facecolor": FOND, "axes.facecolor": FOND, "axes.edgecolor": MUET,
                     "axes.labelcolor": TXT2, "xtick.color": TXT2, "ytick.color": TXT2, "text.color": TXT,
                     "axes.grid": True, "grid.color": GRILLE, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2, "lines.markersize": 7,
                     "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42})


# ------------------------------------------------------------------- données
def charger_episodes(avec_validation=False):
    df = pd.concat([pd.read_parquet(f) for f in sorted((RES / "episodes").glob("*.parquet"))])
    df["n_iter"] = df["n_iter"].fillna(20_000).astype(int) if "n_iter" in df else 20_000
    return df if avec_validation else df[df.partie == "test"]


def charger_diagnostics():
    """Les diagnostics de E1 et E2 ne portent pas le champ « experience » : on le déduit du nom."""
    out = []
    for f in sorted((RES / "diagnostics").glob("*.json")):
        d = json.loads(f.read_text())
        d.setdefault("experience", f.stem.split("_")[0])
        out.append(d)
    return out


def matrice(df, fam, var, L, col="succes"):
    """Tableau (graines, épisodes) ; les épisodes sont alignés par identifiant."""
    d = df[(df.famille == fam) & (df.variante == var) & (df.L == L)]
    if d.empty:
        return None
    return d.pivot_table(index="graine", columns="episode", values=col).to_numpy(float)


def matrice_inertie(diag, fam, var, L=3):
    d = sorted([v for v in diag if v["famille"] == fam and v["variante"] == var and v["L"] == L],
               key=lambda v: v["graine"])
    return np.array([v["inertie_detail"] for v in d], float) if d else None


def bootstrap(termes, n_boot=N_BOOT, graine=0):
    """Contraste linéaire sum(signe * moyenne(M)) avec bootstrap à deux facteurs.
    termes : liste de (M, signe, groupe). Les épisodes d'un même groupe (même liste de test)
    sont rééchantillonnés conjointement (appariement) ; les graines indépendamment par terme."""
    rng = np.random.default_rng(graine)
    groupes = {g: M.shape[1] for M, _, g in termes}
    valeurs = np.empty(n_boot)
    for b in range(n_boot):
        ep = {g: rng.integers(0, n, n) for g, n in groupes.items()}
        v = 0.0
        for M, signe, g in termes:
            s = rng.integers(0, M.shape[0], M.shape[0])
            v += signe * np.nanmean(M[np.ix_(s, ep[g])])
        valeurs[b] = v
    estim = sum(signe * np.nanmean(M) for M, signe, _ in termes)
    return float(estim), [float(x) for x in np.percentile(valeurs, [2.5, 97.5])]


def precision_position(df, fam, var, L):
    """p_j par position, avec IC (bootstrap graines x épisodes)."""
    d = df[(df.famille == fam) & (df.variante == var) & (df.L == L)]
    if d.empty:
        return None
    pos = np.array([[np.nan if x is None else x for x in json.loads(p)] for p in d.pos])
    out = []
    for j in range(L):
        M = d.assign(v=pos[:, j]).pivot_table(index="graine", columns="episode", values="v",
                                              dropna=False).to_numpy(float)
        out.append(bootstrap([(M, 1, "e")], n_boot=2000))
    return out


# ------------------------------------------------------------- mise en forme
def p1(x):
    return f"{100 * x:.1f}"


def ic(r, signe=False):
    """« 46.3 [10.0, 75.0] » pour un résultat de bootstrap."""
    e, (a, b) = r
    f = (lambda v: f"{100 * v:+.1f}") if signe else p1
    return f"{f(e)} [{p1(a)}, {p1(b)}]".replace("-", "$-$")


def pval(p):
    if p >= 1e-3:
        return f"$p={p:.2f}$"
    m, e = f"{p:.1e}".split("e")
    return f"$p={m}\\times10^{{{int(e)}}}$"


def ecrire_table(nom, entete, lignes, colonnes):
    """Tableau booktabs (corps seul ; légende et label dans main.tex)."""
    TAB.mkdir(parents=True, exist_ok=True)
    corps = [f"\\begin{{tabular}}{{{colonnes}}}", "\\toprule", " & ".join(entete) + " \\\\", "\\midrule"]
    corps += [(" & ".join(str(c) for c in l) + " \\\\") if l != "MID" else "\\midrule" for l in lignes]
    corps += ["\\bottomrule", "\\end{tabular}"]
    (TAB / f"{nom}.tex").write_text("\n".join(corps) + "\n")


class Valeurs(dict):
    """Valeurs citées dans le texte du rapport : \\res{cle} dans main.tex."""

    def ecrire(self):
        TAB.mkdir(parents=True, exist_ok=True)
        lignes = ["% Generated by analyse.py; do not edit by hand.",
                  "\\newcommand{\\res}[1]{\\csname res:#1\\endcsname}"]
        lignes += [f"\\expandafter\\def\\csname res:{k}\\endcsname{{{v}}}" for k, v in sorted(self.items())]
        (TAB / "numbers.tex").write_text("\n".join(lignes) + "\n")


V = Valeurs()


# -------------------------------------------------------------------- figures
def sauver(fig, nom):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{nom}.pdf")
    fig.savefig(FIG / f"{nom}.png")
    plt.close(fig)


def courbe(ax, Ls, Ms, cle, dx):
    pts = [(L, bootstrap([(M, 1, "e")], 2000)) for L, M in zip(Ls, Ms) if M is not None]
    if not pts:
        return
    x = np.array([p[0] for p in pts]) + dx
    y = np.array([100 * p[1][0] for p in pts])
    err = np.array([[100 * (p[1][0] - p[1][1][0]), 100 * (p[1][1][1] - p[1][0])] for p in pts]).T
    ax.errorbar(x, y, yerr=err, color=COUL[cle], marker=MARQ[cle], capsize=3, elinewidth=1.2,
                markeredgecolor=FOND, markeredgewidth=1.5, label=NOM[cle])


def fig_succes(df, conf):
    """Succès en fonction de L : variantes de E1, et DP M1 à 60 000 itérations (confirmation)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), sharey=True)
    Ls = [2, 3, 4]
    for ax, fam in zip(axes, FAMILLES):
        ax.plot(Ls, [100 * 4.0 ** -L for L in Ls], ls="--", color=MUET, lw=1.3, label="chance ($4^{-L}$)")
        for k, var in enumerate(("B0", "B1", "M1")):
            courbe(ax, Ls, [matrice(df, fam, var, L) for L in Ls], var, (k - 1.5) * 0.05)
        if fam == "DP":
            courbe(ax, Ls, [matrice(conf, "DP", "M1", L) for L in Ls], "M1_60k", 0.075)
        ax.set_title(fam, color=TXT, fontsize=10, loc="left")
        ax.set_xticks(Ls)
        ax.set_xlabel("sequence length $L$")
        ax.set_ylim(-3, 104)
    axes[0].set_ylabel("test success (%)")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1), fontsize=8)
    sauver(fig, "success_by_length")


def fig_position(df, L=4):
    """Précision par position à L = 4, une courbe par graine (la courbe regroupée mélange des graines
    très différentes : les épisodes qui atteignent les dernières positions viennent de la meilleure)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0), sharey=True)
    for ax, (exp, titre) in zip(axes, (("E1", "DP M1, 20k iterations"), ("C", "DP M1, 60k iterations"))):
        ax.axhline(25, ls="--", color=MUET, lw=1.3)
        ax.annotate("chance", (L, 25), xytext=(4, 3), textcoords="offset points", color=TXT2, fontsize=8)
        d = df[(df.experience == exp) & (df.famille == "DP") & (df.variante == "M1") & (df.L == L)]
        for k, (g, dg) in enumerate(d.groupby("graine")):
            pos = np.array([[np.nan if x is None else x for x in json.loads(p)] for p in dg.pos])
            p = 100 * np.nanmean(pos[:, :L], 0)
            ax.plot(np.arange(1, L + 1), p, marker="o", color=[COUL["M1"], COUL["M1p"], COUL["B1"]][k % 3],
                    markeredgecolor=FOND, label=f"seed {g}: {100 * dg.succes.mean():.0f}% success")
        ax.set_title(titre, color=TXT, fontsize=10, loc="left")
        ax.set_xticks(range(1, L + 1))
        ax.set_xlabel("position $j$ in the sequence")
        ax.set_ylim(-3, 104)
        ax.legend(loc="lower left", fontsize=7.5)
    axes[0].set_ylabel("accuracy $p_j$ given a correct prefix (%)")
    sauver(fig, "serial_position")


def fig_apprentissage(dfv):
    """Perte d'entraînement et succès en validation au fil des itérations, DP à L = 3. La perte de B0,
    B1 et M1 plafonne bien avant que M1 ait appris à utiliser sa mémoire."""
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2))
    ax = axes[0]
    for var, coul in (("B0", COUL["B0"]), ("B1", COUL["B1"]), ("M1", COUL["M1"])):
        motifs = [f"E1_DP_{var}_L3_s*.csv"] + (["A_DP_M1_L3_s*_it60000.csv", "C_DP_M1_L3_s*_it60000.csv"] if var == "M1" else [])
        courbes = [pd.read_csv(f) for m in motifs for f in sorted((RES / "pertes").glob(m))]
        for k, c in enumerate(courbes):
            ax.plot(c["iter"] / 1000, c["perte"].rolling(5, min_periods=1).mean(), color=coul, lw=0.9, alpha=0.55,
                    label=NOM[var] if k == 0 else None)
    ax.axvline(20, color=MUET, ls=":", lw=1)
    ax.annotate("protocol budget", (20, 2.3e-3), xytext=(3, 0), textcoords="offset points", fontsize=8, color=TXT2)
    ax.set_yscale("log")
    ax.set_ylim(2e-3, 3e-2)
    ax.set_xlabel("training iterations (thousands)")
    ax.set_ylabel("denoising loss (MSE)")
    ax.legend(loc="upper right", fontsize=7.5)
    ax = axes[1]
    for fam, exps, coul, nom in (("ACT", ("E1",), COUL["B1"], "ACT M1"), ("DP", ("E1", "X", "A", "C"), COUL["M1"], "DP M1")):
        d = dfv[(dfv.partie == "val") & (dfv.famille == fam) & (dfv.variante == "M1") & (dfv.L == 3)
                & dfv.experience.isin(exps)]
        par = d.groupby(["experience", "graine", "iter"]).succes.mean().reset_index()
        par["k"] = (par["iter"] / 1000).round()
        ax.scatter(par.k + (0.6 if fam == "ACT" else 0), 100 * par.succes, s=12, color=coul, alpha=0.35, edgecolors="none")
        moy = par.groupby("k").succes.mean()
        if fam == "DP":
            V["lcdptwenty"], V["lcdpforty"] = f"{100 * moy[20]:.0f}", f"{100 * moy[40]:.0f}"
            V["lcdpthirteen"] = f"{100 * moy[13]:.0f}"
        ax.plot(moy.index + (0.6 if fam == "ACT" else 0), 100 * moy.values, marker=MARQ["M1"] if fam == "DP" else "o",
                color=coul, markeredgecolor=FOND, label=f"{nom} (mean over seeds)")
    ax.axvline(20, color=MUET, ls=":", lw=1)
    ax.set_xlim(0, 63)
    ax.set_ylim(-3, 104)
    ax.set_xlabel("training iterations (thousands)")
    ax.set_ylabel("validation success at $L=3$ (%)")
    ax.legend(loc="lower right", fontsize=7.5)
    sauver(fig, "learning_curves")


def fig_copycat(df, diag, df_tout, diag_tout):
    """Succès et indice d'inertie par graine, M1 contre M1p à L = 3 : protocole (3 graines),
    DP avec 10 graines (extension) et DP à 60 000 itérations (confirmation)."""
    groupes = [("ACT", ("E1", "E2"), "ACT\n20k, 3 seeds"), ("DP", ("E1", "E2"), "DP\n20k, 3 seeds"),
               ("DP", ("E1", "E2", "X"), "DP\n20k, 10 seeds"), ("DP", ("C",), "DP\n60k, 6 seeds")]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    rng = np.random.default_rng(0)
    for gi, (fam, exps, _) in enumerate(groupes):
        for k, var in enumerate(("M1", "M1p")):
            d = df_tout[(df_tout.famille == fam) & (df_tout.variante == var) & (df_tout.L == 3)
                        & df_tout.experience.isin(exps)]
            succ = d.groupby("graine").succes.mean()
            dv = [x for x in diag_tout if x["famille"] == fam and x["variante"] == var and x["L"] == 3
                  and x["experience"] in exps]
            iner = pd.Series({x["graine"]: x["inertie"] for x in dv})
            for ax, s in zip(axes, (succ, iner)):
                x0 = gi + (k - 0.5) * 0.34
                ax.scatter(x0 + rng.uniform(-0.06, 0.06, len(s)), 100 * s.values, s=16, color=COUL[var], alpha=0.6,
                           edgecolors="none")
                ax.plot([x0 - 0.12, x0 + 0.12], [100 * s.mean()] * 2, color=COUL[var], lw=2.5,
                        label=NOM[var] if gi == 0 else None)
    for ax, t in zip(axes, ("test success at $L=3$ (%)", "inertia index (%)")):
        ax.set_xticks(range(len(groupes)), [g[2] for g in groupes], fontsize=8)
        ax.set_ylabel(t)
        ax.set_ylim(-3, 104)
        ax.grid(axis="x", visible=False)
        for x in (1.5, 2.5):
            ax.axvline(x, color=GRILLE, lw=1)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1), fontsize=8)
    sauver(fig, "copycat")


def fig_sondes(diag):
    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    ax.axvline(62.5, color=MUET, lw=1, ls=":")
    ax.axhline(62.5, color=MUET, lw=1, ls=":")
    for (x, y, t) in ((97, 66, "functional\nmemory"), (97, 12, "under-use"),
                      (22, 12, "encoding\nfailure"), (22, 66, "inconsistent\n(leak?)")):
        ax.text(x, y, t, ha="right" if x > 50 else "left", va="bottom", color=MUET, fontsize=8)
    for fam, ouvert in (("ACT", False), ("DP", True)):
        for var in ("B1", "M1", "M1p"):
            d = [v for v in diag if v["famille"] == fam and v["variante"] == var and "sondes" in v]
            if not d:
                continue
            xs, ys = [100 * v["sondes"][0] for v in d], [100 * v["tsc"] for v in d]
            ax.scatter(xs, ys, s=20, color=COUL[var], alpha=0.35, edgecolors="none")
            ax.scatter(np.mean(xs), np.mean(ys), s=64, marker=MARQ[var], facecolor=FOND if ouvert else COUL[var],
                       edgecolor=COUL[var], linewidth=2, zorder=3)
            dx, dy = {("ACT", "M1"): (-48, -4), ("ACT", "M1p"): (-52, -16), ("ACT", "B1"): (8, -12),
                      ("DP", "B1"): (8, 6)}.get((fam, var), (8, -3))
            ax.annotate(f"{fam} {var}", (np.mean(xs), np.mean(ys)), xytext=(dx, dy), textcoords="offset points",
                        fontsize=8, color=TXT2)
    ax.set_xlim(15, 102)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("linear probe accuracy on $s_1$ (%)")
    ax.set_ylabel("counterfactual following rate (%)")
    sauver(fig, "probes_vs_following")


def fig_inertie(df_tout, diag_tout):
    """DP à L = 3, une graine par point : 20 000 itérations (E1, E2, extension) et 60 000 (confirmation)."""
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    succ_tous = []
    for exps, var, cle in ((("E1", "X"), "M1", "M1"), (("E2", "X"), "M1p", "M1p"),
                           (("C",), "M1", "M1_60k"), (("C",), "M1p", "M1p_60k")):
        dv = [x for x in diag_tout if x["famille"] == "DP" and x["variante"] == var and x["L"] == 3
              and x["experience"] in exps]
        d = df_tout[(df_tout.famille == "DP") & (df_tout.variante == var) & (df_tout.L == 3)
                    & df_tout.experience.isin(exps)]
        succ = d.groupby("graine").succes.mean()
        xs, ys = [100 * x["inertie"] for x in dv], [100 * succ[x["graine"]] for x in dv]
        if cle in ("M1", "M1p"):
            succ_tous += list(zip(xs, ys))
        ax.scatter(xs, ys, s=40, color=COUL[cle], marker=MARQ[cle], edgecolor=FOND, linewidth=1.2,
                   label=f"DP {NOM[cle].split(' (')[0]}" + ("" if "60k" in cle else ", 20k iterations"))
    V["corrinertia"] = f"{np.corrcoef(*zip(*succ_tous))[0, 1]:.2f}".replace("-", "$-$")
    ax.set_xlabel("inertia index (%)")
    ax.set_ylabel("test success at $L=3$ (%)")
    ax.set_ylim(-3, 105)
    ax.legend(loc="center right", fontsize=7.5)
    sauver(fig, "inertia_by_seed")


# -------------------------------------------------------------------- tableaux
def tables_e0(df, e0):
    lignes = []
    ex = e0["expert"]
    lignes.append(["Scripted expert (500 episodes per $L$)", ", ".join(f"{p1(v['succes'])}" for v in ex.values()) + " \\%",
                   "$\\geq 99$\\%", "pass" if all(v["succes"] >= .99 for v in ex.values()) else "fail"])
    lignes.append(["Determinism (same seed and actions)", "bit-identical", "identical", "pass"])
    vis = e0["visibilite"]
    lignes.append(["LED state decoding from $E_h$ (6 classes)", f"{p1(vis['g1'])} \\% ($g=1$)", "$\\geq 99$\\%",
                   f"pass, $g={vis['g_retenu']}$"])
    fu = e0["fuites"]
    for k, nom in (("image", "Leak, random execution frame$^\\dagger$"),
                   ("image_debut_execution", "Leak, first execution frame (finger at rest)"),
                   ("etat", "Leak, finger state at GO"),
                   ("controle_positif_presentation", "Positive control, first presentation frame")):
        ok = fu[k]["absence_de_fuite"] != (k == "controle_positif_presentation")
        lignes.append([nom, f"{p1(fu[k]['precision'])} \\% ({pval(fu[k]['p_valeur'])})",
                       "$>25$\\% expected" if k.startswith("controle") else "not $>25$\\%",
                       "see text" if k == "image" else ("pass" if ok else "fail")])
    for fam in FAMILLES:
        for L in (2, 3):
            r = bootstrap([(matrice(df, fam, "B0", L), 1, "e")])
            lignes.append([f"{fam} B0, $L={L}$", f"{ic(r)} \\%", f"not above {100 * 4.0 ** -L:.1f}\\%",
                           "pass" if r[1][0] <= 4.0 ** -L else "fail"])
        r = bootstrap([(matrice(df, fam, "B1", 3), 1, "e")])
        lignes.append([f"{fam} B1, $L=3$", f"{ic(r)} \\%", "$\\geq 90$\\%", "pass" if r[0] >= .9 else "fail"])
    ecrire_table("e0", ["Check", "Result", "Criterion", "Verdict"], lignes, "llll")
    V["leakrandom"] = p1(fu["image"]["precision"])
    V["leakfirst"] = p1(fu["image_debut_execution"]["precision"])
    V["leakfirstp"] = f"{fu['image_debut_execution']['p_valeur']:.2f}"
    V["leakstate"] = p1(fu["etat"]["precision"])
    for f in sorted((RES / "e0_act_lr1e-5").glob("*.json")):
        V[f"lrlow{f.stem.split('_')[2].lower()}"] = f"{100 * json.loads(f.read_text())['succes_test']:.0f}"


def tables_tests():
    t = json.loads((RES / "test_politiques.json").read_text())
    lignes = [[k.replace("_", " "), f"{v['perte_finale']:.4f}", f"{100 * v['succes_train']:.0f}",
               f"{v['ecart_hist_presentation']:.3f}"] for k, v in t.items()]
    ecrire_table("unit_tests", ["Policy", "Final loss", "Success (\\%)", "$\\max|\\Delta a|$"], lignes, "lrrr")


def tables_e1(df, synth):
    lignes = [["Chance ($4^{-L}$)"] + [f"{100 * 4.0 ** -L:.1f}" for _ in FAMILLES for L in (2, 3, 4)]]
    for var in ("B0", "B1", "M1", "M1p"):
        cell = []
        for fam in FAMILLES:
            for L in (2, 3, 4):
                M = matrice(df, fam, var, L)
                if M is None:
                    cell.append("--")
                    continue
                r = bootstrap([(M, 1, "e")])
                synth["succes"][f"{fam}_{var}_L{L}"] = [r[0], *r[1]]
                V[f"{fam}{var}L{L}".lower()] = p1(r[0])
                cell.append(f"\\makecell{{{p1(r[0])}\\\\[-1pt]{{\\scriptsize [{p1(r[1][0])}, {p1(r[1][1])}]}}}}")
        lignes.append([var] + cell)
    ecrire_table("e1_success", ["Variant"] + [f"{f}, $L={L}$" for f in FAMILLES for L in (2, 3, 4)], lignes, "l" + "c" * 6)
    lignes = []
    for fam in FAMILLES:
        for var in ("B0", "B1", "M1", "M1p"):
            d = df[(df.famille == fam) & (df.variante == var) & (df.L == 3)]
            lignes.append([fam, var, p1(d.succes.mean())] + [p1((d.erreur == e).mean()) for e in ERREURS])
            for e in ERREURS:
                V[f"err{fam}{var}{e}".lower()] = p1((d.erreur == e).mean())
    ecrire_table("outcomes", ["Family", "Variant", "Success"] + [e.capitalize() for e in ERREURS_EN], lignes, "llrrrrrr")
    for fam in FAMILLES:
        for L in (2, 3, 4):
            g = df[(df.famille == fam) & (df.variante == "M1") & (df.L == L)].groupby("graine").succes.mean()
            V[f"seeds{fam}m1L{L}".lower()] = ", ".join(f"{100 * x:.0f}" for x in g)


def tables_contrastes(df, diag, synth):
    m = lambda fam, var, L: matrice(df, fam, var, L)
    mi = lambda fam, var: matrice_inertie(diag, fam, var)
    defs = {
        "C1": ("M1 $-$ B0 at $L=3$", lambda f: [(m(f, "M1", 3), 1, "e"), (m(f, "B0", 3), -1, "e")]),
        "C2": ("$(\\text{B1}-\\text{M1})_{L=4} - (\\text{B1}-\\text{M1})_{L=2}$",
               lambda f: [(m(f, "B1", 4), 1, "L4"), (m(f, "M1", 4), -1, "L4"), (m(f, "B1", 2), -1, "L2"),
                          (m(f, "M1", 2), 1, "L2")]),
        "C3s": ("M1p $-$ M1 at $L=3$, success", lambda f: [(m(f, "M1p", 3), 1, "e"), (m(f, "M1", 3), -1, "e")]),
        "C3i": ("M1p $-$ M1 at $L=3$, inertia", lambda f: [(mi(f, "M1p"), 1, "d"), (mi(f, "M1"), -1, "d")]),
    }
    lignes = []
    for cle, (desc, termes) in defs.items():
        vals = {fam: bootstrap(termes(fam)) for fam in FAMILLES}
        synth["contrastes"][cle] = vals
        for fam, r in vals.items():
            V[f"{cle}{fam}".lower()] = ic(r, True)
        lignes.append([cle[:2], desc, ic(vals["ACT"], True), ic(vals["DP"], True)])
    c4s = bootstrap([(m("ACT", "M1p", 3), 1, "e"), (m("ACT", "M1", 3), -1, "e"),
                     (m("DP", "M1p", 3), -1, "e"), (m("DP", "M1", 3), 1, "e")])
    c4i = bootstrap([(mi("ACT", "M1p"), 1, "d"), (mi("ACT", "M1"), -1, "d"),
                     (mi("DP", "M1p"), -1, "d"), (mi("DP", "M1"), 1, "d")])
    synth["contrastes"]["C4"] = {"succes": c4s, "inertie": c4i}
    V["c4s"], V["c4i"] = ic(c4s, True), ic(c4i, True)
    lignes.append(["C4", "$(\\text{M1p}-\\text{M1})_{\\text{ACT}} - (\\text{M1p}-\\text{M1})_{\\text{DP}}$",
                   f"success {ic(c4s, True)}", f"inertia {ic(c4i, True)}"])
    ecrire_table("contrasts", ["", "Contrast", "ACT", "DP"], lignes, "llcc")


def tables_diagnostics(df, diag):
    lignes = []
    for fam in FAMILLES:
        for var in ("B0", "B1", "M1", "M1p"):
            d = [v for v in diag if v["famille"] == fam and v["variante"] == var]
            ii = p1(np.mean([v["inertie"] for v in d]))
            V[f"inertia{fam}{var}".lower()] = ii
            if "sondes" in d[0]:
                so = np.mean([v["sondes"] for v in d], 0)
                for j, x in enumerate(so):
                    V[f"probe{fam}{var}s{j + 1}".lower()] = f"{100 * x:.0f}"
                V[f"fol{fam}{var}".lower()] = p1(np.mean([v["tsc"] for v in d]))
                lignes.append([fam, var, ii, " / ".join(f"{100 * x:.0f}" for x in so),
                               p1(np.mean([v["tsc"] for v in d])), p1(np.mean([v["p_s1"] for v in d]))])
            else:
                lignes.append([fam, var, ii, "--", "--", "--"])
    ecrire_table("diagnostics", ["Family", "Variant", "Inertia", "Probe $s_1/s_2/s_3$", "Following",
                                 "$P(\\text{1st}=s_1)$"], lignes, "llrcrr")
    lignes = []
    for fam in FAMILLES:
        for var in ("M1", "M1p"):
            for v in sorted([x for x in diag if x["famille"] == fam and x["variante"] == var], key=lambda x: x["graine"]):
                d = df[(df.famille == fam) & (df.variante == var) & (df.L == 3) & (df.graine == v["graine"])]
                lignes.append([fam, var, v["graine"], f"{100 * v['sondes'][0]:.0f}", f"{100 * v['tsc']:.0f}",
                               f"{100 * v['inertie']:.0f}", f"{100 * d.succes.mean():.0f}"])
    ecrire_table("diagnostics_seeds", ["Family", "Variant", "Seed", "Probe $s_1$", "Following", "Inertia", "Success"],
                 lignes, "llrrrrr")


def tables_annexe(df):
    lignes = []
    for fam in FAMILLES:
        for var in ("B0", "B1", "M1", "M1p"):
            for L in (2, 3, 4):
                d = df[(df.famille == fam) & (df.variante == var) & (df.L == L)]
                if len(d):
                    g = d.groupby("graine")
                    lignes.append([fam, var, L] + [f"{100 * x:.0f}" for x in g.succes.mean()] +
                                  [", ".join(f"{int(x) // 1000}k" for x in g["iter"].first())])
    ecrire_table("seeds", ["Family", "Variant", "$L$", "Seed 0", "Seed 1", "Seed 2", "Selected checkpoint"],
                 lignes, "llrrrrl")
    lignes = []
    for nom, motif in (("Protocol (E1, E2)", "E[12]_*.json"), ("Exploratory extension", "X_*.json"),
                       ("DP improvement, pilot", "A_*.json"), ("DP improvement, confirmation", "C_*.json")):
        js = [json.loads(f.read_text()) for f in (RES / "episodes").glob(motif)]
        if js:
            lignes.append([nom, len(js), f"{sum(j['entrainement_s'] for j in js) / 3600:.1f}",
                           f"{sum(j['total_s'] for j in js) / 3600:.1f}"])
    ecrire_table("compute", ["Set", "Conditions", "Training (h)", "Total (h)"], lignes, "lrrr")


def verdicts(synth):
    s, c = synth["succes"], synth["contrastes"]
    v = {"H0": "refuted" if any(a > 4.0 ** -int(k[-1]) for k, (e, a, b) in s.items() if "_B0_" in k) else "confirmed"}
    v["H1"] = {f: ("confirmed" if c["C1"][f][1][0] > 0.10 and c["C2"][f][1][0] > 0 else
                   "refuted" if c["C1"][f][1][1] < 0.10 or c["C2"][f][1][1] < 0 else "inconclusive") for f in FAMILLES}
    v["H2"] = {f: ("confirmed" if c["C3i"][f][1][0] > 0 and c["C3s"][f][1][1] < 0 else
                   "refuted" if c["C3i"][f][1][1] <= 0 or c["C3s"][f][1][0] >= 0 else "inconclusive") for f in FAMILLES}
    return v


def table_exploratoire(df_tout, diag_tout):
    """DP M1 et M1p à L = 3 avec 10 graines (0 à 2 du protocole, 3 à 9 ajoutées), non préenregistré."""
    d = df_tout[df_tout.experience.isin(["E1", "E2", "X"])]
    dg = [x for x in diag_tout if x["experience"] in ("E1", "E2", "X")]
    M = {v: matrice(d, "DP", v, 3) for v in ("M1", "M1p")}
    I = {v: matrice_inertie(dg, "DP", v) for v in ("M1", "M1p")}
    out, lignes = {}, []
    for nom, X, g in (("Success", M, "e"), ("Inertia index", I, "d")):
        r = {v: bootstrap([(X[v], 1, g)]) for v in X}
        diff = bootstrap([(X["M1p"], 1, g), (X["M1"], -1, g)])
        out[nom] = {"M1": r["M1"], "M1p": r["M1p"], "diff": diff}
        lignes.append([nom + " (\\%)", ic(r["M1"]), ic(r["M1p"]), ic(diff, True)])
        V[f"x{nom.split()[0].lower()}m1"], V[f"x{nom.split()[0].lower()}m1p"] = p1(r["M1"][0]), p1(r["M1p"][0])
        V[f"x{nom.split()[0].lower()}diff"] = ic(diff, True)
    for v in ("M1", "M1p"):
        dv = [x for x in dg if x["famille"] == "DP" and x["variante"] == v and x["L"] == 3]
        V[f"xprobe{v}".lower()] = f"{100 * np.mean([x['sondes'][0] for x in dv]):.0f}"
        V[f"xfol{v}".lower()] = f"{100 * np.mean([x['tsc'] for x in dv]):.0f}"
    lignes.append(["Probe $s_1$ / following (\\%)", f"{V['xprobem1']} / {V['xfolm1']}", f"{V['xprobem1p']} / {V['xfolm1p']}", ""])
    ecrire_table("exploratory", ["", "M1 (10 seeds)", "M1p (10 seeds)", "M1p $-$ M1"], lignes, "lccc")
    return out


def table_amelioration(dfv, diag_tout):
    """Pilote (graines 0 à 2, choix sur la validation) puis confirmation sur graines neuves."""
    d = dfv.copy()
    d["methode"] = d.famille + " " + (d.n_iter // 1000).astype(str) + "k"
    pil = pd.concat([d[(d.experience == "E1") & (d.famille == "DP")], d[(d.experience == "A") & (d.graine <= 2)]])
    pil = pil[(pil.variante == "M1") & (pil.L == 3)]
    noms = {"DP 20k": "DP, 20k iterations (E1)", "DPo1 20k": "DP, single observation frame",
            "DPptp 20k": "DP + PTP", "DP 60k": "DP, 60k iterations", "DPptp 60k": "DP + PTP, 60k iterations"}
    out, lignes = {}, []
    for m, nom in noms.items():
        x = pil[pil.methode == m]
        t = x[x.partie == "test"]
        it = t.groupby("graine")["iter"].first()
        v = x[x.partie == "val"]
        val = [v[(v.graine == g) & (v["iter"] == it[g])].succes.mean() for g in it.index]
        r = bootstrap([(t.pivot_table(index="graine", columns="episode", values="succes").to_numpy(float), 1, "e")], 2000)
        out[m] = {"val": float(np.mean(val)), "test": r}
        cle = m.replace(" ", "").lower()
        V[f"pilotval{cle}"], V[f"pilottest{cle}"] = f"{100 * np.mean(val):.0f}", ic(r)
        lignes.append([nom, ", ".join(f"{100 * s:.0f}" for s in val), f"{100 * np.mean(val):.0f}", ic(r)])
    ecrire_table("pilot", ["Candidate", "Validation per seed (\\%)", "Validation mean", "Test (\\%)"], lignes, "lccc")
    conf = d[(d.experience == "C") & (d.partie == "test")]
    lignes = []
    for (var, L), x in conf.groupby(["variante", "L"]):
        r = bootstrap([(x.pivot_table(index="graine", columns="episode", values="succes").to_numpy(float), 1, "e")], 2000)
        dg = [y for y in diag_tout if y["experience"] == "C" and y["variante"] == var and y["L"] == L]
        ii = p1(np.mean([y["inertie"] for y in dg])) if dg else "--"
        V[f"conf{var}L{L}".lower()] = p1(r[0])
        if dg:
            V[f"confinertia{var}".lower()] = ii
        lignes.append([var, L, x.graine.nunique(), ic(r), ii])
    ecrire_table("confirmation", ["Variant", "$L$", "Seeds", "Test success (\\%)", "Inertia (\\%)"], lignes, "lrrcr")
    base = d[(d.experience == "E1") & (d.partie == "test")]
    Mc = matrice(conf, "DP", "M1", 3)
    Mp = matrice(conf, "DP", "M1p", 3)
    out["A1"] = bootstrap([(Mc, 1, "e"), (matrice(base, "DP", "M1", 3), -1, "e")])
    out["A2_succes"] = bootstrap([(Mp, 1, "e"), (Mc, -1, "e")])
    I = {v: matrice_inertie([y for y in diag_tout if y["experience"] == "C"], "DP", v) for v in ("M1", "M1p")}
    out["A2_inertie"] = bootstrap([(I["M1p"], 1, "d"), (I["M1"], -1, "d")])
    V["aone"], V["atwos"], V["atwoi"] = ic(out["A1"], True), ic(out["A2_succes"], True), ic(out["A2_inertie"], True)
    V["confseeds"] = str(conf[conf.variante == "M1"].groupby(["L", "graine"]).ngroups)
    return out


def main():
    df_tout = charger_episodes()
    diag_tout = charger_diagnostics()
    df = df_tout[df_tout.experience.isin(["E1", "E2"])]          # protocole préenregistré seul
    diag = [x for x in diag_tout if x["experience"] in ("E1", "E2")]
    conf = df_tout[df_tout.experience == "C"]
    synth = {"contrastes": {}, "succes": {}}
    fig_succes(df, conf)
    fig_position(df_tout)
    fig_copycat(df, diag, df_tout, diag_tout)
    fig_apprentissage(charger_episodes(avec_validation=True))
    fig_sondes(diag)
    fig_inertie(df_tout, diag_tout)
    tables_e0(df, json.loads((RES / "e0.json").read_text()))
    tables_tests()
    tables_e1(df, synth)
    tables_contrastes(df, diag, synth)
    tables_diagnostics(df, diag)
    tables_annexe(df)
    synth["verdicts"] = verdicts(synth)
    synth["exploratoire"] = table_exploratoire(df_tout, diag_tout)
    synth["amelioration"] = table_amelioration(charger_episodes(avec_validation=True), diag_tout)
    V.ecrire()
    (RES / "synthese.json").write_text(json.dumps(synth, ensure_ascii=False, indent=1))
    print(json.dumps(synth["verdicts"], indent=1))


if __name__ == "__main__":
    main()
