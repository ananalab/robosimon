"""Répartition de la grille E1 + E2 (section 6.2) sur les machines du PPTI.
Chaque machine a ses données (une valeur de L) et deux files de travail en parallèle.
python grille.py  -> affiche les commandes de lancement."""
MACHINES = {
    "ppti-14-502-01": (2, [("ACT", "B0"), ("DP", "B0"), ("ACT", "B1"), ("DP", "B1"), ("ACT", "M1"), ("DP", "M1")], (0,)),
    "ppti-14-407-02": (2, [("ACT", "B0"), ("DP", "B0"), ("ACT", "B1"), ("DP", "B1"), ("ACT", "M1"), ("DP", "M1")], (1, 2)),
    "ppti-14-502-04": (4, [("ACT", "B0"), ("DP", "B0"), ("ACT", "B1"), ("DP", "B1"), ("ACT", "M1"), ("DP", "M1")], (0,)),
    "ppti-14-407-03": (4, [("ACT", "B0"), ("DP", "B0"), ("ACT", "B1"), ("DP", "B1"), ("ACT", "M1"), ("DP", "M1")], (1, 2)),
    # L = 3 : toutes les politiques M1 et M1p sur la même machine (diagnostics)
    "ppti-14-502-02": (3, [("ACT", "M1"), ("DP", "M1"), ("ACT", "M1p"), ("DP", "M1p")], (0, 1, 2)),
    "ppti-14-502-03": (3, [("ACT", "B0"), ("DP", "B0"), ("ACT", "B1"), ("DP", "B1")], (0, 1, 2)),
}


def files(machine):
    L, conds, graines = MACHINES[machine]
    specs = [f"{'E2' if v == 'M1p' else 'E1'}:{f}:{v}:{L}:{g}" for g in graines for f, v in conds]
    # deux files équilibrées : ACT et DP alternés
    return specs[0::2], specs[1::2]


def toutes(L):
    """Toutes les conditions d'une longueur L, dans l'ordre inverse des files fixes : les
    machines ajoutées prennent la grille par la fin (les conditions réservées sont sautées)."""
    specs = [f"E1:{f}:{v}:{L}:{g}" for g in (0, 1, 2) for v in ("B0", "B1", "M1") for f in ("ACT", "DP")]
    if L == 3:
        specs += [f"E2:{f}:M1p:3:{g}" for g in (0, 1, 2) for f in ("ACT", "DP")]
    return specs[::-1]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(" ".join(toutes(int(sys.argv[1]))))
        sys.exit()
    for m in MACHINES:
        for k, f in enumerate(files(m)):
            print(m, f"file{k}", " ".join(f))
