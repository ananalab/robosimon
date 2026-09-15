# Préenregistrement (figé avant E1)

## Hypothèses

| Id | Hypothèse | Prédiction mesurable | Résultat qui la réfute |
|---|---|---|---|
| H0 | Le banc ne laisse fuir aucune information sur la séquence hors de la phase de présentation | B0 réussit au niveau du hasard, 4^-L | B0 significativement au-dessus du hasard |
| H1 | Un historique d'images suffit lorsque la séquence est courte, mais l'écart avec l'oracle grandit avec L | M1 dépasse nettement B0 à L = 3 ; B1 − M1 plus grand à L = 4 qu'à L = 2 | M1 au niveau de B0, ou écart à l'oracle constant avec L |
| H2 | L'état passé du robot dans l'historique provoque le copycat, surtout pour ACT | M1p : indice d'inertie plus élevé et succès plus faible que M1 ; effet plus marqué pour ACT | Indice d'inertie inchangé, ou succès de M1p ≥ M1 |

## Contrastes testés (chacun pour ACT et pour DP)

| Id | Contraste | Hypothèse |
|---|---|---|
| C1 | M1 − B0 à L = 3 | H1 |
| C2 | (B1 − M1)_{L=4} − (B1 − M1)_{L=2} | H1 |
| C3 | M1p − M1 à L = 3 (succès et indice d'inertie) | H2 |
| C4 | (M1p − M1)_ACT − (M1p − M1)_DP | H2 |

Différences en points de pourcentage, IC 95 % par bootstrap à deux facteurs (graines et
épisodes, 10 000 tirages ; épisodes appariés entre méthodes). Pas de correction pour
comparaisons multiples. Taille d'effet d'intérêt : 10 points.

Règles de décision : H1 « confirmée » si la borne basse de C1 dépasse 10 points et celle de C2
dépasse 0 ; « réfutée » si la borne haute de C1 est sous 10 points ou celle de C2 sous 0 ;
sinon non concluante. H2 « confirmée » si C3 (inertie) > 0 et C3 (succès) < 0 (IC excluant 0) ;
« réfutée » si C3 (inertie) ≤ 0 ou C3 (succès) ≥ 0 (IC) ; sinon non concluante.

## Listes de test

Fichiers `listes/listes_L{2,3,4}.json` (empreintes SHA-1 ci-dessous).

    d6db6b43e48de727ec2cc0cda952e37f0137620f  listes/listes_L2.json
    bdefad2b9bb5ee05b9eadaf9b7a8557b80a796c6  listes/listes_L3.json
    b8dffa82398c67818478a472bba4d60f8324a824  listes/listes_L4.json
