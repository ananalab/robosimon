# Préenregistrement de l'extension « amélioration de DP » (figé avant la confirmation)

## Pilote (sélection sur la validation uniquement, graines 0 à 2, L = 3, variante M1)

Candidats : DP de base (20 000 itérations, E1), DP 60 000 itérations, DPptp (prédiction des
16 actions passées) 20 000 et 60 000 itérations, DPo1 (une seule image d'observation) 20 000.

Règle de choix : meilleur succès moyen en validation (30 épisodes, point de sauvegarde retenu).
Si plusieurs candidats sont à moins de 5 points du meilleur, on retient le plus simple : sans PTP
plutôt qu'avec, puis le moins d'itérations.

## Confirmation (nouvelles graines, jamais utilisées : 10 à 15)

- Variante retenue, M1 et M1p à L = 3, 6 graines chacune ; M1 à L = 2 et L = 4, 3 graines (10 à 12).
- Évaluation sur les mêmes 100 épisodes de test que E1 ; diagnostics identiques.
- Contrastes (bootstrap graines × épisodes, IC 95 %) :
  - A1 : succès, M1 retenue − DP M1 de E1 (graines 0 à 2) à L = 3 ;
  - A2 : succès et indice d'inertie, M1p retenue − M1 retenue à L = 3 (copycat sous la variante retenue).
- Prédiction : A1 > 10 points (borne basse). Pour A2, pas de prédiction directionnelle.
