#!/bin/sh
# File de travail sur une machine ajoutée : données de L puis conditions restantes.
#   travail.sh <rôle 0|1> <L> <conditions...>   (rôle 0 génère les données, rôle 1 attend)
R=$1; L=$2; shift 2
cd ~/robosimon
if [ "$R" = 1 ]; then
  until [ -f /temporary/$USER/requirements.lock ] && [ -f /temporary/$USER/donnees/L${L}_test_g1.npy ]; do sleep 20; done
fi
. ./ppti_env.sh
[ "$R" = 0 ] && nice -n 10 $PY donnees.py 1 $L
nice -n 10 $PY entrainement.py "$@"
