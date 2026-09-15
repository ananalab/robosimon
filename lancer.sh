#!/bin/sh
# Lance une commande dans tmux sur une machine du PPTI, en basse priorité.
#   ./lancer.sh <machine> <session> "<commande python, ex. donnees.py 1 3>"
M=$1; S=$2; shift 2
ssh -o BatchMode=yes $M "who | grep -v \$USER | head -3; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader;
 tmux new -d -s $S 'cd ~/robosimon && . ./ppti_env.sh && nice -n 10 \$PY $* > \$TMPD/$S.log 2>&1'" 2>&1 | grep -v GUROBI
