#!/bin/sh
# Synchronisation avec le PPTI. Le .bashrc distant écrit une ligne sur la sortie standard,
# ce qui casse rsync et scp : on passe par tar et on retire cette première ligne.
#   ./sync.sh push            envoie le code vers ~/robosimon (NFS)
#   ./sync.sh pull <chemin>   rapatrie ~/robosimon/<chemin>
H=${PPTI_HOTE:-ppti}
case "$1" in
  push) COPYFILE_DISABLE=1 tar c --no-xattrs --exclude .venv --exclude donnees --exclude __pycache__ --exclude figures \
          --exclude resultats --exclude videos *.py *.sh *.md 2>/dev/null | ssh -o BatchMode=yes $H 'mkdir -p ~/robosimon && tar x -C ~/robosimon' ;;
  pull) ssh -o BatchMode=yes $H "cd ~/robosimon && tar c $2" | tail -c +15 | tar x ;;
esac
