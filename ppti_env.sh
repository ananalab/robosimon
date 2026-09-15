# Environnement PPTI (section 9.2). À sourcer : . ppti_env.sh
export http_proxy='http://proxy.ufr-info-p6.jussieu.fr:3128'
export https_proxy='http://proxy.ufr-info-p6.jussieu.fr:3128'
export MUJOCO_GL=egl
export TMPD=/temporary/$USER
export UV_CACHE_DIR=$TMPD/uv-cache UV_PYTHON_INSTALL_DIR=$TMPD/uv-python TORCH_HOME=$TMPD/torch
mkdir -p $TMPD
if [ ! -x $TMPD/.venv/bin/python ]; then
  uv venv -q -p 3.12 $TMPD/.venv
  uv pip install -q -p $TMPD/.venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
  uv pip install -q -p $TMPD/.venv/bin/python mujoco zarr numpy scipy scikit-learn pandas pyarrow matplotlib imageio imageio-ffmpeg
  uv pip freeze -p $TMPD/.venv/bin/python > $TMPD/requirements.lock
fi
PY=$TMPD/.venv/bin/python
export ROBOSIMON_DONNEES=$TMPD/donnees
