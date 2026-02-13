wget -qO- https://astral.sh/uv/install.sh | sh

uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate

uv sync

hf download qbration21/verdandi-datasets --repo-type dataset --local-dir ./datasets