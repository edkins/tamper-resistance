set -ex
apt update
apt install -y less foot vim rsync screen
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/root/.local/bin:$PATH"
poetry install
