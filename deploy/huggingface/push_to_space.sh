#!/usr/bin/env bash
# Push SAI-CNS to a Hugging Face Space.
#
# Run this yourself: it needs a Hugging Face access token, which is a credential and is not something
# to hand to an assistant. Create one at https://huggingface.co/settings/tokens with WRITE scope, and
# let git prompt you for it, or use the huggingface-cli login flow. Do not paste it into a file.
#
# What this does that a plain push does not: the model binaries are excluded by .gitignore, so a
# normal push deploys an application with no models. They are force-added here and tracked with
# git-lfs, in the Space clone only, because the GitHub repository's free LFS quota is 1 GB and the
# binaries are 0.78 GB.
set -euo pipefail

USER="${1:-Krishnag999}"
SPACE="${2:-sai-cns}"
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${TMPDIR:-/tmp}/hf-${SPACE}"

echo "source:      $SRC"
echo "destination: https://huggingface.co/spaces/${USER}/${SPACE}"
echo

command -v git-lfs >/dev/null || { echo "git-lfs is not installed: https://git-lfs.com"; exit 1; }

# Create the Space in the browser first, SDK = Docker, visibility = Public.
rm -rf "$WORK"
git clone "https://huggingface.co/spaces/${USER}/${SPACE}" "$WORK"
cd "$WORK"
git lfs install --local

# the application, its models, and the artefacts the interface reads
for p in app.py api.py serve.py requirements.txt Dockerfile .streamlit src models_rf assets results docs; do
    [ -e "$SRC/$p" ] && cp -r "$SRC/$p" .
done
cp "$SRC/deploy/huggingface/README.md" README.md     # the Space README carries the YAML frontmatter

git lfs track "models_rf/**/*.joblib" "models_rf/*.pkl" "assets/*.png"
git add .gitattributes
git add -f models_rf                                  # -f: .gitignore excludes *.joblib
git add -A

echo
echo "about to publish the following to a PUBLIC Space:"
git status --short | head -20
echo "  ... $(git status --porcelain | wc -l) paths in total"
echo
read -r -p "publish? [y/N] " ok
[ "$ok" = "y" ] || { echo "aborted"; exit 1; }

git commit -m "SAI-CNS: web server and REST API"
git push
echo
echo "done. The Space will build for several minutes, then be live at:"
echo "  https://${USER}-${SPACE}.hf.space"
echo
echo "check it with:  curl -fsS https://${USER}-${SPACE}.hf.space/_stcore/health"
