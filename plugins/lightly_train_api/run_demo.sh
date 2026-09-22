#!/usr/bin/env bash
#
# Runs the LightlyTrain API in the background and opens LightlyStudio with the two
# plugin operators on the example images, indexed without annotations.
#
#   plugins/lightly_train_api/run_demo.sh
#
# The API and the Studio database live in .demo/. Ctrl-C stops both processes.

set -euo pipefail
# Job control, so the whole API process tree can be stopped through its process group.
set -m

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_DIR="$(cd "$PLUGIN_DIR/../../lightly-train-api" && pwd)"
WORK_DIR="$PLUGIN_DIR/.demo"
API_URL="http://127.0.0.1:8000"
API_LOG="$WORK_DIR/api.log"
STUDIO_PYTHON="$PLUGIN_DIR/.venv/bin/python"

mkdir -p "$WORK_DIR"

echo "==> Syncing the LightlyTrain API environment"
uv sync --project "$API_DIR"

echo "==> Installing LightlyStudio and the plugin"
if [ ! -x "$STUDIO_PYTHON" ]; then
    uv venv --python 3.12 "$PLUGIN_DIR/.venv"
fi
uv pip install --quiet --python "$STUDIO_PYTHON" -e "$PLUGIN_DIR"

echo "==> Starting the API, logging to $API_LOG"
(cd "$WORK_DIR" && uv run --project "$API_DIR" python -m lightly_train_api) \
    </dev/null >"$API_LOG" 2>&1 &
API_PID=$!
# The API stops on SIGTERM, but the embedded Hatchet engine can hang on the way out.
stop() {
    [ -n "${1:-}" ] || return 0
    kill -- "-$1" 2>/dev/null || return 0
    local limit=$((SECONDS + 10))
    while kill -0 -- "-$1" 2>/dev/null; do
        if [ "$SECONDS" -ge "$limit" ]; then
            kill -KILL -- "-$1" 2>/dev/null || true
            return 0
        fi
        sleep 1
    done
}
cleanup() {
    trap - EXIT
    echo "==> Stopping"
    stop "${STUDIO_PID:-}"
    stop "$API_PID"
}
trap cleanup EXIT INT TERM

# The first start downloads and loads the pretrained backbone, which takes a while.
deadline=$((SECONDS + 300))
until curl -sf "$API_URL/health" >/dev/null; do
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "The API stopped during startup:" >&2
        tail -n 20 "$API_LOG" >&2
        exit 1
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "The API is not serving $API_URL/health, see $API_LOG" >&2
        exit 1
    fi
    sleep 2
done
echo "    API ready on $API_URL/docs"

cat <<EOF
==> Starting LightlyStudio, the images come in without annotations
    1. Annotate a few images in the GUI.
    2. Run the "LightlyTrain API training" operator on the view.
    3. Run the "LightlyTrain API inference" operator to predict the rest.
EOF

# In the background, so Ctrl-C reaches the trap instead of waiting for Studio to exit.
cd "$WORK_DIR"
"$STUDIO_PYTHON" "$PLUGIN_DIR/demo_dataset.py" &
STUDIO_PID=$!
wait "$STUDIO_PID" || true
