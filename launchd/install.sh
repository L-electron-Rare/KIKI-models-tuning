#!/usr/bin/env bash
set -euo pipefail

: "${MINIO_ACCESS_KEY:?env required}"
: "${MINIO_SECRET_KEY:?env required}"
: "${HITL_TRAINER_TOKEN:?env required}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="$REPO_DIR/.venv/bin/python"
HOME_DIR="$HOME"
TARGET="$HOME/Library/LaunchAgents/cc.ailiance.hitl-trainer.plist"

mkdir -p "$HOME/Library/LaunchAgents" "$HOME/.config/hitl-trainer" "$HOME/Library/Logs"

sed \
  -e "s|__VENV_PYTHON__|$VENV_PYTHON|g" \
  -e "s|__REPO_DIR__|$REPO_DIR|g" \
  -e "s|__HOME__|$HOME_DIR|g" \
  -e "s|__MINIO_ACCESS_KEY__|$MINIO_ACCESS_KEY|g" \
  -e "s|__MINIO_SECRET_KEY__|$MINIO_SECRET_KEY|g" \
  -e "s|__HITL_TRAINER_TOKEN__|$HITL_TRAINER_TOKEN|g" \
  "$REPO_DIR/launchd/cc.ailiance.hitl-trainer.plist.template" > "$TARGET"

if [ ! -f "$HOME/.config/hitl-trainer/state.json" ]; then
  echo '{"last_trained_key": null}' > "$HOME/.config/hitl-trainer/state.json"
fi

launchctl unload "$TARGET" 2>/dev/null || true
launchctl load -w "$TARGET"
echo "loaded: $TARGET"
launchctl list | grep hitl-trainer || true
