#!/usr/bin/env bash
set -euo pipefail

: "${HITL_TRAINER_TOKEN:?env required}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="$REPO_DIR/.venv-eval/bin/python"
HOME_DIR="$HOME"
TARGET="$HOME/Library/LaunchAgents/cc.ailiance.hitl-eval-worker.plist"

mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"

sed \
  -e "s|__VENV_PYTHON__|$VENV_PYTHON|g" \
  -e "s|__REPO_DIR__|$REPO_DIR|g" \
  -e "s|__HOME__|$HOME_DIR|g" \
  -e "s|__HITL_TRAINER_TOKEN__|$HITL_TRAINER_TOKEN|g" \
  "$REPO_DIR/launchd/cc.ailiance.hitl-eval-worker.plist.template" > "$TARGET"

launchctl unload "$TARGET" 2>/dev/null || true
launchctl load -w "$TARGET"
echo "loaded: $TARGET"
launchctl list | grep hitl-eval || true
