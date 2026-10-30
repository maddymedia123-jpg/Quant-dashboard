#!/bin/sh
# Push-to-deploy, run by the quant-deploy timer every 2 minutes: if the tracked branch moved on GitHub,
# pull it and install requirements as the quant-worker user, then restart every enabled unit.
# Installed at /opt/quant/deploy.sh; this copy is the record.
set -e
BRANCH=$(cat /opt/quant/deploy-branch 2>/dev/null || echo main)
as_app() { runuser -u quant-worker -- "$@"; }
cd /opt/quant/app
as_app git fetch -q origin "$BRANCH"
[ "$(as_app git rev-parse HEAD)" = "$(as_app git rev-parse "origin/$BRANCH")" ] && exit 0
as_app git checkout -q "$BRANCH" 2>/dev/null || as_app git checkout -q -b "$BRANCH" "origin/$BRANCH"
as_app git reset -q --hard "origin/$BRANCH"
as_app /opt/quant/venv/bin/pip install -q -r requirements.txt
echo "deployed $(as_app git log --oneline -1)"
for unit in quant-worker quant-book quant-mcp; do
  if systemctl is-enabled -q "$unit"; then systemctl restart "$unit"; fi
done
