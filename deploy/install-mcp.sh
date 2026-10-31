#!/bin/sh
# Installs the MCP server unit on the VM, once, as root:
#   sudo sh /opt/quant/app/deploy/install-mcp.sh
# Copies the unit, writes a fresh bearer token to /opt/quant/mcp.env (kept if one is already there),
# refreshes /opt/quant/deploy.sh so deploys restart the unit, enables it and checks it answers.
# Root copies two files out of the worker-owned checkout here; they are installed as plain root-owned
# files (never symlinks) with fixed modes, the same way the README has the other units installed by hand.
set -e
SRC=/opt/quant/app/deploy
for f in quant-mcp.service deploy.sh; do
  [ -f "$SRC/$f" ] && [ ! -L "$SRC/$f" ] || { echo "refusing: $SRC/$f is not a regular file"; exit 1; }
done
install -o root -g root -m 644 "$SRC/quant-mcp.service" /etc/systemd/system/quant-mcp.service
install -o root -g root -m 755 "$SRC/deploy.sh" /opt/quant/deploy.sh
ENV=/opt/quant/mcp.env
[ ! -e "$ENV" ] || { [ -f "$ENV" ] && [ ! -L "$ENV" ]; } || { echo "refusing: $ENV is not a regular file"; exit 1; }
if ! grep -q '^QUANT_MCP_TOKEN=.\{16,\}$' "$ENV" 2>/dev/null; then
  # the secret lands in a fresh 0600 file and is renamed into place: never written into an open-mode file
  install -o quant-worker -g quant-worker -m 600 /dev/null "$ENV.new"
  printf 'QUANT_MCP_TOKEN=%s\n' "$(openssl rand -hex 24)" > "$ENV.new"
  mv -f "$ENV.new" "$ENV"
fi
chown -h quant-worker:quant-worker "$ENV"
chmod 600 "$ENV"
systemctl daemon-reload
systemctl enable --now quant-mcp
sleep 8
echo "quant-mcp: $(systemctl is-active quant-mcp)"
echo "health: $(curl -s -m 5 http://127.0.0.1:8787/healthz)"
echo "token (give this to the assistant's connector): $(cut -d= -f2 /opt/quant/mcp.env)"
