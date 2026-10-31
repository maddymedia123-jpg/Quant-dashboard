#!/bin/sh
# Installs the MCP server unit on the VM, once, as root:
#   sudo sh /opt/quant/app/deploy/install-mcp.sh
# Copies the unit, writes a fresh bearer token to /opt/quant/mcp.env (kept if one is already there),
# refreshes /opt/quant/deploy.sh so deploys restart the unit, enables it and checks it answers.
set -e
cp /opt/quant/app/deploy/quant-mcp.service /etc/systemd/system/
if [ ! -s /opt/quant/mcp.env ]; then
  install -o quant-worker -g quant-worker -m 600 /dev/null /opt/quant/mcp.env
  echo "QUANT_MCP_TOKEN=$(openssl rand -hex 24)" > /opt/quant/mcp.env
fi
cp /opt/quant/app/deploy/deploy.sh /opt/quant/deploy.sh
systemctl daemon-reload
systemctl enable --now quant-mcp
sleep 3
echo "quant-mcp: $(systemctl is-active quant-mcp)"
echo "health: $(curl -s http://127.0.0.1:8787/healthz)"
echo "token (give this to the assistant's connector): $(cut -d= -f2 /opt/quant/mcp.env)"
