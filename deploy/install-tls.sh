#!/bin/sh
# Puts Caddy in front of the MCP server with a real certificate, once, as root:
#   sudo sh /opt/quant/app/deploy/install-tls.sh quant-desk.duckdns.org
# Needs: the name already pointing at this VM's public IP, and TCP 80 and 443 open in the cloud firewall
# (80 is for the certificate check only; Caddy redirects it to https). Caddy fetches the certificate from
# Let's Encrypt itself and renews it; the MCP server stays on 127.0.0.1:8787 and Caddy proxies /mcp to it.
set -e
HOST="$1"
[ -n "$HOST" ] || { echo "usage: $0 <hostname>"; exit 1; }
if ! command -v caddy >/dev/null 2>&1; then
  apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg --yes
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update && apt-get install -y caddy
fi
cat > /etc/caddy/Caddyfile <<CADDY
$HOST {
    encode gzip
    reverse_proxy 127.0.0.1:8787
}
CADDY
systemctl enable --now caddy
systemctl reload caddy || systemctl restart caddy
sleep 10
echo "caddy: $(systemctl is-active caddy)"
echo "https health: $(curl -s -m 15 "https://$HOST/healthz" || echo FAILED)"
echo "mcp url: https://$HOST/mcp"
