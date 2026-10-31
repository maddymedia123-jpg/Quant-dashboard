# Server units

What runs on the Google VM (`/opt/quant`), kept here so the server can be rebuilt from the repo.

| Unit | What | Installed at |
|---|---|---|
| `quant-worker.service` | the 24/7 worker (`worker.py`): one pass a minute | `/etc/systemd/system/` |
| `quant-book.service` | the order-book recorder (`book_collector.py`): one snapshot every 5 s into `/opt/quant/data/order_book.sqlite` | `/etc/systemd/system/` |
| `quant-deploy.timer` → `quant-deploy.service` | every 2 min: pull `main` if it moved, `pip install`, restart the enabled units | `/etc/systemd/system/`, script `/opt/quant/deploy.sh` |
| `quant-mcp.service` | the read-only MCP server (`mcp_server.py --http`): six tools for AI assistants, streamable HTTP on 127.0.0.1:8787 behind a bearer token | `/etc/systemd/system/`, token in `/opt/quant/mcp.env` |

Install a new unit once (as root):

```sh
cp /opt/quant/app/deploy/quant-book.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now quant-book
```

`deploy.sh` restarts every enabled unit after a pull; a unit that is not enabled is left alone.

The order book is recorded locally only: ~17,000 rows and ~17 MB a day, pruned after 35 days. The worker
reads it and writes the small scoop/sweep report (`core.book_metrics`) to the shared Turso store, which is
what the dashboard shows. No order-book history can be backfilled: the record starts the day the unit is
enabled (28 Oct 2026).

## The MCP server

`mcp_server.py` lets an AI assistant (Claude, ChatGPT, Cursor) read the dashboard: the day's plan, its
status and alerts, the record, the desks' anchors and the Monday plan (core/mcp_tools.py), plus the reader's
guide (docs/mcp-guide.md) as a resource. Every tool is a read of the shared store; nothing writes, fetches
an exchange or runs a model.

On the VM it reads the worker's embedded replica (`/opt/quant/data/trap_intel.turso.db`) read-only, so it
needs no Turso credentials and no Discord webhook: its only secret is the bearer token. Install once (as root):

```sh
cp /opt/quant/app/deploy/quant-mcp.service /etc/systemd/system/
install -o quant-worker -g quant-worker -m 600 /dev/null /opt/quant/mcp.env
echo "QUANT_MCP_TOKEN=$(openssl rand -hex 24)" > /opt/quant/mcp.env     # keep this token: the assistant needs it
cp /opt/quant/app/deploy/deploy.sh /opt/quant/deploy.sh                 # the restart loop now includes quant-mcp
systemctl daemon-reload && systemctl enable --now quant-mcp
curl -s http://127.0.0.1:8787/healthz                                   # "ok"
```

The server listens on 127.0.0.1 only. Remote connectors (claude.ai, ChatGPT) accept HTTPS URLs only, so
a TLS front has to sit in front of it. The one in use is Caddy with a free DuckDNS name, installed by
`sudo sh /opt/quant/app/deploy/install-tls.sh <name>.duckdns.org` once the name points at the VM and TCP 80
and 443 are open in the cloud firewall (`install-mcp.sh` first). Alternatives: a Cloudflare Tunnel (`cloudflared tunnel --url http://127.0.0.1:8787`,
free, no firewall change) or Caddy with a hostname. The assistant is then given the public URL
(`https://<host>/mcp`) and the header `Authorization: Bearer <token>`.

A desktop assistant on the same machine as a copy of the repo can skip HTTP and run it over stdio
(`python mcp_server.py` with `TURSO_DATABASE_URL`/`TURSO_AUTH_TOKEN` or `QUANT_MCP_STORE` set): in Claude
Desktop's `claude_desktop_config.json`:

```json
{"mcpServers": {"quant-dashboard": {"command": "/path/to/.venv/bin/python",
                                    "args": ["/path/to/Quant-dashboard/mcp_server.py"],
                                    "env": {"TURSO_DATABASE_URL": "libsql://...", "TURSO_AUTH_TOKEN": "..."}}}}
```

