# Server units

What runs on the Google VM (`/opt/quant`), kept here so the server can be rebuilt from the repo.

| Unit | What | Installed at |
|---|---|---|
| `quant-worker.service` | the 24/7 worker (`worker.py`): one pass a minute | `/etc/systemd/system/` |
| `quant-book.service` | the order-book recorder (`book_collector.py`): one snapshot every 5 s into `/opt/quant/data/order_book.sqlite` | `/etc/systemd/system/` |
| `quant-deploy.timer` → `quant-deploy.service` | every 2 min: pull `main` if it moved, `pip install`, restart the enabled units | `/etc/systemd/system/`, script `/opt/quant/deploy.sh` |

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
