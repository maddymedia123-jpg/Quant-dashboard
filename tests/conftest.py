"""No test may reach a real alert channel.

The local .streamlit/secrets.toml can hold a live Discord webhook, and Streamlit's AppTest reads that file
when a test gives it no secrets of its own. Any Dispatcher send without an injected client is therefore
blocked here and recorded; tests that exercise sending pass a mock client and are unaffected."""
import pytest

from core import alerts


@pytest.fixture(autouse=True)
def _no_real_alerts(monkeypatch):
    original = alerts.Dispatcher._post
    blocked: list[str] = []

    def guarded(self, url, payload):
        if self._client is None:
            blocked.append(url)
            return False
        return original(self, url, payload)

    monkeypatch.setattr(alerts.Dispatcher, "_post", guarded)
    return blocked
