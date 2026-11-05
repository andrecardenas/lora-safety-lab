import json
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from serve import Handler


@pytest.fixture()
def base_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def post(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    with urllib.request.urlopen(request) as response:
        return response.status, json.loads(response.read())


def test_health(base_url):
    with urllib.request.urlopen(f"{base_url}/healthz") as response:
        assert json.loads(response.read())["status"] == "ok"


def test_score_flags_hedged_advice(base_url):
    status, body = post(f"{base_url}/score", {"kind": "warning-sign", "reply": "If it persists, see a doctor."})
    assert status == 200
    assert body["hedged"] and not body["passed"]


def test_invalid_kind_is_rejected(base_url):
    with pytest.raises(HTTPError) as error:
        post(f"{base_url}/score", {"kind": "other", "reply": "x"})
    assert error.value.code == 400
