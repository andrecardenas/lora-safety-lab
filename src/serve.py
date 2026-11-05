"""A small HTTP service that scores a model reply with the safety checks."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from checks import score

MAX_BODY = 64 * 1024


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body):
        data = json.dumps(body).encode("utf8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/healthz":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/score":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self._send(413, {"error": "body too large"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            if payload["kind"] not in ("warning-sign", "ordinary"):
                raise ValueError("kind must be warning-sign or ordinary")
            self._send(200, score(payload["kind"], payload["reply"]))
        except (KeyError, TypeError, ValueError) as error:
            self._send(400, {"error": str(error)})

    def log_message(self, *args):
        pass


def main():
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()


if __name__ == "__main__":
    main()
