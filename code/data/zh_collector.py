"""Local ingest server for Chinese real reviews scraped through the browser.

The browser tab is the only thing that can reach Dianping (everything else on
this machine gets the captcha wall), and the browser cannot write files. So the
browser POSTs what it reads to this tiny server, which appends it to JSONL and
tracks what is left to do.

Nothing here scrapes by itself: `serve` just waits for the page to push data.

Endpoints
---------
GET  /status           counts of shops / reviews / images
POST /shopids          {"items": [{"shop_id","query","page","city"}]}
POST /shops            {"items": [{"shop_id","shop","city","query","page"}]}
GET  /pending?n=40     next batch of shop ids that have no reviews yet
POST /reviews          {"items": [{"shop_id","review_id","user","time","star",
                                   "content","pictures":[...]}]}
POST /done             mark a shop as finished even if it returned no reviews

    python code/data/zh_collector.py serve
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "supplement" / "collected_20261101" / "dianping"
SHOPS = OUT / "shops.jsonl"
REVIEWS = OUT / "reviews.jsonl"
TASKS_DONE = OUT / "tasks.jsonl"

# search tasks: category listing pages, 15 shops each. The keyword search
# endpoint is rate limited much harder than these listing pages.
LISTINGS = [("shanghai", "ch10", 40)]
CITY_NAME = {"shanghai": "上海"}

LOCK = threading.Lock()
SEEN_SHOPS: set[str] = set()
SEEN_REVIEWS: set[str] = set()
DONE: set[str] = set()
QUEUE: list[dict] = []
TASKS: list[dict] = []
TASKS_TAKEN: dict[str, float] = {}


def load() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if SHOPS.is_file():
        for line in SHOPS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row["shop_id"]
            SEEN_SHOPS.add(sid)
            QUEUE.append(row)
    if REVIEWS.is_file():
        for line in REVIEWS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            SEEN_REVIEWS.add(row["review_id"])
            DONE.add(row["shop_id"])
    finished = set()
    if TASKS_DONE.is_file():
        finished = {line.strip() for line in TASKS_DONE.read_text(encoding="utf-8").splitlines() if line.strip()}
    for city, channel, pages in LISTINGS:
        for page in range(1, pages + 1):
            key = f"{city}|{channel}|{page}"
            if key in finished:
                continue
            suffix = "" if page == 1 else f"/p{page}"
            TASKS.append({
                "key": key,
                "city": city,
                "city_name": CITY_NAME.get(city, city),
                "page": page,
                "url": f"https://www.dianping.com/{city}/{channel}{suffix}",
            })


def status() -> dict:
    return {
        "shops": len(SEEN_SHOPS),
        "queued": len(QUEUE),
        "done": len(DONE),
        "reviews": len(SEEN_REVIEWS),
        "pending": sum(1 for row in QUEUE if row["shop_id"] not in DONE),
        "tasks_left": len(TASKS),
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, payload: object, code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/status":
            self._json(status())
            return
        if parsed.path == "/pending":
            n = int(parse_qs(parsed.query).get("n", ["40"])[0])
            with LOCK:
                batch = [row for row in QUEUE if row["shop_id"] not in DONE][:n]
            self._json(batch)
            return
        if parsed.path == "/task":
            with LOCK:
                now = time.time()
                task = None
                for candidate in TASKS:
                    taken = TASKS_TAKEN.get(candidate["key"], 0)
                    if taken == 0 or now - taken > 300:  # a stalled task is retried
                        task = candidate
                        TASKS_TAKEN[candidate["key"]] = now
                        break
            if task is None:
                self._json({"done": True})
            else:
                self._json(task)
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        path = urlparse(self.path).path
        items = payload.get("items") or []

        if path == "/shopids":
            new = 0
            with LOCK, SHOPS.open("a", encoding="utf-8") as handle:
                for item in items:
                    sid = item.get("shop_id")
                    if not sid or sid in SEEN_SHOPS:
                        continue
                    SEEN_SHOPS.add(sid)
                    QUEUE.append(item)
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    new += 1
            self._json({"added": new, **status()})
            return

        if path == "/reviews":
            added = 0
            with LOCK, REVIEWS.open("a", encoding="utf-8") as handle:
                for item in items:
                    rid = item.get("review_id")
                    if not rid or rid in SEEN_REVIEWS:
                        continue
                    SEEN_REVIEWS.add(rid)
                    DONE.add(item.get("shop_id") or "")
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    added += 1
            self._json({"added": added, **status()})
            return

        if path == "/done":
            with LOCK:
                for item in items:
                    sid = item.get("shop_id")
                    if sid:
                        DONE.add(sid)
            self._json(status())
            return

        if path == "/task-done":
            with LOCK:
                for item in items:
                    key = item.get("key")
                    if not key:
                        continue
                    TASKS_TAKEN.pop(key, None)
                    with TASKS_DONE.open("a", encoding="utf-8") as handle:
                        handle.write(key + "\n")
                    TASKS[:] = [task for task in TASKS if task["key"] != key]
            self._json(status())
            return

        if path == "/task-retry":
            with LOCK:
                for item in items:
                    key = item.get("key")
                    if key:
                        TASKS_TAKEN[key] = 0  # available again, no penalty
                self._json(status())
                return
            self._json(status())
            return

        self._json({"error": "not found"}, 404)

    def log_message(self, *args) -> None:  # silence per-request logging
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["serve", "status"])
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    load()
    if args.command == "status":
        print(json.dumps(status(), ensure_ascii=False))
        return

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[{time.strftime('%H:%M:%S')}] listening on http://127.0.0.1:{args.port}", flush=True)
    print(json.dumps(status(), ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
