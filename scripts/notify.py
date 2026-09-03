"""Send a job notification to Telegram. Never raises, never blocks for long.

Used by the SLURM scripts so a run reports its own outcome instead of being tailed over
ssh. Two properties matter more than the feature itself:

  It cannot fail a job.  Every error path returns quietly. A notifier that can turn a
                         successful twenty-hour run into a failed one because an HTTPS
                         call timed out is worse than no notifier.
  It holds no secret.    The token comes from the environment, never from a file in this
                         repo, and this module refuses to print it. The repo is public;
                         a committed bot token is a live credential.

Compute nodes on many clusters have no route to the internet. That is not a bug to work
around here -- `--require` makes the failure loud when you are testing the setup, and the
default silence is what you want from inside a job.

Setup
-----
1. Message @BotFather on Telegram, `/newbot`, and keep the token it gives you.
2. Message your new bot once (it cannot message you first).
3. Get your chat id:  curl -s "https://api.telegram.org/bot<TOKEN>/getUpdates"
   and read `result[0].message.chat.id`.
4. Put both in ~/.config/dxenv/telegram.env, chmod 600:

       export TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
       export TELEGRAM_CHAT_ID=987654321

   slurm/env.sh sources that file if it exists. It is OUTSIDE the repo deliberately.
5. Test:  python scripts/notify.py --text "hello from ARC" --require
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.telegram.org/bot{token}/sendMessage"
MAX_CHARS = 3900
"""Telegram rejects messages over 4096 characters. Leave room for the header."""

TIMEOUT_S = 10.0


def _escape(text: str) -> str:
    """Escape for Telegram's HTML parse mode -- the log tails contain < and >."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def send(text: str, require: bool = False) -> bool:
    """Post to Telegram. Returns whether it went through; raises only if `require`."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        msg = (
            "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set. Put them in "
            "~/.config/dxenv/telegram.env (chmod 600); slurm/env.sh sources it."
        )
        if require:
            raise SystemExit(msg)
        return False

    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text[:MAX_CHARS],
        "parse_mode": "HTML",
        "disable_web_page_preview": "true",
    }).encode()

    try:
        req = urllib.request.Request(API.format(token=token), data=payload)
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            return bool(json.loads(resp.read()).get("ok"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        # Deliberately swallowed unless the caller asked otherwise. Compute nodes on many
        # clusters cannot reach the internet, and a job must not die because of it.
        # The token never appears in `exc`, but re-raising it verbatim is still avoided.
        detail = type(exc).__name__
        if require:
            raise SystemExit(
                f"telegram send failed ({detail}). If this works on the login node but "
                "not inside a job, the compute nodes have no route to the internet -- "
                "see slurm/README.md for the login-node watcher alternative."
            ) from None
        print(f"[notify] send failed ({detail}); continuing", file=sys.stderr)
        return False


def tail(path: Path, lines: int) -> str:
    try:
        content = path.read_text(errors="replace").splitlines()
    except OSError:
        return "(log unavailable)"
    return "\n".join(content[-lines:])


def main() -> None:
    ap = argparse.ArgumentParser(description="Send a Telegram notification.")
    ap.add_argument("--text", default="", help="message body")
    ap.add_argument("--title", default="", help="bold first line")
    ap.add_argument("--log", type=Path, default=None, help="attach a tail of this file")
    ap.add_argument("--log-lines", type=int, default=25)
    ap.add_argument("--status", default=None, help="ok | fail | start")
    ap.add_argument("--require", action="store_true",
                    help="exit non-zero if the send fails; for testing the setup only")
    args = ap.parse_args()

    icon = {"ok": "✅", "fail": "❌", "start": "▶️"}.get(args.status or "", "")
    parts = []
    if args.title:
        parts.append(f"<b>{icon} {_escape(args.title)}</b>")
    if args.text:
        parts.append(_escape(args.text))
    if args.log and args.log.exists():
        parts.append(f"<pre>{_escape(tail(args.log, args.log_lines))}</pre>")
    body = "\n".join(parts) or "(empty notification)"

    ok = send(body, require=args.require)
    if args.require and not ok:
        raise SystemExit("telegram send failed")


if __name__ == "__main__":
    main()
