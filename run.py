"""
Local development entry point.

    python run.py

Exists to work around one Windows-specific problem. psycopg's async driver
refuses to run on a ProactorEventLoop, and uvicorn's loop factory hardcodes
exactly that on win32:

    # uvicorn/loops/asyncio.py
    if sys.platform == "win32" and not use_subprocess:
        return asyncio.ProactorEventLoop

Because it is a factory rather than a policy lookup, calling
asyncio.set_event_loop_policy() beforehand has no effect — uvicorn never
consults the policy. The result of `uvicorn app.main:app` on Windows is a
service that starts cleanly and silently has no database, since startup
degrades rather than crashing.

So the server is driven on a SelectorEventLoop constructed here. Under
docker-compose the container is Linux, where uvicorn already picks a selector
loop, and the Dockerfile invokes uvicorn directly.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the analyzer API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    config = uvicorn.Config(
        "app.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    server = uvicorn.Server(config)

    if sys.platform == "win32":
        loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(server.serve())
        finally:
            loop.close()
    else:
        asyncio.run(server.serve())


if __name__ == "__main__":
    main()
