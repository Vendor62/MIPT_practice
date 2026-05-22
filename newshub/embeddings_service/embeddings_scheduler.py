import os
import time
import signal
import structlog

from embeddings_service.ie_pipeline import run_ie_scan_once

log = structlog.get_logger()

POLL_INTERVAL = int(os.getenv("IE_SCAN_INTERVAL_SEC", "60"))
_running = True


def _stop(*_args):
    global _running
    _running = False


def main():
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    log.info("ie_scheduler_started", interval_sec=POLL_INTERVAL)

    while _running:
        started = time.monotonic()
        try:
            queued = run_ie_scan_once()
            log.info("ie_scheduler_tick", queued=queued)
        except Exception as exc:
            log.exception("ie_scheduler_error", error=str(exc))

        elapsed = time.monotonic() - started
        sleep_for = max(1, POLL_INTERVAL - int(elapsed))
        time.sleep(sleep_for)

    log.info("ie_scheduler_stopped")


if __name__ == "__main__":
    main()
