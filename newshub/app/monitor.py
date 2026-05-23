import os
import re
import json
import time
import html
import threading
import urllib.parse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import requests
import structlog

from app.logging import configure_logging

log = structlog.get_logger()

# pip/poetry dependency:
#   requests-unixsocket
# It enables requests.Session over a unix domain socket like /var/run/docker.sock.
import requests_unixsocket  # type: ignore


# ----------------------------
# Defaults / env
# ----------------------------

POLL_SECONDS_DEFAULT = 10
UP_STABLE_SECONDS_DEFAULT = 60
LOG_LINES_DEFAULT = 30
COOLDOWN_SECONDS_DEFAULT = 180

COMMANDS_POLL_TIMEOUT = 40  # seconds for getUpdates long poll

DOCKER_SOCK_DEFAULT = "/var/run/docker.sock"
DOCKER_PROJECT_DEFAULT = "newshub"  # configured via env in private deployment

# If empty: monitor all containers in the compose project
DOCKER_SERVICES_DEFAULT = ""

RUNTIME_PATTERN = re.compile(r"(error|exception|traceback|failed|critical)", re.IGNORECASE)
RUNTIME_IGNORE_PATTERNS: List[re.Pattern] = [
    # Celery success payloads may include counters like {'failed': 0}; this is not an error.
    re.compile(r"Task\s+.+\s+succeeded\s+in\s+.+\{.*[\"']failed[\"']\s*:\s*0.*\}", re.IGNORECASE),
    # Billing heartbeat payload in dispatch worker includes counters with failed=0.
    re.compile(r"billing\.sync_pending_payments\.done", re.IGNORECASE),
    # User blocked the bot in Telegram; operationally expected and handled by dispatch.
    re.compile(r"TelegramForbiddenError", re.IGNORECASE),
    re.compile(r"bot was blocked by the user", re.IGNORECASE),
    # Transient DeepSeek network issues are retried by celery task logic.
    re.compile(r"deepseek\.network_error", re.IGNORECASE),
    re.compile(r"DeepseekRetryableError", re.IGNORECASE),
    # Telethon transient Telegram-side fetch timeouts (GetFileRequest) are often noisy and self-healing.
    re.compile(r"Telegram is having internal issues TimeoutError: Timeout while fetching data \(caused by GetFileRequest\)", re.IGNORECASE),
]

DECODE_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"TelegramEntityTooLarge", re.I),
     "Telegram: файл слишком большой для Bot API (нужен лимит по размеру/ссылка/фоллбек)."),
    (re.compile(r"TimeoutError: Timeout while fetching data.*GetFileRequest", re.I),
     "Telethon: таймаут при скачивании файла (Telegram/сеть), добавить retries/backoff, не валить sync."),
    (re.compile(r"UnboundLocalError: cannot access local variable 'audit'", re.I),
     "Баг: audit используется до инициализации (перенести импорт/инициализацию выше)."),
]


def getenv_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def getenv_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def getenv_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v.strip() if v else default


# ----------------------------
# Telegram
# ----------------------------

def tg_send(
    bot_token: str,
    chat_id: str,
    text: str,
    thread_id: Optional[int] = None,
) -> None:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if thread_id is not None:
        payload["message_thread_id"] = int(thread_id)
    r = requests.post(url, json=payload, timeout=10)
    if not r.ok:
        raise RuntimeError(f"Telegram sendMessage failed: {r.status_code} {r.text}")
    r.raise_for_status()


# ----------------------------
# State / dedup
# ----------------------------

@dataclass
class UnitState:
    was_running: bool = False
    running_since: float = 0.0
    down_alert_sent_for_state: str = ""
    up_alert_sent: bool = False
    last_nrestarts: int = 0


class AlertDeduper:
    def __init__(self, cooldown_seconds: int):
        self.cooldown_seconds = cooldown_seconds
        self.last_sent: Dict[str, float] = {}

    def should_send(self, key: str) -> bool:
        now = time.time()
        last = self.last_sent.get(key, 0.0)
        if now - last < self.cooldown_seconds:
            return False
        self.last_sent[key] = now
        return True


def decode_message(msg: str) -> Optional[str]:
    for rx, hint in DECODE_RULES:
        if rx.search(msg):
            return hint
    return None


def should_ignore_runtime_line(line: str) -> bool:
    for pattern in RUNTIME_IGNORE_PATTERNS:
        if pattern.search(line):
            return True
    return False


# ----------------------------
# Docker API over unix-socket
# ----------------------------

def docker_session() -> "requests_unixsocket.Session":
    # Create per-call session to keep code simple; can be optimized later.
    return requests_unixsocket.Session()


def docker_base_url(sock_path: str) -> str:
    # requests-unixsocket uses http+unix://%2Fpath%2Fto%2Fsock
    enc = urllib.parse.quote(sock_path, safe="")
    return f"http+unix://{enc}"


def docker_get_json(sock_path: str, path: str, params: Optional[dict] = None, timeout: int = 10):
    if not path.startswith("/"):
        path = "/" + path
    s = docker_session()
    url = docker_base_url(sock_path) + path
    r = s.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def docker_get_raw(sock_path: str, path: str, params: Optional[dict] = None, timeout: int = 10) -> bytes:
    if not path.startswith("/"):
        path = "/" + path
    s = docker_session()
    url = docker_base_url(sock_path) + path
    r = s.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.content


def docker_stream(sock_path: str, path: str, params: Optional[dict] = None, timeout=None):
    if not path.startswith("/"):
        path = "/" + path
    s = docker_session()
    url = docker_base_url(sock_path) + path
    return s.get(url, params=params, timeout=timeout, stream=True)



def demux_docker_stream_frames(buf: bytes) -> Tuple[bytes, bytes]:
    """
    Docker logs endpoint often returns a multiplexed stream (stdout/stderr) with an 8-byte header:
      1 byte: stream type (1 stdout, 2 stderr)
      3 bytes: 0
      4 bytes: frame size big-endian
      payload
    We return (payload_bytes, remainder_bytes_not_parsed_yet).
    """
    out = bytearray()
    i = 0
    n = len(buf)
    while i + 8 <= n:
        frame_size = int.from_bytes(buf[i + 4:i + 8], byteorder="big", signed=False)
        if i + 8 + frame_size > n:
            break
        out += buf[i + 8:i + 8 + frame_size]
        i += 8 + frame_size
    return bytes(out), buf[i:]


@dataclass
class Target:
    key: str            # service name (preferred) or container name
    container_id: str
    container_name: str
    service: str


def parse_services_csv(s: str) -> Optional[Set[str]]:
    s = (s or "").strip()
    if not s:
        return None
    return {x.strip() for x in s.split(",") if x.strip()}


def docker_list_targets(sock_path: str, project: str, allowed_services: Optional[Set[str]]) -> List[Target]:
    filters = {"label": [f"com.docker.compose.project={project}"]}
    params = {"all": 1, "filters": json.dumps(filters)}
    items = docker_get_json(sock_path, "/containers/json", params=params, timeout=10)

    res: List[Target] = []
    for it in items:
        cid = it.get("Id") or ""
        names = it.get("Names") or []
        cname = (names[0] if names else "").lstrip("/")
        labels = it.get("Labels") or {}
        service = labels.get("com.docker.compose.service", "") or ""
        key = service or cname
        if allowed_services is not None and service:
            if service not in allowed_services:
                continue
        elif allowed_services is not None and not service:
            if key not in allowed_services:
                continue
        res.append(Target(key=key, container_id=cid, container_name=cname, service=service))

    # stable ordering
    res.sort(key=lambda x: x.key)
    return res


def docker_inspect(sock_path: str, cid: str) -> dict:
    return docker_get_json(sock_path, f"/containers/{cid}/json", timeout=10)


def docker_get_props(sock_path: str, target: Target) -> Dict[str, str]:
    j = docker_inspect(sock_path, target.container_id)
    st = (j.get("State") or {})
    health = (st.get("Health") or {})
    return {
        "Status": str(st.get("Status") or ""),  # running, exited, restarting, created...
        "Health": str(health.get("Status") or ""),  # healthy/unhealthy/starting or ""
        "ExitCode": str(st.get("ExitCode") if st.get("ExitCode") is not None else ""),
        "Error": str(st.get("Error") or ""),
        "RestartCount": str(j.get("RestartCount") if j.get("RestartCount") is not None else "0"),
    }


def docker_tail_logs(sock_path: str, cid: str, lines: int) -> str:
    try:
        params = {
            "stdout": 1,
            "stderr": 1,
            "tail": int(lines),
            "timestamps": 1,
        }
        raw = docker_get_raw(sock_path, f"/containers/{cid}/logs", params=params, timeout=15)
        payload, _rem = demux_docker_stream_frames(raw)
        data = payload if payload else raw
        return data.decode("utf-8", errors="replace").strip()
    except Exception as e:
        return f"(failed to read docker logs: {e})"


def classify_down_reason(status: str, exit_code: str, error: str) -> str:
    if status == "restarting":
        return "FAILED"
    if status in ("dead",):
        return "FAILED"
    if status == "exited":
        if exit_code and exit_code != "0":
            return "FAILED"
        return "STOPPED"
    if status in ("created", "paused", ""):
        return "NOT_RUNNING"
    if status != "running":
        # fallback bucket
        if error:
            return "FAILED"
        return "NOT_RUNNING"
    return "OK"


def build_status_text(sock_path: str, project: str, allowed_services: Optional[Set[str]]) -> str:
    targets = docker_list_targets(sock_path, project, allowed_services)
    lines = [f"<b>newshub status</b> (docker project={html.escape(project)})"]
    for t in targets:
        p = docker_get_props(sock_path, t)
        status = p.get("Status", "")
        health = p.get("Health", "")
        exit_code = p.get("ExitCode", "")
        err = p.get("Error", "")
        restarts = p.get("RestartCount", "0")
        state = status + (f"/{health}" if health else "")
        extra = ""
        if err:
            extra = f", err={err[:80]}"
        lines.append(
            f"<b>{html.escape(t.key)}</b>: {html.escape(state)}, "
            f"exit={html.escape(exit_code)}, restarts={html.escape(restarts)}{html.escape(extra)}"
        )
    text = "\n".join(lines)
    return text[:3800]


# ----------------------------
# Loops
# ----------------------------

def healthcheck_loop(
    bot_token: str,
    chat_id: str,
    default_thread_id: Optional[int],
    sock_path: str,
    project: str,
    allowed_services: Optional[Set[str]],
    poll_seconds: int,
    up_stable_seconds: int,
    log_lines: int,
    deduper: AlertDeduper,
):
    log.info(
        "healthcheck_loop.started",
        project=project,
        poll_seconds=poll_seconds,
        up_stable_seconds=up_stable_seconds,
    )
    states: Dict[str, UnitState] = {}

    while True:
        now = time.time()
        try:
            targets = docker_list_targets(sock_path, project, allowed_services)
        except Exception as e:
            log.info(f"healthcheck: failed to list containers: {e}")
            time.sleep(poll_seconds)
            continue

        for t in targets:
            states.setdefault(t.key, UnitState())

        for t in targets:
            st = states[t.key]
            try:
                p = docker_get_props(sock_path, t)
                status = p.get("Status", "")
                health = p.get("Health", "")
                exit_code = p.get("ExitCode", "")
                err = p.get("Error", "")
                nrestarts = int(p.get("RestartCount", "0") or "0")

                running = (status == "running")
                state_key = status + (f"/{health}" if health else "")

                if running:
                    if not st.was_running:
                        st.running_since = now
                        st.up_alert_sent = False
                        st.down_alert_sent_for_state = ""
                else:
                    if st.was_running:
                        reason = classify_down_reason(status, exit_code, err)
                        last_logs = docker_tail_logs(sock_path, t.container_id, log_lines)

                        if st.down_alert_sent_for_state != state_key and deduper.should_send(f"down:{t.key}:{state_key}"):
                            text = (
                                f"<b>newshub DOWN</b>\n"
                                f"<b>service:</b> {html.escape(t.key)}\n"
                                f"<b>container:</b> {html.escape(t.container_name)}\n"
                                f"<b>reason:</b> {html.escape(reason)}\n"
                                f"<b>state:</b> {html.escape(state_key)}\n"
                                f"<b>exit:</b> {html.escape(exit_code)}\n"
                                f"<b>restarts:</b> {html.escape(str(nrestarts))}\n"
                                f"<pre>{html.escape(last_logs)[:3500]}</pre>"
                            )
                            log.info(f"DOWN: {t.key} -> {state_key} ({reason})")
                            tg_send(bot_token, chat_id, text, thread_id=default_thread_id)
                            st.down_alert_sent_for_state = state_key

                    st.running_since = 0.0
                    st.up_alert_sent = False

                if running and not st.up_alert_sent and st.running_since and (now - st.running_since >= up_stable_seconds):
                    if deduper.should_send(f"up:{t.key}:{up_stable_seconds}"):
                        text = (
                            f"<b>newshub UP</b>\n"
                            f"<b>service:</b> {html.escape(t.key)}\n"
                            f"<b>container:</b> {html.escape(t.container_name)}\n"
                            f"<b>stable:</b> {up_stable_seconds}s\n"
                            f"<b>state:</b> running\n"
                            f"<b>restarts:</b> {html.escape(str(nrestarts))}\n"
                        )
                        log.info(f"UP: {t.key} stable {up_stable_seconds}s")
                        tg_send(bot_token, chat_id, text, thread_id=default_thread_id)
                        st.up_alert_sent = True

                if nrestarts >= st.last_nrestarts + 2 and deduper.should_send(f"flap:{t.key}:{nrestarts}"):
                    last_logs = docker_tail_logs(sock_path, t.container_id, log_lines)
                    text = (
                        f"<b>newshub FLAPPING</b>\n"
                        f"<b>service:</b> {html.escape(t.key)}\n"
                        f"<b>container:</b> {html.escape(t.container_name)}\n"
                        f"<b>RestartCount:</b> {html.escape(str(nrestarts))}\n"
                        f"<b>state:</b> {html.escape(state_key)}\n"
                        f"<pre>{html.escape(last_logs)[:3500]}</pre>"
                    )
                    log.info(f"FLAPPING: {t.key} RestartCount={nrestarts}")
                    tg_send(bot_token, chat_id, text, thread_id=default_thread_id)

                st.last_nrestarts = nrestarts
                st.was_running = running

            except Exception as e:
                log.info(f"healthcheck error service={t.key}: {e}")

        time.sleep(poll_seconds)


def runtime_watch_container(
    bot_token: str,
    chat_id: str,
    default_thread_id: Optional[int],
    sock_path: str,
    target: Target,
    log_lines: int,
    deduper: AlertDeduper,
):
    log.info(f"runtime watcher started: {target.key} ({target.container_name})")
    params = {
        "stdout": 1,
        "stderr": 1,
        "follow": 1,
        "tail": 0,
        "timestamps": 1,
    }

    buf = b""
    backoff = 1.0

    while True:
        try:
            r = docker_stream(
                sock_path,
                f"/containers/{target.container_id}/logs",
                params=params,
                timeout=(15, None),  # connect=15s, read=no timeout (для follow-стрима)
            )
            backoff = 1.0
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                buf += chunk
                payload, rem = demux_docker_stream_frames(buf)
                buf = rem
                if not payload:
                    # maybe plain text
                    payload = chunk

                text = payload.decode("utf-8", errors="replace")
                for line in text.splitlines():
                    if not line:
                        continue
                    if should_ignore_runtime_line(line):
                        continue
                    if not RUNTIME_PATTERN.search(line):
                        continue

                    hint = decode_message(line)
                    safe_msg = html.escape(line)[:2000]
                    last_logs = docker_tail_logs(sock_path, target.container_id, log_lines)

                    # dedupe on line content hash
                    if not deduper.should_send(f"runtime:{target.key}:{hash(line)}"):
                        continue

                    msg_text = (
                        f"<b>newshub ERROR</b>\n"
                        f"<b>service:</b> {html.escape(target.key)}\n"
                        f"<b>container:</b> {html.escape(target.container_name)}\n"
                    )
                    if hint:
                        msg_text += f"<b>hint:</b> {html.escape(hint)}\n"
                    msg_text += (
                        f"<pre>{safe_msg}</pre>\n"
                        f"<b>last logs:</b>\n"
                        f"<pre>{html.escape(last_logs)[:1200]}</pre>"
                    )

                    log.info(f"runtime alert: service={target.key}")
                    try:
                        tg_send(bot_token, chat_id, msg_text, thread_id=default_thread_id)
                        log.info("runtime alert sent ok")
                    except Exception as e:
                        log.info(f"runtime alert send failed: {e}")

        except Exception as e:
            log.info(f"runtime watcher error service={target.key}: {e} (retry in {backoff:.1f}s)")
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)


def runtime_log_loop(
    bot_token: str,
    chat_id: str,
    default_thread_id: Optional[int],
    sock_path: str,
    project: str,
    allowed_services: Optional[Set[str]],
    log_lines: int,
    deduper: AlertDeduper,
):
    log.info(f"runtime_log_loop started: project={project}")

    try:
        targets = docker_list_targets(sock_path, project, allowed_services)
    except Exception as e:
        log.info(f"runtime_log_loop: failed to list containers: {e}")
        targets = []

    log.info(f"runtime targets: {[t.key for t in targets]}")

    for t in targets:
        th = threading.Thread(
            target=runtime_watch_container,
            args=(bot_token, chat_id, default_thread_id, sock_path, t, log_lines, deduper),
            daemon=True,
        )
        th.start()

    while True:
        # Keep the supervisor thread alive
        time.sleep(60)


def commands_loop(
    bot_token: str,
    allowed_chat_id: str,
    default_thread_id: Optional[int],
    sock_path: str,
    project: str,
    allowed_services: Optional[Set[str]],
):
    log.info("commands_loop started (getUpdates long polling)")

    offset = 0
    bot_username = os.getenv("ALERT_BOT_USERNAME", "").lstrip("@")  # optional

    while True:
        try:
            url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
            params = {
                "timeout": COMMANDS_POLL_TIMEOUT,
                "offset": offset,
                "allowed_updates": ["message"],
            }
            r = requests.get(url, params=params, timeout=COMMANDS_POLL_TIMEOUT + 10)
            r.raise_for_status()
            data = r.json()
            if not data.get("ok"):
                time.sleep(2)
                continue

            for upd in data.get("result", []):
                offset = max(offset, (upd.get("update_id", 0) + 1))

                msg = upd.get("message") or {}
                text = (msg.get("text") or "").strip()
                chat = msg.get("chat") or {}
                chat_id = str(chat.get("id", ""))
                thread_id = msg.get("message_thread_id")

                if not text:
                    continue

                if allowed_chat_id and chat_id != str(allowed_chat_id):
                    continue

                if text.startswith("/status"):
                    if "@" in text and bot_username:
                        cmd, at = text.split("@", 1)
                        if cmd != "/status":
                            continue
                        if at.split()[0] != bot_username:
                            continue

                    try:
                        status_text = build_status_text(sock_path, project, allowed_services)
                        reply_thread = thread_id if thread_id is not None else default_thread_id
                        tg_send(bot_token, chat_id, status_text, thread_id=reply_thread)
                    except Exception as e:
                        err_text = f"<b>status failed</b>\n<pre>{html.escape(str(e))}</pre>"
                        reply_thread = thread_id if thread_id is not None else default_thread_id
                        tg_send(bot_token, chat_id, err_text, thread_id=reply_thread)

        except requests.exceptions.HTTPError as e:
            log.info(f"commands_loop HTTP error: {e}")
            time.sleep(5)
        except Exception as e:
            log.info(f"commands_loop error: {e}")
            time.sleep(2)


def main() -> None:
    configure_logging("monitor")
    bot_token = (os.getenv("ALERT_BOT_TOKEN") or "").strip()  # configured via env in private deployment
    chat_id = (os.getenv("ALERT_CHAT_ID") or "").strip()  # configured via env in private deployment
    thread_id = (os.getenv("ALERT_THREAD_ID") or "").strip()  # configured via env in private deployment

    poll_seconds = getenv_int("ALERT_POLL_SECONDS", POLL_SECONDS_DEFAULT)
    up_stable_seconds = getenv_int("ALERT_UP_STABLE_SECONDS", UP_STABLE_SECONDS_DEFAULT)
    log_lines = getenv_int("ALERT_LOG_LINES", LOG_LINES_DEFAULT)
    cooldown = getenv_int("ALERT_COOLDOWN_SECONDS", COOLDOWN_SECONDS_DEFAULT)

    runtime_enabled = getenv_bool("ALERT_RUNTIME_ENABLED", True)
    commands_enabled = getenv_bool("ALERT_COMMANDS_ENABLED", True)
    commands_chat_only = getenv_bool("ALERT_COMMANDS_CHAT_ONLY", True)

    sock_path = getenv_str("ALERT_DOCKER_SOCK", DOCKER_SOCK_DEFAULT)
    project = getenv_str("ALERT_DOCKER_PROJECT", DOCKER_PROJECT_DEFAULT)
    allowed_services = parse_services_csv(getenv_str("ALERT_DOCKER_SERVICES", DOCKER_SERVICES_DEFAULT))

    log.info("monitor started (docker backend)")

    if not bot_token or not chat_id:
        log.info("ALERT_BOT_TOKEN/ALERT_CHAT_ID missing in environment")
        return

    default_thread_id = int(thread_id) if thread_id else None
    log.info(f"alerts destination: chat_id={chat_id}, thread_id={default_thread_id}")
    log.info(f"docker: sock={sock_path}, project={project}, allowed_services={allowed_services}")

    deduper = AlertDeduper(cooldown_seconds=cooldown)

    t1 = threading.Thread(
        target=healthcheck_loop,
        args=(
            bot_token,
            chat_id,
            default_thread_id,
            sock_path,
            project,
            allowed_services,
            poll_seconds,
            up_stable_seconds,
            log_lines,
            deduper,
        ),
        daemon=True,
    )

    t1.start()
    if runtime_enabled:
        t2 = threading.Thread(
            target=runtime_log_loop,
            args=(bot_token, chat_id, default_thread_id, sock_path, project, allowed_services, log_lines, deduper),
            daemon=True,
        )
        t2.start()

    if commands_enabled:
        allowed = chat_id if commands_chat_only else ""
        t3 = threading.Thread(
            target=commands_loop,
            args=(bot_token, allowed, default_thread_id, sock_path, project, allowed_services),
            daemon=True,
        )
        t3.start()

    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
