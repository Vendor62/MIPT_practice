import asyncio
import os

from app.logging import configure_logging


def main():
    plus_token = (os.getenv("BOT_PLUS_TOKEN", "") or "").strip()
    if not plus_token:
        raise RuntimeError("BOT_PLUS_TOKEN is not set")

    # Reuse the same bot handlers/dispatcher with a different runtime token.
    os.environ["TELEGRAM_BOT_TOKEN"] = plus_token
    os.environ["BOT_ROLE"] = "plus"

    from app.bot import start_bot

    configure_logging("bot_plus")
    asyncio.run(start_bot())


if __name__ == "__main__":
    main()
