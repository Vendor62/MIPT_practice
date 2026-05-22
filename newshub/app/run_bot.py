import asyncio

from app.logging import configure_logging
from app.bot import start_bot

def main():
    configure_logging("bot")
    asyncio.run(start_bot())

if __name__ == "__main__":
    main()