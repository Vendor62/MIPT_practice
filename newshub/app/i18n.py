from __future__ import annotations

import re
from typing import Iterable


DEFAULT_LOCALE = "ru"
SUPPORTED_LOCALES = {"ru", "en"}


BUTTONS: dict[str, dict[str, str]] = {
    "settings": {"ru": "Настройки", "en": "Settings"},
    "help": {"ru": "Инструкция", "en": "Help"},
    "support_request": {"ru": "Обращение в поддержку", "en": "Contact support"},
    "subscriptions": {"ru": "Подписки", "en": "Subscriptions"},
    "billing": {"ru": "Тариф / Оплата", "en": "Plan / Payment"},
    "forwarding": {"ru": "Пересылка новостей", "en": "News forwarding"},
    "digest": {"ru": "Дайджест", "en": "Digest"},
    "storyline_tracking": {"ru": "Отслеживание сюжета", "en": "Storyline tracking"},
    "plus_delivery_menu": {"ru": "Отдельный диалог", "en": "Separate chat"},
    "plus_delivery_digest_on": {"ru": "Вывести дайджест в отдельный диалог", "en": "Send digest to separate chat"},
    "plus_delivery_digest_off": {"ru": "Отменить доставку дайджеста в отдельный диалог", "en": "Stop sending digest to separate chat"},
    "plus_delivery_storyline_on": {"ru": "Вывести отслеживание сюжетов в отдельный диалог", "en": "Send storyline tracking to separate chat"},
    "plus_delivery_storyline_off": {"ru": "Отменить доставку отслеживания сюжетов в отдельный диалог", "en": "Stop sending storyline tracking to separate chat"},
    "storyline_remove": {"ru": "Убрать сюжет", "en": "Remove storyline"},
    "storyline_clear_all": {"ru": "Очистить всё", "en": "Clear all"},
    "language_menu": {"ru": "Язык / Language", "en": "Language / Язык"},
    "admin_panel": {"ru": "Админ панель", "en": "Admin panel"},
    "storyline_debug_enable": {"ru": "Включить дебаг сторитрекинга", "en": "Enable storyline debug"},
    "storyline_debug_disable": {"ru": "Выключить дебаг сторитрекинга", "en": "Disable storyline debug"},
    "back": {"ru": "Назад", "en": "Back"},
    "pay_current_limit": {"ru": "Подключить Premium", "en": "Activate Premium"},
    "pay_tbank": {"ru": "Оплатить через T-Bank", "en": "Pay with T-Bank"},
    "pay_crypto": {"ru": "Оплатить криптой (USDT)", "en": "Pay with Crypto (USDT)"},
    "pay_stars": {"ru": "Оплатить через Telegram Stars", "en": "Pay with Telegram Stars"},
    "pay_term_30d": {"ru": "30 дней", "en": "30 days"},
    "pay_term_90d": {"ru": "90 дней", "en": "90 days"},
    "pay_term_180d": {"ru": "180 дней", "en": "180 days"},
    "pay_term_365d": {"ru": "365 дней", "en": "365 days"},
    "pay_plus_10": {"ru": "Оплатить +10 подписок", "en": "Pay +10 subscriptions"},
    "enter_promo": {"ru": "Ввести промокод", "en": "Enter promo code"},
    "refresh_payment": {"ru": "Обновить статус оплаты", "en": "Refresh payment status"},
    "filter_all": {"ru": "Всё", "en": "Everything"},
    "filter_not_interesting": {"ru": "Не пересылать неинтересные", "en": "Hide low-interest posts"},
    "filter_only_fire": {"ru": "Только огненные", "en": "Only hot posts"},
    "filter_digest_only": {
        "ru": "Отключить пересылку, буду читать только дайджест",
        "en": "Turn off live feed, keep digest only",
    },
    "reco_reset_start": {"ru": "Настроить рекомендации заново", "en": "Reset recommendations"},
    "reco_reset_confirm": {"ru": "Да, начать заново", "en": "Yes, start over"},
    "reco_reset_cancel": {"ru": "Отмена", "en": "Cancel"},
    "toggle_summary": {"ru": "Переключить саммаризацию", "en": "Toggle summarization"},
    "toggle_forwarding": {"ru": "Переключить пересылку", "en": "Toggle forwarding"},
    "instruction_filter": {"ru": "AI-фильтр: настройка", "en": "AI filter: setup"},
    "instruction_set_global": {"ru": "Задать global промпт", "en": "Set global prompt"},
    "instruction_set_group": {"ru": "Задать промпт для канала", "en": "Set channel prompt"},
    "instruction_show_rules": {"ru": "Показать активные правила", "en": "Show active rules"},
    "instruction_disable_global": {"ru": "Выключить global промпт", "en": "Disable global prompt"},
    "instruction_disable_group": {"ru": "Выключить промпт для канала", "en": "Disable channel prompt"},
    "forwarding_enable": {"ru": "Включить пересылку", "en": "Enable forwarding"},
    "forwarding_disable": {"ru": "Выключить пересылку", "en": "Disable forwarding"},
    "forwarding_filter_all": {"ru": "Фильтр: Всё", "en": "Filter: Everything"},
    "forwarding_filter_not_interesting": {
        "ru": "Фильтр: Не пересылать неинтересные",
        "en": "Filter: Hide low-interest posts",
    },
    "forwarding_filter_only_fire": {"ru": "Фильтр: Только огненные", "en": "Filter: Only hot posts"},
    "digest_disable": {"ru": "Выключить", "en": "Turn off"},
    "digest_toggle_on": {"ru": "Включить", "en": "On"},
    "digest_toggle_off": {"ru": "Выключить", "en": "Off"},
    "digest_setup_time": {"ru": "Настроить время", "en": "Set schedule"},
    "digest_daily": {"ru": "Раз в сутки", "en": "Once a day"},
    "digest_send_now": {"ru": "Получить сейчас", "en": "Send now"},
    "language_ru": {"ru": "Русский", "en": "Russian"},
    "language_en": {"ru": "English", "en": "English"},
    "news_time_option_under_hour": {"ru": "до часа", "en": "до часа"},
    "news_time_option_1h": {"ru": "1 час", "en": "1 час"},
    "news_time_option_2h": {"ru": "2 часа", "en": "2 часа"},
    "news_time_option_3h": {"ru": "3 часа", "en": "3 часа"},
    "news_time_option_4h": {"ru": "4 часа", "en": "4 часа"},
    "news_time_option_5h": {"ru": "5 часов", "en": "5 часов"},
    "news_time_saved_yes": {"ru": "Да", "en": "Да"},
    "news_time_saved_no": {"ru": "Нет", "en": "Нет"},
    "csi_score_1": {"ru": "1", "en": "1"},
    "csi_score_2": {"ru": "2", "en": "2"},
    "csi_score_3": {"ru": "3", "en": "3"},
    "csi_score_4": {"ru": "4", "en": "4"},
    "csi_score_5": {"ru": "5", "en": "5"},
    "csi_score_6": {"ru": "6", "en": "6"},
    "csi_score_7": {"ru": "7", "en": "7"},
    "csi_score_8": {"ru": "8", "en": "8"},
    "csi_score_9": {"ru": "9", "en": "9"},
    "csi_score_10": {"ru": "10", "en": "10"},
}


BUTTON_ALIASES: dict[str, list[str]] = {
    "filter_not_interesting": ["не пересылать не интересные"],
    "filter_only_fire": ["только огненые"],
    "reco_reset_start": ["Сбросить фильтр", "Обнулить веса фильтра", "Начать фильтр заново"],
    "reco_reset_confirm": ["Да начать заново", "Начать заново"],
    "reco_reset_cancel": ["Отменить"],
    "pay_current_limit": [
        "Оплатить сейчас",
        "Оплатить по текущему лимиту",
        "Pay now",
        "Pay current due",
    ],
    "pay_tbank": ["Подключить Premium", "Activate Premium"],
    "pay_crypto": ["Crypto Pay", "Оплатить через Crypto Pay"],
    "pay_stars": ["Stars", "Telegram Stars", "Оплатить через Stars"],
    "pay_plus_10": ["Оплатить +10 подписок", "Pay +10 subscriptions"],
    "toggle_summary": ["Саммаризация", "Summarization"],
    "digest_toggle_on": ["Дайджест: включить", "Digest: on"],
    "digest_toggle_off": ["Дайджест: выключить", "Digest: off"],
    "help": ["Помощь"],
    "support_request": ["Поддержка", "Обращение в техподдержку", "Support", "Contact support"],
    "forwarding": ["Live feed", "Фильтр", "Filter"],
    "instruction_filter": [
        # backward-compatible variants (old button text / user phrasing)
        "AI-фильтр",
        "AI filter",
        "AI-фильтр: включить",
        "AI-фильтр: выключить",
        "AI filter: on",
        "AI filter: off",
        "Instruction filter",
        "AI instruction",
        "AI-инструкция",
    ],
    "instruction_set_global": ["Задать global prompt", "Задать global промпт"],
    "instruction_set_group": ["Задать prompt для канала", "Задать промпт для канала"],
    "instruction_disable_global": ["Выключить global prompt", "Выключить global промпт"],
    "instruction_disable_group": ["Выключить prompt для канала", "Выключить промпт для канала"],
}


BUTTON_DECORATIONS = {
    "⚙️",
    "📘",
    "🆘",
    "📡",
    "📰",
    "🗞️",
    "🧭",
    "🌐",
    "🇷🇺",
    "🇬🇧",
    "🛠️",
    "🏦",
    "💵",
    "⭐",
    "🎁",
    "🔄",
    "🗓️",
    "📝",
    "🤖",
    "🎯",
    "▫️",
    "◀️",
    "🌍",
    "🧩",
    "📋",
    "⛔",
    "🧹",
    "⏰",
    "🚀",
    "✅",
    "❌",
}


TEXTS: dict[str, dict[str, str]] = {
    "welcome_message": {
        "ru": (
            "👋 **Привет!**\n\n"
            "Мы команда проекта NewsHub и мы делаем ваш личный агрегатор новостей в Telegram на основе AI. "
            "Мы хотим, чтоб наш продукт помог вам сэкономить время на чтение новостей. "
            "Наша цель - персонализировать и сосредоточить вашу ленту в этом боте и предоставить вам самые современные "
            "AI возможности обработки новостного потока.\n\n"
            "📌 **Как пользоваться ботом:**\n\n"
            "1️⃣ **Добавь паблик или группу**\n"
            "Просто укажи боту название интересующего тебя канала в формате `@имя_канала` или ссылку `https://t.me/имя_канала`.\n\n"
            "2️⃣ **Получай новости прямо в чат**\n"
            "Как только в выбранном канале что-то публикуется, бот пересылает свежий пост тебе.\n\n"
            "3️⃣ **Оценивай новости**\n"
            "Ставь «👍», «👎» или «🔥», чтобы бот понимал, какие темы тебе действительно важны.\n\n"
            "4️⃣ **Умная фильтрация от AI**\n"
            "Чем точнее твои оценки, тем лучше бот отбирает и присылает то, что соответствует твоим интересам.\n\n"
            "5️⃣ **Дайджест**\n"
            "Включи в настройках дайджест и укажи время доставки. Бот соберёт новости из твоих подписок, "
            "сгруппирует их по темам или сюжетам и пришлёт тебе одним сообщением в то время, в которое тебе будет удобно.\n\n"
            "6️⃣ **Отслеживание сюжета**\n"
            "Нажми под постом «Хронология сюжета», чтобы быстро получить краткую сводку того, как развивалась тема.\n"
            "А кнопка «Следить за сюжетом» позволит получать новые публикации по этой теме, даже если их нет в твоих текущих подписках.\n\n"
            "7️⃣ **Новые функции**\n"
            "Скоро в боте появятся новые умные возможности, которые помогут навести порядок в информационном потоке твоих подписок. "
            "Оставайся с нами, мы будем постепенно открывать доступ!"
        ),
        "en": (
            "👋 **Hi!**\n\n"
            "This is your personal **AI-powered Telegram news aggregator**. "
            "All the important updates from your favorite channels and groups — in one place. "
            "Cut the noise and keep only what actually matters.\n\n"
            "📌 **How to use the bot:**\n\n"
            "1️⃣ **Add a channel or group**\n"
            "Send a Telegram handle like `@channel_name` or a link like `https://t.me/channel_name`.\n\n"
            "2️⃣ **Receive news right in chat**\n"
            "As soon as something new appears in a selected channel, the bot forwards it to you.\n\n"
            "3️⃣ **Rate the news**\n"
            "Use «👍» or «👎» so the bot learns which topics really matter to you.\n\n"
            "4️⃣ **Smart AI filtering**\n"
            "The better your feedback, the better the bot will match and deliver posts aligned with your interests.\n\n"
            "✨ Try it now and enjoy a cleaner, smarter way to follow the news."
        ),
    },
    "plus_welcome_message": {
        "ru": (
            "Привет!\n\n"
            "Это второй бот проекта NewsHub. Сюда ты будешь получать доставки по дайджесту "
            "и отслеживаниям сюжета, чтобы они не терялись в общей ленте, если включишь такую "
            "опцию в настройках основного бота."
        ),
        "en": (
            "Hi!\n\n"
            "This is the second NewsHub bot. Digest deliveries and storyline-tracking updates can be sent here "
            "so they do not get lost in the main feed, once you enable this option in the main bot settings."
        ),
    },
    "help_message": {
        "ru": (
            "📘 **Инструкция**\n\n"
            "1. *Подписки:*\n"
            "Нажми Настройки → Подписки.\n\n"
            "Отправь `@канал` или `https://t.me/канал` (можно несколько строк).\n\n"
            "2. *Что делает \"Пересылка новостей\":*\n"
            "Бот пересылает тебе в этот чат посты из твоих подписок.\n"
            "В настройках можно включить саммаризацию. Это сжатие текста по смыслу. "
            "Либо оставить текст в исходном виде.\n\n"
            "3. *Реакции под постами:*\n"
            "Бот учится на твоих реакциях.\n"
            "👎 - эта тема мне не интересна. Я не хочу об этом читать.\n"
            "👍 - нормально. Новости по такой теме я не против получать.\n"
            "🔥 - хочу больше такого контента.\n\n"
            "Боту нужно немного времени, чтобы понять твои предпочтения. "
            "С каждой реакцией он становится точнее.\n\n"
            "4. *Что такое \"Дайджест\":*\n"
            "Это периодическая сводка постов из твоих подписок.\n"
            "Бот сгруппирует посты по темам (вне зависимости от реакций) и пришлёт краткую сводку.\n"
            "Включи в меню \"Дайджест\" и задай время получения.\n\n"
            "5. *Что делает фильтр в разделе \"Пересылка новостей\":*\n"
            "• Всё — бот пересылает все посты из подписок, не учитывая предпочтения.\n"
            "• Не пересылать неинтересные — бот старается не показывать то, что тебе не интересно.\n"
            "• Только огненные — бот пересылает только посты, которые лучше всего подходят под твои интересы.\n\n"
            "Также можно полностью отключить live-пересылку. "
            "Тогда ты можешь получать только дайджест по расписанию или пользоваться другими функциями бота.\n\n"
            "6. *Отслеживание сюжетов:*\n"
            "• Нажми \"Хронология сюжета\" под постом и бот даст тебе краткую сводку по сюжету.\n"
            "• Нажми \"Следить за сюжетом\" и бот будет пересылать тебе новые публикации по этому сюжету. "
            "В том числе и из каналов, которых нет у тебя в подписках.\n"
            "• Изменить список отслеживаемых сюжетов можно в настройках.\n\n"
            "7. *Отдельный диалог:*\n"
            "Зайдя в это меню в настройках, ты можешь переключить доставку дайджестов и карточек "
            "с отслеживанием сюжета в отдельное окно, чтобы они не терялись в общей ленте.\n\n"
            "8. *Тариф. Оплата.*\n"
            "• До 7 подписок. Бесплатно.\n"
            "• Premium можно включить на 30 / 90 / 180 / 365 дней от даты оплаты.\n"
            "• T-Bank. 490 / 1470 / 2940 / 5880 ₽.\n"
            "• Crypto Pay (USDT). 7 / 19 / 35 / 60 USDT.\n"
            "• Premium открывает расширенный лимит подписок и доступ к наиболее продвинутым AI возможностям. "
            "В том числе к тем, которые будут добавлены позже.\n\n"
            "9. *Промокоды.*\n"
            "• Промокод можно ввести в разделе Тариф / Оплата.\n"
            "• Обычно промокод включает Premium на подарочный срок. Чаще всего на 30 дней.\n\n"
            "10. Если у тебя возникли любые вопросы, пиши нам через кнопку в меню \"Обращение в поддержку\". "
            "Мы обязательно ответим!\n\n"
            "Отдельно хотим отметить, что на данный момент проект в стадии раннего доступа и всё ещё активно дорабатывается. "
            "Он будет становиться лучше, и в нём будут добавлены новые функции. "
            "Мы будем признательны за любую обратную связь через форму обращений!"
        ),
        "en": (
            "📘 **Help**\n\n"
            "1) Open **Settings** → **Subscriptions**  \n"
            "Send `@channel` or `https://t.me/channel` (multiple lines are supported).\n\n"
            "2) What **News forwarding** does  \n"
            "This is the live-feed section:\n"
            "- enable or disable forwarding\n"
            "- summarize long posts\n"
            "- apply a live-feed filter\n\n"
            "3) What **Digest** means  \n"
            "This is a periodic summary: a short list of the most important news in a time range.  \n"
            "Enable it in **Digest** and set a schedule.\n\n"
            "4) What **Filter** does inside **News forwarding**  \n"
            "- **Everything** — regular live forwarding  \n"
            "- **Hide low-interest posts** — softer filtering of weak posts\n"
            "- **Only hot posts** — only top posts\n"
            "- **Turn off live feed, keep digest only** — no live posts, only digest\n\n"
            "5) Reactions under a post  \n"
            "👍 / 👎 / 🔥 help personalize your feed.\n\n"
            "6) Plan / Payment  \n"
            "- Up to 7 subscriptions — free  \n"
            "- Premium can be activated for 30 / 90 / 180 / 365 days from the payment date  \n"
            "- T-Bank: 490 / 1470 / 2940 / 5880 ₽  \n"
            "- Crypto Pay (USDT): 7 / 19 / 35 / 60 USDT  \n"
            "- Premium unlocks an expanded subscription limit\n\n"
            "7) Promo codes  \n"
            "- You can enter a promo code in **Plan / Payment**  \n"
            "- A promo code usually activates gifted Premium, most often for 30 days"
        ),
    },
    "plus_help_message": {
        "ru": (
            "📘 *Инструкция*\n\n"
            "Основной смысл второго окна — не терять обновления сюжета и дайджесты в общей ленте "
            "основного бота.\n\n"
            "Если ты нажмешь «Хронология сюжета» в этом окне, то ответ получишь тоже сюда.\n\n"
            "В остальном все настройки твоего аккаунта остаются в основном боте.\n\n"
            "Отключить пересылку в это окно и вернуть ее в основное можно в настройках основного бота."
        ),
        "en": (
            "📘 *Guide*\n\n"
            "The purpose of this second chat is to keep storyline updates and digests from getting lost in the main bot feed.\n\n"
            "If you tap “Storyline timeline” here, the reply will also arrive here.\n\n"
            "All other account settings stay in the main bot.\n\n"
            "You can disable forwarding to this chat and switch it back in the main bot settings."
        ),
    },
    "support_request_prompt": {
        "ru": "Напишите свой вопрос и команда NewsHub обязательно свяжется с вами в течение 24 часов.",
        "en": "Please write your question, and the NewsHub team will contact you within 24 hours.",
    },
    "support_request_received": {
        "ru": "Спасибо! Ваш вопрос принят. Ожидайте ответа от команды.",
        "en": "Thanks! Your request is accepted. Please wait for the team's response.",
    },
    "support_request_queued": {
        "ru": "Спасибо! Обращение зарегистрировано. Сейчас канал поддержки временно недоступен, но мы доставим запрос автоматически при восстановлении.",
        "en": "Thanks! Your request is registered. Support delivery is temporarily unavailable, and we will deliver it automatically after recovery.",
    },
    "nlu_settings_overview_title": {
        "ru": "Краткая сводка настроек:",
        "en": "Settings overview:",
    },
    "nlu_confirm_button": {"ru": "Подтвердить", "en": "Confirm"},
    "nlu_cancel_button": {"ru": "Отменить", "en": "Cancel"},
    "nlu_action_expired": {
        "ru": "Это подтверждение уже устарело. Отправь команду ещё раз.",
        "en": "This confirmation has expired. Please send the request again.",
    },
    "nlu_action_cancelled": {
        "ru": "Действие отменено. Ничего не менял.",
        "en": "Action cancelled. Nothing was changed.",
    },
    "nlu_action_failed": {
        "ru": "Не получилось применить действие. Попробуй ещё раз.",
        "en": "Could not apply the action. Please try again.",
    },
    "nlu_action_store_failed": {
        "ru": "Не смог сохранить подтверждение действия. Попробуй ещё раз чуть позже.",
        "en": "Could not store the pending action. Please try again shortly.",
    },
    "nlu_action_not_supported": {
        "ru": "Это действие пока не поддерживается в текстовом управлении.",
        "en": "This action is not supported in text control yet.",
    },
    "nlu_clarify_generic": {
        "ru": "Уточни, пожалуйста, что именно нужно изменить.",
        "en": "Please clarify what exactly you want to change.",
    },
    "nlu_clarify_filter_mode": {
        "ru": "Какой фильтр включить?\n\nВарианты:\n- Всё\n- Не пересылать неинтересные\n- Только огненные\n- Только дайджест\n\nКоротко:\n- «Не пересылать неинтересные» — мягкий фильтр, скрывает слабые и нерелевантные посты.\n- «Только огненные» — строгий фильтр, оставляет только самые сильные сигналы.",
        "en": "Which filter should I set?\n\nOptions:\n- Everything\n- Hide low-interest posts\n- Only hot posts\n- Digest only\n\nIn short:\n- “Hide low-interest posts” is a softer filter that removes weak and less relevant posts.\n- “Only hot posts” is a stricter filter that keeps only the strongest signals.",
    },
    "nlu_clarify_importance_filter": {
        "ru": "Ты имеешь в виду обычный фильтр «Только огненные» или AI-фильтр с правилом «только важное»?",
        "en": "Do you mean the regular “Only hot posts” filter, or an AI filter with the rule “only important”?",
    },
    "nlu_confirm_generic": {
        "ru": "Подтверди действие.",
        "en": "Please confirm the action.",
    },
    "nlu_confirm_add_subscriptions": {
        "ru": "Добавлю подписки:\n{links}\n\nПодтверди действие.",
        "en": "I will add these subscriptions:\n{links}\n\nPlease confirm.",
    },
    "nlu_confirm_remove_subscriptions": {
        "ru": "Удалю подписки:\n{links}\n\nПодтверди действие.",
        "en": "I will remove these subscriptions:\n{links}\n\nPlease confirm.",
    },
    "nlu_confirm_set_filter": {
        "ru": "Изменю фильтр live-ленты на: {filter_label}.\n\nПодтверди действие.",
        "en": "I will change the live-feed filter to: {filter_label}.\n\nPlease confirm.",
    },
    "nlu_confirm_toggle_forwarding": {
        "ru": "Переключу live-пересылку: {state}.\n\nПодтверди действие.",
        "en": "I will switch live forwarding: {state}.\n\nPlease confirm.",
    },
    "nlu_confirm_toggle_summary": {
        "ru": "Переключу саммаризацию: {state}.\n\nПодтверди действие.",
        "en": "I will switch summarization: {state}.\n\nPlease confirm.",
    },
    "nlu_confirm_set_global_instruction_filter": {
        "ru": "Сохраню global AI-фильтр:\n{prompt_text}\n\nОн будет применяться ко всем подпискам, где нет отдельного prompt для канала.\n\nПодтверди действие.",
        "en": "I will save the global AI filter:\n{prompt_text}\n\nIt will apply to all subscriptions without a separate channel prompt.\n\nPlease confirm.",
    },
    "nlu_confirm_digest_enable": {
        "ru": "Включу дайджест.\n\nПодтверди действие.",
        "en": "I will enable the digest.\n\nPlease confirm.",
    },
    "nlu_confirm_digest_disable": {
        "ru": "Выключу дайджест.\n\nПодтверди действие.",
        "en": "I will disable the digest.\n\nPlease confirm.",
    },
    "nlu_confirm_digest_send_now": {
        "ru": "Соберу и отправлю дайджест за последние 24 часа.\n\nПодтверди действие.",
        "en": "I will build and send the last-24h digest.\n\nPlease confirm.",
    },
    "nlu_confirm_digest_set_time": {
        "ru": "Установлю время дайджеста: {time_hhmm}.\n\nПодтверди действие.",
        "en": "I will set the digest time to: {time_hhmm}.\n\nPlease confirm.",
    },
    "nlu_confirm_digest_set_offset": {
        "ru": "Установлю UTC-смещение: {utc_offset}.\n\nПодтверди действие.",
        "en": "I will set the UTC offset to: {utc_offset}.\n\nPlease confirm.",
    },
    "nlu_confirm_set_language": {
        "ru": "Переключу язык интерфейса на: {language_code}.\n\nПодтверди действие.",
        "en": "I will switch the interface language to: {language_code}.\n\nPlease confirm.",
    },
    "nlu_help_subscriptions": {
        "ru": "Подписки: {active}/{allowed}.\n\nТекущий список:\n{links}\n\nМожно отправлять @канал или https://t.me/канал. Если хочешь именно добавить или удалить через текст, сформулируй это явно.",
        "en": "Subscriptions: {active}/{allowed}.\n\nCurrent list:\n{links}\n\nYou can send @channel or https://t.me/channel. If you want to explicitly add or remove through text, say it directly.",
    },
    "nlu_help_settings": {
        "ru": "В настройках можно управлять подписками, live-пересылкой, дайджестом, языком, тарифом и дополнительными AI-настройками. Можно спросить прямо по разделу: например, «что такое дайджест?» или «как работает AI-фильтр?».",
        "en": "Settings let you manage subscriptions, live forwarding, digest, language, billing, and extra AI options. You can ask directly about a section, for example: “what is digest?” or “how does the AI filter work?”.",
    },
    "nlu_help_forwarding": {
        "ru": "Здесь управляются live-пересылка, фильтр и саммаризация. Доступные режимы фильтра: «Всё», «Не пересылать неинтересные», «Только огненные» и «Только дайджест». Можно писать фразы вроде «включи пересылку», «только огненные» или «выключи саммаризацию».",
        "en": "This section controls live forwarding, filtering, and summarization. Available filter modes are: “Everything”, “Hide low-interest posts”, “Only hot posts”, and “Digest only”. You can say things like “enable forwarding”, “only hot posts”, or “disable summarization”.",
    },
    "nlu_help_summary": {
        "ru": "Саммаризация сокращает длинные пересылаемые посты до 1–3 предложений, чтобы быстро понять суть новости. Сейчас она {state}. Можно написать «включи саммаризацию» или «выключи саммаризацию».",
        "en": "Summarization shortens long forwarded posts to 1-3 sentences so you can grasp the main point quickly. It is currently {state}. You can say “enable summarization” or “disable summarization”.",
    },
    "nlu_help_filters": {
        "ru": "Режимы фильтра live-ленты:\n- Всё — пересылать все новые посты.\n- Не пересылать неинтересные — мягко отсеивать слабые и нерелевантные посты.\n- Только огненные — оставлять только самые сильные сигналы.\n- Только дайджест — выключить live-пересылку и читать только дайджест.\n\nЕсли хочешь, можешь сразу выбрать режим кнопкой ниже или написать его текстом.",
        "en": "Live-feed filter modes:\n- Everything — forward all new posts.\n- Hide low-interest posts — softly filter out weaker and less relevant posts.\n- Only hot posts — keep only the strongest signals.\n- Digest only — turn off live forwarding and read only the digest.\n\nIf you want, choose a mode with the buttons below or type it in plain text.",
    },
    "nlu_help_digest": {
        "ru": "Для дайджеста можно написать «включи дайджест», «поставь дайджест на 20:30» или «смещение +3».",
        "en": "For digest control you can say “enable digest”, “set digest to 20:30”, or “offset +3”.",
    },
    "nlu_help_language": {
        "ru": "Текущий язык: {current_language}.\n\nМожно написать «переключи язык на русский» или «switch language to English».",
        "en": "Current language: {current_language}.\n\nYou can say “switch language to English” or “переключи язык на русский”.",
    },
    "nlu_help_ai_filter": {
        "ru": "AI-фильтр добавляет поверх обычного фильтра свои инструкции: можно задать global prompt для всех подписок и отдельный prompt для конкретного канала. Например: «показывай только новости по экономике и политике». Функция доступна в Premium или в первую неделю после регистрации.",
        "en": "AI filter adds its own instructions on top of the regular filter: you can set a global prompt for all subscriptions and a separate prompt for a specific channel. For example: “show only economy and politics news”. This feature is available in Premium or during the first week after registration.",
    },
    "nlu_help_storyline_tracking": {
        "ru": "Отслеживание сюжета помогает следить за развитием одной темы во времени: бот собирает timeline и новые апдейты по выбранному сюжету. Здесь можно посмотреть активные сюжеты и перестать следить за ненужными.",
        "en": "Storyline tracking helps you follow how one topic evolves over time: the bot builds a timeline and sends new updates for the selected storyline. Here you can review active storylines and stop following unneeded ones.",
    },
    "nlu_billing_cta_due": {
        "ru": "Сейчас есть сумма к оплате: {due_amount} ₽. Если хочешь, открой раздел «Тариф / Оплата» и подключи Premium.",
        "en": "There is an amount due now: {due_amount} ₽. If you want, open “Plan / Payment” and activate Premium.",
    },
    "nlu_billing_cta_low_reserve": {
        "ru": "Запас по подпискам почти закончился: осталось {reserve}. При необходимости можно заранее открыть «Тариф / Оплата».",
        "en": "Your subscription reserve is almost exhausted: {reserve} left. If needed, you can open “Plan / Payment” in advance.",
    },
    "start_required": {"ru": "Сначала /start.", "en": "Run /start first."},
    "start_required_register": {
        "ru": "Сначала нажми /start, чтобы я тебя зарегистрировал.",
        "en": "Press /start first so I can register you.",
    },
    "invalid_digest_button": {"ru": "Некорректная кнопка", "en": "Invalid button"},
    "digest_stale": {"ru": "Дайджест устарел", "en": "This digest has expired"},
    "settings_title": {"ru": "Настройки:", "en": "Settings:"},
    "main_menu_title": {"ru": "Главное меню:", "en": "Main menu:"},
    "digest_menu_title": {"ru": "Меню дайджеста:", "en": "Digest menu:"},
    "language_menu_title": {
        "ru": "Выбери язык интерфейса:",
        "en": "Choose the interface language:",
    },
    "language_changed": {
        "ru": "Язык интерфейса переключен на русский.",
        "en": "Interface language switched to English.",
    },
    "language_unknown": {
        "ru": "Не понял выбор языка.",
        "en": "I could not understand the language choice.",
    },
    "news_time_baseline_question": {
        "ru": "Помогите нам сделать NewsHub полезнее: ответьте, пожалуйста, на короткий вопрос.\n\nСколько времени в день обычно уходит на поиск и чтение новостей в Telegram?",
        "en": "Сколько времени в день вы тратите на поиск и чтение новостей в Telegram?",
    },
    "news_time_saved_question": {
        "ru": "Ответьте, пожалуйста, на ещё один вопрос. Стали ли вы меньше тратить времени на чтение новостей благодаря боту?",
        "en": "Стали ли вы меньше тратить времени на поиск и чтение новостей благодаря боту?",
    },
    "news_time_current_question": {
        "ru": "Сколько времени в день вы теперь тратите на новости?",
        "en": "Сколько времени в день вы теперь тратите на новости?",
    },
    "news_time_thanks": {
        "ru": "Спасибо за ответ!",
        "en": "Спасибо! Записал ответ.",
    },
    "csi_question": {
        "ru": "Ответьте, пожалуйста, насколько вы довольны нашим ботом для мониторинга новостей?",
        "en": "Ответьте, пожалуйста, насколько вы довольны нашим ботом для мониторинга новостей?",
    },
    "csi_thanks": {
        "ru": "Спасибо! Записал оценку.",
        "en": "Спасибо! Записал оценку.",
    },
    "forwarding_intro": {
        "ru": "<b>📩 Пересылка новостей</b>\n\n<b>Текущие настройки:</b>",
        "en": "<b>📩 News Forwarding</b>\n\n<b>Current settings:</b>",
    },
    "forwarding_live_label": {"ru": "Live-пересылка", "en": "Live forwarding"},
    "forwarding_filter_label": {"ru": "Фильтр", "en": "Filter"},
    "forwarding_summary_label": {
        "ru": "Саммаризация",
        "en": "Summarization",
    },
    "forwarding_live_on": {"ru": "[включена]", "en": "[enabled]"},
    "forwarding_live_off": {"ru": "[выключена]", "en": "[disabled]"},
    "forwarding_summary_on": {"ru": "[включена]", "en": "[enabled]"},
    "forwarding_summary_off": {"ru": "[выключена]", "en": "[disabled]"},
    "forwarding_summary_effect_on": {
        "ru": "Теперь пересылаемые новости будут сокращаться до 1–3 предложений.",
        "en": "Forwarded news will now be shortened to 1-3 sentences.",
    },
    "forwarding_summary_effect_off": {
        "ru": "Теперь пересылаемые новости не будут сокращаться до 1–3 предложений.",
        "en": "Forwarded news will no longer be shortened to 1-3 sentences.",
    },
    "forwarding_menu_footer": {
        "ru": (
            "⚙️ <b>Здесь можно:</b>\n"
            "• Включать и выключать live-пересылку из выбранных каналов\n"
            "• Управлять саммаризацией новости (сокращение новости до 1-3 предложений)\n"
            "• Настраивать фильтр новостей\n"
            "• Добавлять AI-фильтр (Premium или первая неделя после регистрации)"
        ),
        "en": (
            "⚙️ <b>Here you can:</b>\n"
            "• Turn live forwarding from selected channels on or off\n"
            "• Control news summarization (shortening news to 1-3 sentences)\n"
            "• Adjust the news filter\n"
            "• Add an AI filter (Premium or first week after registration)"
        ),
    },
    "instruction_intro": {
        "ru": (
            "<b>🤖 AI-фильтр</b>\n\n"
            "<b>Как это работает:</b>\n"
            "• Доступно в Premium или в первую неделю после регистрации\n"
            "• Global промпт действует на все каналы\n"
            "• Промпт для канала заменяет global промпт\n"
            "• AI-фильтр работает поверх обычных фильтров"
        ),
        "en": (
            "<b>🤖 AI Filter</b>\n\n"
            "<b>How it works:</b>\n"
            "• Available in Premium or during the first week after registration\n"
            "• A global prompt applies to all channels\n"
            "• A channel prompt overrides the global prompt\n"
            "• The AI filter works on top of the regular filters"
        ),
    },
    "instruction_rules_title": {"ru": "Активные правила:", "en": "Active rules:"},
    "instruction_no_rules": {
        "ru": "📭 <b>Активных AI-правил пока нет.</b>",
        "en": "📭 <b>There are no active AI rules yet.</b>",
    },
    "instruction_global_rule_item": {
        "ru": "Global: {prompt}",
        "en": "Global: {prompt}",
    },
    "instruction_group_rule_item": {
        "ru": "{group}: {prompt}",
        "en": "{group}: {prompt}",
    },
    "instruction_premium_required": {
        "ru": "Пробная неделя AI-фильтра закончилась. Подключи Premium, чтобы снова задавать AI-инструкции для live-ленты.",
        "en": "Your free AI filter week has ended. Activate Premium to add AI instructions for the live feed again.",
    },
    "instruction_global_prompt_request": {
        "ru": "Отправь одним сообщением global prompt. Он будет применяться ко всем каналам, где нет локального override.",
        "en": "Send the global prompt in one message. It will apply to all channels without a local override.",
    },
    "instruction_group_prompt_choose": {
        "ru": "Выбери канал кнопкой ниже. Если список длинный, пролистай его стрелками.",
        "en": "Choose a channel using the buttons below. If the list is long, switch pages with the arrows.",
    },
    "instruction_group_prompt_request": {
        "ru": "Отправь prompt для выбранного канала одним сообщением. Он полностью заменит global prompt только для этого канала.",
        "en": "Send a prompt for the selected channel in one message. It will fully override the global prompt for that channel only.",
    },
    "instruction_group_prompt_request_selected": {
        "ru": "Канал выбран: {group}\n\nТеперь отправь prompt одним сообщением. Он полностью заменит global prompt только для этого канала.",
        "en": "Selected channel: {group}\n\nNow send the prompt in one message. It will fully override the global prompt for that channel only.",
    },
    "instruction_group_no_subscriptions": {
        "ru": "Сначала нужна хотя бы одна активная подписка, чтобы задать prompt для канала.",
        "en": "You need at least one active subscription before setting a channel prompt.",
    },
    "instruction_global_saved": {
        "ru": "Global prompt сохранен. Теперь он будет применяться ко всем каналам без локального override.",
        "en": "The global prompt has been saved. It will now apply to all channels without a local override.",
    },
    "instruction_group_saved": {
        "ru": "Prompt для канала сохранен. Для этого канала он теперь заменяет global prompt.",
        "en": "The channel prompt has been saved. It now overrides the global prompt for this channel.",
    },
    "instruction_disable_choose": {
        "ru": "Выбери номер правила для отключения:",
        "en": "Reply with the number of the rule to disable:",
    },
    "instruction_group_disable_choose": {
        "ru": "Выбери канал кнопкой ниже, чтобы выключить для него prompt. Если список длинный, используй стрелки.",
        "en": "Choose a channel using the buttons below to disable its prompt. If the list is long, use the arrows.",
    },
    "instruction_global_disabled": {
        "ru": "Global prompt выключен.",
        "en": "The global prompt has been disabled.",
    },
    "instruction_global_already_disabled": {
        "ru": "Global prompt уже был выключен.",
        "en": "The global prompt was already disabled.",
    },
    "instruction_group_disabled": {
        "ru": "Prompt для канала выключен.",
        "en": "The channel prompt has been disabled.",
    },
    "instruction_group_rules_missing": {
        "ru": "Для каналов пока нет активных prompt-правил.",
        "en": "There are no active channel prompt rules yet.",
    },
    "instruction_invalid_choice": {
        "ru": "Не понял выбор. Попробуй ещё раз через кнопки списка.",
        "en": "I could not understand the choice. Please try again using the picker buttons.",
    },
    "instruction_picker_prev": {"ru": "←", "en": "←"},
    "instruction_picker_next": {"ru": "→", "en": "→"},
    "instruction_picker_expired": {
        "ru": "Этот список выбора уже устарел. Открой его заново из меню AI-инструкции.",
        "en": "This picker has expired. Open it again from the AI instruction menu.",
    },
    "instruction_soft_bypass_notice": {
        "ru": "[Сервисная пометка] Instruction-фильтр временно недоступен, пост доставлен только по основному AI-фильтру.",
        "en": "[Service note] The instruction filter is temporarily unavailable, so this post was delivered using only the main AI filter.",
    },
    "empty_prompt": {
        "ru": "Prompt не должен быть пустым.",
        "en": "The prompt must not be empty.",
    },
    "prompt_too_long": {
        "ru": "Prompt получился слишком длинным. Укороти его и попробуй снова.",
        "en": "The prompt is too long. Please shorten it and try again.",
    },
    "feed_filter_all": {"ru": "[всё]", "en": "[everything]"},
    "feed_filter_not_interesting": {"ru": "[не пересылать неинтересные]", "en": "[hide low-interest posts]"},
    "feed_filter_only_fire": {"ru": "[только огненные]", "en": "[only hot posts]"},
    "feed_filter_digest_only": {"ru": "[только дайджест]", "en": "[digest only]"},
    "billing_title": {"ru": "💳 Тариф / Оплата", "en": "💳 Plan / Payment"},
    "billing_active_subscriptions": {"ru": "Сейчас подписок", "en": "Current subscriptions"},
    "billing_free_limit": {"ru": "Включено бесплатно", "en": "Included for free"},
    "billing_pack_size": {"ru": "Размер пакета", "en": "Package size"},
    "billing_pack_price": {"ru": "Цена пакета", "en": "Package price"},
    "billing_subscriptions_word": {"ru": "подписок", "en": "subscriptions"},
    "billing_required_packs": {"ru": "Требуется пакетов", "en": "Packages required"},
    "billing_paid_packs": {"ru": "Оплачено пакетов", "en": "Packages paid"},
    "billing_premium_price": {"ru": "Premium", "en": "Premium"},
    "billing_premium_status": {"ru": "Ваш тариф", "en": "Your plan"},
    "billing_premium_active": {"ru": "Premium", "en": "Premium"},
    "billing_premium_inactive": {"ru": "бесплатный", "en": "Free"},
    "billing_premium_unlocked": {
        "ru": (
            "Теперь вам открыты:\n"
            "• Безлимит на подписки\n"
            "• Отслеживание сюжетов и тем\n"
            "• Умная AI-фильтрация новостей"
        ),
        "en": (
            "Now available to you:\n"
            "• Unlimited subscriptions\n"
            "• Storyline and topic tracking\n"
            "• Smart AI news filtering"
        ),
    },
    "billing_premium_expires_at": {"ru": "Оплачен до", "en": "Paid until"},
    "billing_promo_groups": {"ru": "Legacy-доп. группы по промокодам", "en": "Legacy extra groups from promo codes"},
    "billing_allowed_subscriptions": {"ru": "Лимит после оплат", "en": "Limit after payments"},
    "billing_reserve_subscriptions": {"ru": "Запас по подпискам", "en": "Subscription reserve"},
    "billing_due_amount": {"ru": "Минимум к оплате сейчас", "en": "Minimum due now"},
    "billing_status": {"ru": "Статус", "en": "Status"},
    "billing_next_period_status": {"ru": "Следующий месяц ({period})", "en": "Next month ({period})"},
    "billing_next_period_prepaid": {"ru": "уже оплачен", "en": "already prepaid"},
    "billing_next_period_not_prepaid": {"ru": "ещё не оплачен", "en": "not prepaid yet"},
    "billing_unknown": {"ru": "unknown", "en": "unknown"},
    "billing_status_free": {"ru": "бесплатно", "en": "free"},
    "billing_status_paid": {"ru": "оплачено", "en": "paid"},
    "billing_status_payment_required": {"ru": "нужна оплата", "en": "payment required"},
    "billing_status_degraded": {"ru": "ограничено", "en": "limited"},
    "billing_instruction": {
        "ru": (
            "<b>💎 Что открывает Premium:</b>\n"
            "• Безлимит на подписки\n"
            "• Отслеживание сюжетов и тем\n"
            "• Умная AI-фильтрация новостей\n\n"
            "Срок действия Premium начинается от даты оплаты.\n\n"
            "💳 Оплата:\n"
            "• T-Bank: 490 / 1470 / 2940 / 5880 ₽\n"
            "• Crypto (USDT): 7 / 19 / 35 / 60 USDT\n"
            "• Telegram Stars: 400 / 1120 / 2000 / 3600 XTR"
        ),
        "en": (
            "<b>💎 Premium unlocks:</b>\n"
            "• Unlimited subscriptions\n"
            "• Story and topic tracking\n"
            "• Smart AI news filtering\n\n"
            "Premium starts from the payment date.\n\n"
            "💳 Payment:\n"
            "• T-Bank: 490 / 1470 / 2940 / 5880 ₽\n"
            "• Crypto (USDT): 7 / 19 / 35 / 60 USDT\n"
            "• Telegram Stars: 400 / 1120 / 2000 / 3600 XTR"
        ),
    },
    "billing_pending_link": {"ru": "Текущая ссылка на оплату:", "en": "Current payment link:"},
    "billing_pending_payment": {
        "ru": "Текущий незавершённый платёж: {provider} — {amount} {currency}.",
        "en": "Current pending payment: {provider} — {amount} {currency}.",
    },
    "billing_pending_period": {
        "ru": "Срок Premium: {period_start} → {period_end} ({days} дней).",
        "en": "Premium term: {period_start} → {period_end} ({days} days).",
    },
    "payment_provider_tbank": {"ru": "T-Bank", "en": "T-Bank"},
    "payment_provider_cryptopay": {"ru": "Crypto Pay", "en": "Crypto Pay"},
    "payment_provider_stars": {"ru": "Telegram Stars", "en": "Telegram Stars"},
    "payment_crypto_prompt": {
        "ru": (
            "<b>💳 Оплата через Crypto (USDT)</b>\n\n"
            "Вы выбрали оплату криптовалютой через Crypto Pay.\n\n"
            "<b>Стоимость:</b>\n"
            "• 7 USDT — 30 дней\n"
            "• 19 USDT — 90 дней\n"
            "• 35 USDT — 180 дней\n"
            "• 60 USDT — 365 дней\n\n"
            "<b>Как оплатить:</b>\n"
            "1. Нажмите кнопку ниже с нужным сроком и откройте ссылку\n"
            "2. Завершите оплату в Crypto Pay\n"
            "3. Вернитесь в бот и нажмите <b>«Обновить статус оплаты»</b>\n\n"
            "⚠️ После оплаты доступ активируется не всегда мгновенно — обязательно обновите статус вручную"
        ),
        "en": (
            "<b>💳 Pay with Crypto (USDT)</b>\n\n"
            "You chose cryptocurrency payment via Crypto Pay.\n\n"
            "<b>Pricing:</b>\n"
            "• 7 USDT — 30 days\n"
            "• 19 USDT — 90 days\n"
            "• 35 USDT — 180 days\n"
            "• 60 USDT — 365 days\n\n"
            "<b>How to pay:</b>\n"
            "1. Tap the button below for the needed term and open the link\n"
            "2. Complete the payment in Crypto Pay\n"
            "3. Return to the bot and tap <b>“Refresh payment status”</b>\n\n"
            "⚠️ Access is not always activated instantly after payment — make sure to refresh the status manually"
        ),
    },
    "payment_tbank_prompt": {
        "ru": (
            "<b>💳 Оплата через T-Bank</b>\n\n"
            "Вы выбрали оплату банковской картой.\n\n"
            "<b>Стоимость:</b>\n"
            "• 490 ₽ — 30 дней\n"
            "• 1470 ₽ — 90 дней\n"
            "• 2940 ₽ — 180 дней\n"
            "• 5880 ₽ — 365 дней\n\n"
            "<b>Как оплатить:</b>\n"
            "1. Нажмите кнопку ниже с нужным сроком и откройте ссылку\n"
            "2. Оплатите любым удобным способом: карта или СБП\n"
            "3. После оплаты нажмите <b>«Обновить статус оплаты»</b>\n\n"
            "⚡ Обычно доступ активируется автоматически, но если статус не обновился — нажмите кнопку вручную"
        ),
        "en": (
            "<b>💳 Pay with T-Bank</b>\n\n"
            "You chose bank card payment.\n\n"
            "<b>Pricing:</b>\n"
            "• 490 ₽ — 30 days\n"
            "• 1470 ₽ — 90 days\n"
            "• 2940 ₽ — 180 days\n"
            "• 5880 ₽ — 365 days\n\n"
            "<b>How to pay:</b>\n"
            "1. Tap the button below for the needed term and open the link\n"
            "2. Pay in any convenient way: card or Faster Payments\n"
            "3. After payment, tap <b>“Refresh payment status”</b>\n\n"
            "⚡ Access is usually activated automatically, but if the status did not update — refresh it manually"
        ),
    },
    "payment_stars_prompt": {
        "ru": (
            "<b>💳 Оплата через Telegram Stars</b>\n\n"
            "Оплата проходит прямо внутри Telegram.\n\n"
            "<b>Стоимость:</b>\n"
            "• 400 Stars — 30 дней\n"
            "• 1120 Stars — 90 дней\n"
            "• 2000 Stars — 180 дней\n"
            "• 3600 Stars — 365 дней\n\n"
            "<b>Как оплатить:</b>\n"
            "1. Нажмите кнопку с нужным сроком\n"
            "2. Подтвердите счёт в Telegram\n"
            "3. Premium активируется автоматически после успешной оплаты"
        ),
        "en": (
            "<b>💳 Pay with Telegram Stars</b>\n\n"
            "The payment happens directly inside Telegram.\n\n"
            "<b>Pricing:</b>\n"
            "• 400 Stars — 30 days\n"
            "• 1120 Stars — 90 days\n"
            "• 2000 Stars — 180 days\n"
            "• 3600 Stars — 365 days\n\n"
            "<b>How to pay:</b>\n"
            "1. Tap the needed term button\n"
            "2. Confirm the invoice in Telegram\n"
            "3. Premium activates automatically after successful payment"
        ),
    },
    "payment_method_selected": {
        "ru": "Способ оплаты: {provider}.",
        "en": "Payment method: {provider}.",
    },
    "payment_quote_selected": {
        "ru": "Стоимость: {amount} {currency}.",
        "en": "Price: {amount} {currency}.",
    },
    "payment_term_selected": {
        "ru": "Срок: {days} дней.",
        "en": "Term: {days} days.",
    },
    "payment_link_replaced": {
        "ru": "Предыдущая ссылка на оплату больше не активна. Действительной считается только последняя созданная ссылка.",
        "en": "The previous payment link is no longer active. Only the most recently created link should be used.",
    },
    "payment_crypto_polling_hint": {
        "ru": "⚠️ После оплаты доступ может обновиться не сразу — вернись в бот и нажми «Обновить статус оплаты».",
        "en": "⚠️ Access may not update instantly after payment — return to the bot and tap “Refresh payment status”.",
    },
    "payment_stars_sent": {
        "ru": "Счёт в Telegram Stars отправлен отдельным сообщением выше. После успешной оплаты Premium включится автоматически.",
        "en": "The Telegram Stars invoice was sent in a separate message above. Premium will activate automatically after successful payment.",
    },
    "payment_stars_pending_hint": {
        "ru": "Счёт создан внутри Telegram. Если окно оплаты закрылось, создай новый счёт из меню «Тариф / Оплата».",
        "en": "The invoice was created inside Telegram. If you closed it, create a new one from “Plan / Payment”.",
    },
    "payment_stars_invoice_title": {
        "ru": "Premium на {days} дней",
        "en": "Premium for {days} days",
    },
    "payment_stars_invoice_description": {
        "ru": "Доступ Premium на {days} дней от даты оплаты.",
        "en": "Premium access for {days} days from the payment date.",
    },
    "payment_stars_precheckout_failed": {
        "ru": "Не удалось подтвердить счёт Telegram Stars. Создай новый счёт и попробуй ещё раз.",
        "en": "Could not validate the Telegram Stars invoice. Create a new invoice and try again.",
    },
    "payment_stars_success": {
        "ru": "Оплата через Telegram Stars подтверждена. Premium активирован.",
        "en": "Telegram Stars payment confirmed. Premium is active.",
    },
    "promo_enter": {"ru": "Введи промокод (например: SPRING10).", "en": "Enter a promo code (for example: SPRING10)."},
    "promo_applied": {
        "ru": "Промокод применен: +{extra_groups} групп (до {expires}).\n\n{status}",
        "en": "Promo code applied: +{extra_groups} groups (until {expires}).\n\n{status}",
    },
    "promo_premium_applied": {
        "ru": "Промокод применен: Premium на {days} дней, активен до {expires}.\n\n{status}",
        "en": "Promo code applied: Premium for {days} days, active until {expires}.\n\n{status}",
    },
    "promo_no_expiry": {"ru": "без срока", "en": "no expiry"},
    "promo_apply_failed": {"ru": "Не удалось применить промокод: {error}", "en": "Could not apply promo code: {error}"},
    "payment_not_required": {
        "ru": "На текущий месяц доплата не требуется. Лимит уже покрыт.",
        "en": "No extra payment is needed for the current month. Your limit is already covered.",
    },
    "payment_premium_already_active": {
        "ru": "Premium уже активен.",
        "en": "Premium is already active.",
    },
    "payment_premium_already_prepaid": {
        "ru": "Следующий месяц ({period}) уже оплачен. Двойное продление сейчас не нужно.",
        "en": "The next month ({period}) is already prepaid. Another renewal is not needed right now.",
    },
    "payment_target_period": {
        "ru": "Период доступа: {period_start} → {period_end} ({days} дней).",
        "en": "Access period: {period_start} → {period_end} ({days} days).",
    },
    "payment_reserve_selected": {
        "ru": "Выбран запас: +{count} подписок.",
        "en": "Selected reserve: +{count} subscriptions.",
    },
    "payment_link_title": {"ru": "Ссылка для оплаты:", "en": "Payment link:"},
    "button_selected_suffix": {"ru": "[выбрано]", "en": "[selected]"},
    "payment_create_failed": {"ru": "Не удалось создать платеж: {error}", "en": "Could not create payment: {error}"},
    "payment_option_disabled": {
        "ru": "Сейчас доступны только стандартные сроки Premium: 30 / 90 / 180 / 365 дней.",
        "en": "Only the standard Premium terms are available right now: 30 / 90 / 180 / 365 days.",
    },
    "payment_select_method_first": {
        "ru": "Сначала выбери способ оплаты в разделе «Тариф / Оплата».",
        "en": "Choose a payment method first in “Plan / Payment”.",
    },
    "subscriptions_none": {
        "ru": "Пока подписок нет. Отправь @канал или ссылку https://t.me/канал, и я подпишу тебя.",
        "en": "You do not have any subscriptions yet. Send @channel or a https://t.me/channel link and I will subscribe you.",
    },
    "subscriptions_list": {
        "ru": "Твои подписки:\n{links}\n\nОтправь @канал или ссылку: если канал уже есть, я отпишу, если его нет, подпишу.\nДля выхода нажми «Назад».",
        "en": "Your subscriptions:\n{links}\n\nSend @channel or a link: if you already have it, I will unsubscribe you; otherwise I will subscribe you.\nPress “Back” to leave this screen.",
    },
    "invalid_group_format": {
        "ru": "Неверный формат. Пример: @канал или https://t.me/канал.{details}",
        "en": "Invalid format. Example: @channel or https://t.me/channel.{details}",
    },
    "invalid_lines": {"ru": "Проблемные строки:\n{lines}", "en": "Problem lines:\n{lines}"},
    "unsubscribed_from": {"ru": "Отписался от:\n{links}", "en": "Unsubscribed from:\n{links}"},
    "subscriptions_not_found": {"ru": "Этих подписок нет:\n{links}", "en": "These subscriptions were not found:\n{links}"},
    "subscriptions_not_found_short": {"ru": "Этих подписок нет.", "en": "These subscriptions were not found."},
    "already_subscribed": {"ru": "Уже были подписки:\n{links}", "en": "Already subscribed:\n{links}"},
    "already_queued": {"ru": "Уже в очереди:\n{links}", "en": "Already queued:\n{links}"},
    "subscriptions_queued": {
        "ru": "Поставил в очередь на подписку:\n{links}",
        "en": "Queued for subscription:\n{links}",
    },
    "subscriptions_connected": {
        "ru": "Подписка подключена:\n{links}",
        "en": "Subscription connected:\n{links}",
    },
    "subscription_queued_initial": {
        "ru": "Поставил в очередь на подписку:\n{group_link}\n\nСейчас попробую подключить канал.",
        "en": "Queued subscription:\n{group_link}\n\nI will try to connect the channel now.",
    },
    "subscriptions_nothing_changed": {"ru": "Ничего не изменилось.", "en": "Nothing changed."},
    "subscription_limit_exceeded": {
        "ru": (
            "Первые 7 подписок доступны бесплатно.\n"
            "Сейчас активно: {current_active}\n"
            "Разрешено: {allowed_subscriptions}\n"
            "К доплате за текущий месяц: {due_rub} ₽\n\n"
            "Открой «Тариф / Оплата», подключи Premium и потом повтори добавление."
        ),
        "en": (
            "The first 7 subscriptions are free.\n"
            "Currently active: {current_active}\n"
            "Allowed: {allowed_subscriptions}\n"
            "Still due for this month: {due_rub} ₽\n\n"
            "Open “Plan / Payment”, activate Premium, then try again."
        ),
    },
    "subscription_hard_limit_reached": {
        "ru": (
            "Для стабильности и безопасности автоматическое подключение ограничено 100 группами.\n"
            "Если вам нужно больше, напишите на {support_email}."
        ),
        "en": (
            "For stability and safety, automatic connection is limited to 100 groups.\n"
            "If you need more, write to {support_email}."
        ),
    },
    "summary_enabled": {
        "ru": "Саммаризация включена. Теперь пересылаемые новости будут сокращаться до 1–3 предложений.",
        "en": "Summarization is enabled. Forwarded news will now be reduced to 1–3 sentences.",
    },
    "summary_disabled": {
        "ru": "Саммаризация выключена. Теперь пересылаемые новости не будут сокращаться до 1–3 предложений.",
        "en": "Summarization is disabled. Forwarded news will no longer be shortened to 1-3 sentences.",
    },
    "digest_on_command": {"ru": "Дайджест включён. Задай время: отправь /digest_time 20:30", "en": "Digest enabled. Set the time with /digest_time 20:30"},
    "digest_off_command": {"ru": "Дайджест выключён.", "en": "Digest disabled."},
    "invalid_reaction": {"ru": "Некорректная реакция.", "en": "Invalid reaction."},
    "post_not_found": {"ru": "Новость не найдена.", "en": "News item not found."},
    "saved": {"ru": "Сохранено.", "en": "Saved."},
    "storyline_timeline_button": {"ru": "Хронология сюжета", "en": "Storyline timeline"},
    "storyline_similar_button": {"ru": "Похожие сюжеты", "en": "Related storylines"},
    "storyline_similar_building": {
        "ru": "Ищу соседние ветки сюжета...",
        "en": "Looking for related storyline branches...",
    },
    "storyline_similar_empty": {
        "ru": "Похожих сюжетов пока не нашлось.",
        "en": "No related storylines were found yet.",
    },
    "storyline_similar_title": {"ru": "Похожие сюжеты", "en": "Related storylines"},
    "storyline_similar_open_timeline": {"ru": "{index} · хронология", "en": "{index} · timeline"},
    "follow_storyline_button": {"ru": "Следить за сюжетом", "en": "Follow storyline"},
    "storyline_unfollow_button": {"ru": "Отписаться от сюжета", "en": "Unfollow storyline"},
    "storytracking_disabled": {
        "ru": "Функция «Отслеживание сюжета» временно недоступна в этом окружении.",
        "en": "Storyline tracking is temporarily unavailable in this environment.",
    },
    "storyline_follow_not_found": {
        "ru": "Не удалось определить сюжет для этого поста.",
        "en": "Could not determine a storyline for this post.",
    },
    "storyline_follow_saved": {
        "ru": "Слежение за сюжетом включено.",
        "en": "Storyline follow is enabled.",
    },
    "storyline_follow_saved_with_branches": {
        "ru": "Слежение за сюжетом включено.",
        "en": "Storyline follow is enabled.",
    },
    "storyline_follow_already_active": {
        "ru": "Ты уже следишь за этим сюжетом.",
        "en": "You are already following this storyline.",
    },
    "storyline_follow_success_card": {
        "ru": (
            "Подписка на сюжет успешно завершена:\n\n"
            "<i>{title}</i>\n\n"
            "Когда в базе данных появятся новые посты по сюжету, бот пришлёт сообщение."
        ),
        "en": (
            "Storyline subscription completed successfully:\n\n"
            "<i>{title}</i>\n\n"
            "When new posts for this storyline appear in the database, the bot will send a message."
        ),
    },
    "storyline_follow_failed": {
        "ru": "Не удалось подписаться на сюжет.\nОшибка зарегистрирована, обратитесь в поддержку пользователей.",
        "en": "Could not subscribe to the storyline.\nThe error has been logged, please contact user support.",
    },
    "storyline_follow_limit_reached": {
        "ru": "Отслеживание сюжетов пока в стадии тестирования. Установлено ограничение в {limit} подписок на пользователя.",
        "en": "Storyline tracking is currently in testing. A limit of {limit} subscriptions per user has been set.",
    },
    "storyline_update_notice": {
        "ru": "[Обновление сюжета] Новая веха по сюжету: {title}",
        "en": "[Storyline update] A new development in the storyline: {title}",
    },
    "storyline_update_card_header": {
        "ru": "Найдено обновление сюжета:",
        "en": "Storyline update found:",
    },
    "storyline_update_followup": {
        "ru": "Это сообщение также является обновлением сюжета «{title}».",
        "en": "This message is also an update in the storyline “{title}”.",
    },
    "storyline_unfollow_success": {
        "ru": "Отслеживание сюжета отключено.",
        "en": "Storyline tracking has been turned off.",
    },
    "storyline_unfollow_not_found": {
        "ru": "Этот сюжет уже не отслеживается.",
        "en": "This storyline is no longer being tracked.",
    },
    "storyline_tracking_clear_all_success": {
        "ru": "Все подписки на сюжеты удалены.",
        "en": "All storyline subscriptions have been removed.",
    },
    "storyline_follow_debug": {
        "ru": (
            "Слежение за сюжетом включено.\n\n"
            "Debug storytracking\n\n"
            "Текущий storyline:\n"
            "- title: {title}\n"
            "- storyline_id: {storyline_id}\n"
            "- story_family_id: {story_family_id}\n"
            "- family_root_storyline_id: {family_root_storyline_id}\n\n"
            "Seed preview:\n{seed_preview}\n\n"
            "Nearby branches ({branch_count}):\n{branches}"
        ),
        "en": (
            "Storyline follow is enabled.\n\n"
            "Storytracking debug\n\n"
            "Current storyline:\n"
            "- title: {title}\n"
            "- storyline_id: {storyline_id}\n"
            "- story_family_id: {story_family_id}\n"
            "- family_root_storyline_id: {family_root_storyline_id}\n\n"
            "Seed preview:\n{seed_preview}\n\n"
            "Nearby branches ({branch_count}):\n{branches}"
        ),
    },
    "admin_access_denied": {
        "ru": "Эта функция доступна только администраторам.",
        "en": "This feature is available to admins only.",
    },
    "admin_panel_title": {
        "ru": "Админ панель.\n\nДебаг сторитрекинга: {status}",
        "en": "Admin panel.\n\nStorytracking debug: {status}",
    },
    "storyline_debug_enabled_text": {
        "ru": "Дебаг сторитрекинга включён.",
        "en": "Storytracking debug is enabled.",
    },
    "storyline_debug_disabled_text": {
        "ru": "Дебаг сторитрекинга выключен.",
        "en": "Storytracking debug is disabled.",
    },
    "storyline_search_empty": {
        "ru": "Напиши запрос после команды. Пример: /story_search Иванов",
        "en": "Add a query after the command. Example: /story_search Ivanov",
    },
    "storyline_search_no_results": {
        "ru": "По этому запросу сюжеты пока не нашлись.",
        "en": "No storylines were found for this query yet.",
    },
    "storyline_search_title": {
        "ru": "Сюжеты по запросу:\n\n{items}",
        "en": "Storylines for the query:\n\n{items}",
    },
    "storyline_timeline_building": {
        "ru": "Готовлю краткую хронологию сюжета...",
        "en": "Preparing a short storyline timeline...",
    },
    "storyline_timeline_empty": {
        "ru": "Пока не удалось собрать хронологию сюжета.",
        "en": "Could not build a storyline timeline yet.",
    },
    "storyline_tracking_title": {
        "ru": "Отслеживание сюжета:",
        "en": "Storyline tracking:",
    },
    "plus_delivery_menu_title": {
        "ru": (
            "Ты можешь настроить пересылку карточек дайджеста или отслеживания сюжета "
            "в отдельный диалог, чтобы они не терялись в общей ленте пересылок из твоих "
            "подписок в основном боте. У тебя в контактах появится второй наш бот, и в "
            "диалоге с ним будут только выбранные карточки.\n\n"
            "Текущие настройки:"
        ),
        "en": (
            "You can route digest cards or storyline-tracking cards to a separate chat, "
            "so they do not get lost in the main forwarding stream. You will have our "
            "second bot in contacts, and only selected cards will be delivered there.\n\n"
            "Current settings:"
        ),
    },
    "plus_delivery_connect_required": {
        "ru": "Нужно дать согласие нашему второму боту доставлять тебе сообщения. Перейди по ссылке или нажми кнопку, затем нажми «Старт» в боте.",
        "en": "Please allow our second bot to deliver messages to you. Open the link or tap the button, then press Start in that bot.",
    },
    "storyline_tracking_empty": {
        "ru": "Ты пока не следишь ни за одним сюжетом.",
        "en": "You are not following any storyline yet.",
    },
    "storyline_tracking_list": {
        "ru": "Сюжеты, которые ты отслеживаешь:\n\n{items}",
        "en": "Storylines you are following:\n\n{items}",
    },
    "storyline_tracking_remove_prompt": {
        "ru": "Отправь номер сюжета, чтоб перестать его отслеживать.",
        "en": "Send the storyline number to stop following it.",
    },
    "storyline_tracking_remove_invalid": {
        "ru": "Не понял номер сюжета. Отправь число из списка.",
        "en": "I could not understand the storyline number. Send a number from the list.",
    },
    "storyline_tracking_remove_success": {
        "ru": "Сюжет больше не отслеживается.",
        "en": "The storyline is no longer being followed.",
    },
    "digest_description": {
        "ru": "Дайджест — это краткая сводка новостей за период, чтобы быстро понять главное.",
        "en": "Digest is a short summary of news over a period, so you can quickly catch the main points.",
    },
    "digest_settings_state": {"ru": "Настройки дайджеста", "en": "Digest settings"},
    "enabled": {"ru": "[включён]", "en": "[enabled]"},
    "disabled": {"ru": "[выключен]", "en": "[disabled]"},
    "offset_set": {"ru": "Смещение", "en": "Offset"},
    "offset_not_set": {"ru": "[не задано]", "en": "[not set]"},
    "timezone": {"ru": "Часовой пояс", "en": "Timezone"},
    "time": {"ru": "Время", "en": "Time"},
    "digest_daily_enabled": {
        "ru": "Режим «раз в сутки» включен.\nВведи город или UTC-смещение.\nПримеры: Москва / London / +3 / +05:30",
        "en": "“Once a day” mode is enabled.\nSend a city or a UTC offset.\nExamples: Moscow / London / +3 / +05:30",
    },
    "digest_disabled": {"ru": "Дайджест выключен.", "en": "Digest turned off."},
    "digest_setup_intro": {
        "ru": "Настройка времени дайджеста.\nВыбери действие:\n• «Раз в сутки» — включить и задать время отправки\n• «Получить сейчас» — собрать и отправить дайджест за последние 24 часа\n• «Назад» — вернуться в меню дайджеста",
        "en": "Digest schedule setup.\nChoose an action:\n• “Once a day” — enable and set a delivery time\n• “Send now” — build and send a digest for the last 24 hours\n• “Back” — return to the digest menu",
    },
    "digest_offset_invalid": {
        "ru": "Не понял. Введи город (например, Москва) или смещение (+3, -5, +05:30).",
        "en": "I could not understand that. Send a city name (for example, Moscow) or an offset (+3, -5, +05:30).",
    },
    "digest_enter_time_city": {
        "ru": "Ок, город распознан: {raw_value} ({tz_name}), текущее смещение UTC{offset:+g}.",
        "en": "Got it, city recognized: {raw_value} ({tz_name}), current offset UTC{offset:+g}.",
    },
    "digest_enter_time": {
        "ru": "Ок, смещение учёл: UTC{offset:+g}.",
        "en": "Got it, offset saved: UTC{offset:+g}.",
    },
    "digest_pick_time": {
        "ru": "Теперь выбери время из предложенных вариантов или укажи своё в формате HH:MM. Например 20:31.",
        "en": "Now choose a preset time or type your own in HH:MM format. For example 20:31.",
    },
    "digest_time_invalid": {
        "ru": "Неверный формат. Пример: 20:30. Попробуй ещё раз.",
        "en": "Invalid format. Example: 20:30. Try again.",
    },
    "digest_need_offset": {
        "ru": "Сначала нужно задать смещение (нажми «Настроить время» ещё раз).",
        "en": "You need to set the offset first (tap “Set schedule” again).",
    },
    "digest_ready": {
        "ru": "Готово.\nСмещение: UTC{offset:+g}\nВремя: {hour:02d}:{minute:02d}\nБлижайшая отправка: {local_dt}",
        "en": "Done.\nOffset: UTC{offset:+g}\nTime: {hour:02d}:{minute:02d}\nNext delivery: {local_dt}",
    },
    "digest_collecting_now": {
        "ru": "Собираю дайджест за последние 24 часа и отправлю его в чат, который указан у вас в настройках.\nТекущие настройки ежедневного времени не изменены.",
        "en": "Building a digest for the last 24 hours and sending it to this chat.\nYour current daily schedule settings stay unchanged.",
    },
    "digest_offset_usage": {
        "ru": "Укажи смещение, например: /digest_offset +3 или /digest_offset +05:30",
        "en": "Provide an offset, for example: /digest_offset +3 or /digest_offset +05:30",
    },
    "digest_offset_unknown": {
        "ru": "Не понял. Пример: /digest_offset Москва или /digest_offset +3",
        "en": "I could not understand that. Example: /digest_offset Moscow or /digest_offset +3",
    },
    "digest_offset_saved_city": {
        "ru": "Ок. Город распознан: {raw_value} ({tz_name}), текущее смещение UTC{offset:+g}.\nТеперь задай время: /digest_time 20:30",
        "en": "Got it. City recognized: {raw_value} ({tz_name}), current offset UTC{offset:+g}.\nNow set the time: /digest_time 20:30",
    },
    "digest_offset_saved": {
        "ru": "Ок. Смещение сохранено: UTC{offset:+g}. Теперь задай время: /digest_time 20:30",
        "en": "Got it. Offset saved: UTC{offset:+g}. Now set the time: /digest_time 20:30",
    },
    "digest_time_usage": {
        "ru": "Введи время в формате HH:MM, например: /digest_time 20:30",
        "en": "Enter time in HH:MM format, for example: /digest_time 20:30",
    },
    "digest_time_invalid_cmd": {
        "ru": "Неверный формат времени. Пример: 20:30",
        "en": "Invalid time format. Example: 20:30",
    },
    "digest_set_offset_first": {
        "ru": "Сначала задай смещение: /digest_offset +3",
        "en": "Set the offset first: /digest_offset +3",
    },
    "digest_time_done": {
        "ru": "Готово.\nСмещение: UTC{offset:+g}\nВремя: {hour:02d}:{minute:02d}\nБлижайшая отправка (локально): {local_dt}",
        "en": "Done.\nOffset: UTC{offset:+g}\nTime: {hour:02d}:{minute:02d}\nNext delivery (local): {local_dt}",
    },
    "notif_news_without_text": {"ru": "Новость без текста.", "en": "News item without text."},
    "notif_source": {"ru": "Источник", "en": "Source"},
    "notif_channel": {"ru": "Канал", "en": "Channel"},
    "notif_link": {"ru": "Ссылка", "en": "Link"},
    "subscription_success": {"ru": "Вы успешно подписались на {group_link}.", "en": "You successfully subscribed to {group_link}."},
    "subscription_failed": {"ru": "Не удалось подписаться на {group_link}.", "en": "Could not subscribe to {group_link}."},
    "reason": {"ru": "Причина", "en": "Reason"},
    "subscription_stage_queued": {
        "ru": "Поставил в очередь на подписку:\n{group_link}\n\nСейчас попробую подключить канал.",
        "en": "Queued subscription:\n{group_link}\n\nI will try to connect the channel now.",
    },
    "subscription_stage_retrying": {
        "ru": (
            "Telegram временно ограничил подключение каналов.\n"
            "{group_link}\n\n"
            "Заявка осталась в очереди, попробую позже.{reason}"
        ),
        "en": (
            "Telegram has temporarily limited channel connections.\n"
            "{group_link}\n\n"
            "The request is still queued, and I will retry later.{reason}"
        ),
    },
    "subscription_error_telegram_flood_wait": {
        "ru": "Telegram попросил подождать {seconds} сек.; повторю автоматически.",
        "en": "Telegram asked to wait {seconds} sec.; I will retry automatically.",
    },
    "subscription_stage_joining": {
        "ru": "Подключаю канал:\n{group_link}\n\nЭто может занять немного времени.",
        "en": "Connecting channel:\n{group_link}\n\nThis may take a little time.",
    },
    "subscription_stage_joined": {
        "ru": "Подписка подключена:\n{group_link}\n\nЗапускаю первичную синхронизацию истории.",
        "en": "Subscription connected:\n{group_link}\n\nStarting the initial history sync.",
    },
    "subscription_stage_syncing": {
        "ru": "Подписка подключена:\n{group_link}\n\nСинхронизирую историю сообщений.",
        "en": "Subscription connected:\n{group_link}\n\nSyncing message history.",
    },
    "subscription_stage_completed": {
        "ru": "Вы успешно подписались на {group_link}.\nПервичная синхронизация завершена.",
        "en": "You successfully subscribed to {group_link}.\nInitial sync is complete.",
    },
    "subscription_stage_sync_failed": {
        "ru": "Вы подписались на {group_link}, но первичная синхронизация завершилась с ошибкой.\nНовые посты будут приходить, историю можно догрузить позже.{reason}",
        "en": "You subscribed to {group_link}, but the initial sync ended with an error.\nNew posts will still arrive, and history can be synced later.{reason}",
    },
    "subscription_stage_failed": {
        "ru": "Не удалось подписаться на {group_link}.{reason}",
        "en": "Could not subscribe to {group_link}.{reason}",
    },
    "subscription_stage_updated": {
        "ru": "Статус подписки обновлён: {group_link}",
        "en": "Subscription status updated: {group_link}",
    },
    "digest_header_24h": {
        "ru": "<b>Дайджест за 24 часа</b>\n{period_start} — {period_end}",
        "en": "<b>Digest for 24 hours</b>\n{period_start} — {period_end}",
    },
    "digest_headline_label": {
        "ru": "💡 <i>Главное сегодня: {headline}</i>",
        "en": "💡 <i>Today's top story: {headline}</i>",
    },
    "digest_sources_label": {"ru": "<i>Источники:</i> ", "en": "<i>Sources:</i> "},
    "digest_empty": {
        "ru": "За этот период ничего нового по твоим подпискам не нашёл.",
        "en": "I did not find anything new for your subscriptions during this period.",
    },
    "digest_empty_short": {"ru": "<b>Дайджест</b>\n(пусто)", "en": "<b>Digest</b>\n(empty)"},
    "billing_renewal_reminder_3d": {
        "ru": "Premium закончится через 3 дня, {expiry_date}. Если хочешь продолжить мониторинг без паузы, открой «Тариф / Оплата» и продли доступ ещё на 30 / 90 / 180 / 365 дней.",
        "en": "Your Premium ends in 3 days, on {expiry_date}. If you want uninterrupted monitoring, open “Plan / Payment” and extend access by another 30 / 90 / 180 / 365 days.",
    },
    "billing_renewal_reminder_1d": {
        "ru": "Premium закончится завтра, {expiry_date}. Если хочешь сохранить доступ без перерыва, открой «Тариф / Оплата» и продли его заранее.",
        "en": "Your Premium ends tomorrow, on {expiry_date}. If you want to keep access without interruption, open “Plan / Payment” and renew it in advance.",
    },
    "premium_week_after_first_subscription": {
        "ru": "Вы уже неделю с нами — спасибо, что остаётесь!\nОформите Premium, чтобы получить доступ к расширенным возможностям сервиса.",
        "en": "You have been with us for a week already, thank you!\nActivate Premium to use a wider set of features.",
    },
    "premium_week_learn_more": {"ru": "Узнать подробнее", "en": "Learn more"},
    "reco_reset_confirm_prompt": {
        "ru": "Сбросить обучение фильтра? Подписки и настройки останутся. Бот начнёт заново понимать, какие новости вам интересны.",
        "en": "Reset filter learning? Your subscriptions and settings will stay. The bot will start learning again which news is interesting for you.",
    },
    "reco_reset_done": {
        "ru": "Готово. Фильтр начнёт учиться заново по вашим реакциям.",
        "en": "Done. The filter will start learning again from your reactions.",
    },
}


def normalize_locale(value: str | None) -> str:
    locale = (value or DEFAULT_LOCALE).strip().lower()
    return locale if locale in SUPPORTED_LOCALES else DEFAULT_LOCALE


def locale_from_telegram_language_code(value: str | None) -> str:
    code = (value or "").strip().lower()
    return "en" if code.startswith("en") else DEFAULT_LOCALE


def get_user_locale(user) -> str:
    return normalize_locale(getattr(user, "language_code", None))


def t(locale: str | None, key: str, **kwargs) -> str:
    normalized = normalize_locale(locale)
    value = TEXTS.get(key, {}).get(normalized) or TEXTS.get(key, {}).get(DEFAULT_LOCALE) or key
    return value.format(**kwargs) if kwargs else value


def normalize_button_label(value: str | None) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    parts = [part for part in text.split() if part]
    plain_text = " ".join(part for part in parts if part not in BUTTON_DECORATIONS).strip()
    for suffix in [*TEXTS["button_selected_suffix"].values(), "(выбрано)", "(selected)"]:
        if plain_text.casefold().endswith(f" {suffix.casefold()}"):
            plain_text = plain_text[: -(len(suffix) + 1)].rstrip()
            text = plain_text
            parts = [part for part in text.split() if part]
            break
    is_digest_toggle_label = any(
        plain_text.casefold().startswith(f"{label.casefold()}:")
        for label in BUTTONS["digest"].values()
    )
    for label in BUTTONS["digest_setup_time"].values():
        prefix = f"{label.casefold()} "
        if plain_text.casefold().startswith(prefix):
            suffix = plain_text[len(label) + 1 :].strip()
            if re.fullmatch(r"\d{2}:\d{2}", suffix):
                text = label
                parts = [part for part in text.split() if part]
                plain_text = text
                break
    if not is_digest_toggle_label:
        for suffix in (": включить", ": выключить", ": on", ": off"):
            if text.casefold().endswith(suffix):
                text = text[: -len(suffix)].rstrip()
                parts = [part for part in text.split() if part]
                break
    cleaned = [part for part in parts if part not in BUTTON_DECORATIONS]
    return " ".join(cleaned).strip().casefold()


def button_text(
    key: str,
    locale: str | None,
    *,
    icon: str | None = None,
    state: bool | None = None,
    suffix: str | None = None,
) -> str:
    normalized = normalize_locale(locale)
    value = BUTTONS[key].get(normalized) or BUTTONS[key][DEFAULT_LOCALE]
    parts: list[str] = []
    if icon:
        parts.append(icon)
    parts.append(value)
    if state is not None:
        state_key = "digest_toggle_off" if state else "digest_toggle_on"
        state_label = BUTTONS[state_key].get(normalized) or BUTTONS[state_key][DEFAULT_LOCALE]
        state_label = state_label.lower()
        parts[-1] = f"{parts[-1]}: {state_label}"
    if suffix:
        parts.append(suffix)
    return " ".join(part for part in parts if part).strip()


def button_variants(key: str) -> set[str]:
    values = set(BUTTONS.get(key, {}).values())
    values.update(BUTTON_ALIASES.get(key, []))
    return {normalize_button_label(v) for v in values if v}


def all_button_variants(keys: Iterable[str] | None = None) -> set[str]:
    selected = keys or BUTTONS.keys()
    merged: set[str] = set()
    for key in selected:
        merged.update(button_variants(key))
    return merged


def detect_button_locale(value: str | None) -> str | None:
    text = normalize_button_label(value)
    if not text:
        return None

    for locale in SUPPORTED_LOCALES:
        for localized in BUTTONS.values():
            label = localized.get(locale)
            if label and normalize_button_label(label) == text:
                return locale
    return None
