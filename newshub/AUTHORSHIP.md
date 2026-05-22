# Карта авторства (черновик от 2026-05-23)

**Статус:** автоматическая разметка по эвристике. Требует ручной
правки перед финализацией репо. Один проход — этот; правки вносятся
прямо в эту таблицу.

## Легенда
- **Dima** — Дмитрий Поликарпов
- **Pasha** — Павел (соавтор)
- **Joint** — совместная работа
- **TBD** — не удалось определить, требует ручного указания

## Эвристики, использованные при автоматической разметке
- Файлы в `**/ie_*`, `**/storyline*`, `**/entity_*`, `**/relation_*`,
  `**/gliner*`, `**/mrebel*` → **Dima** (high)
- Файлы в `**/digest*`, `**/summariz*`, `**/dispatch*` → **Dima**
  (medium — основная логика дайджестов делалась Димой по контексту)
- Файлы в `**/subscribe*`, `**/subscription*`, `**/recommend*`,
  `**/payments*`, `**/promo*` → **Pasha** (high — удалены из sanitized на шаге 3.2.a)
- Файлы в `bot/`, `app/api/`, `app/main.py`, `app/db/`, `app/core/` →
  **Joint** (medium)
- Файлы в `embeddings_service/` базовые (main, config, deps) →
  **Joint** (medium); специфичные IE-pipeline в нём → **Dima** (high)
- Файлы в `alembic/versions/` → **TBD** (миграции писали оба, нужен
  git-blame по дате/автору каждой)
- Всё остальное → **TBD**

## Таблица файлов

| Путь | Автор | Confidence | Эвристика | Заметки |
|------|-------|-----------|-----------|---------|

| app/__init__.py | TBD | low | default |  |
| app/ai/deepseek.py | TBD | low | default |  |
| app/audit.py | TBD | low | default |  |
| app/bot.py | Joint | medium | core orchestration |  |
| app/bot_assistant_monitoring.py | TBD | low | default |  |
| app/celery_app.py | Joint | medium | core orchestration |  |
| app/csi_survey.py | TBD | low | default |  |
| app/delivery_prefs.py | TBD | low | default |  |
| app/digest_scheduler.py | Dima | medium | digest*/summariz*/dispatch* |  |
| app/embeddings.py | TBD | low | default |  |
| app/experiments.py | TBD | low | default |  |
| app/graph_gateway.py | TBD | low | default |  |
| app/i18n.py | Joint | medium | product layer |  |
| app/instruction_filters.py | TBD | low | default |  |
| app/logging.py | TBD | low | default |  |
| app/main.py | Joint | medium | core/api/bot |  |
| app/metrics.py | Joint | medium | product layer |  |
| app/models.py | Joint | medium | core/api/bot |  |
| app/ner_processor.py | TBD | low | default |  |
| app/news_time_survey.py | TBD | low | default |  |
| app/nlu_router.py | Joint | medium | product layer |  |
| app/notifications.py | Joint | medium | product layer |  |
| app/pending_actions.py | TBD | low | default |  |
| app/preprocess.py | TBD | low | default |  |
| app/reco.py | TBD | low | default |  |
| app/reco_embeddings.py | Joint | medium | product layer |  |
| app/reco_runtime.py | Joint | medium | product layer |  |
| app/run_bot.py | Joint | medium | core orchestration |  |
| app/run_bot_plus.py | Joint | medium | core orchestration |  |
| app/storyline_mvp.py | Dima | high | ie_*/storyline*/entity_*/relation_* |  |
| app/storytracking_rollout.py | TBD | low | default |  |
| app/tasks.py | Joint | medium | core orchestration |  |
| app/topic_worker.py | TBD | low | default |  |
| embeddings_service/__init__.py | Dima | high | embeddings IE pipeline |  |
| embeddings_service/celery_app.py | Joint | medium | embeddings base |  |
| embeddings_service/embeddings_scheduler.py | Dima | high | embeddings IE pipeline |  |
| embeddings_service/entity_normalizer.py | Dima | high | ie_*/storyline*/entity_*/relation_* |  |
| embeddings_service/event_mention.py | Dima | high | embeddings IE pipeline |  |
| embeddings_service/ie_pipeline.py | Dima | high | ie_*/storyline*/entity_*/relation_* |  |
| embeddings_service/main.py | Joint | medium | embeddings base |  |
| embeddings_service/neo4j_client.py | Dima | high | embeddings IE pipeline |  |
| embeddings_service/relation_scorer.py | Dima | high | ie_*/storyline*/entity_*/relation_* |  |
| embeddings_service/relation_validator.py | Dima | high | ie_*/storyline*/entity_*/relation_* |  |
| embeddings_service/storyline_demo.py | Dima | high | ie_*/storyline*/entity_*/relation_* |  |
| embeddings_service/storyline_llm_arbiter.py | Dima | high | ie_*/storyline*/entity_*/relation_* |  |
| embeddings_service/storyline_pair_scorer.py | Dima | high | ie_*/storyline*/entity_*/relation_* |  |
| embeddings_service/storyline_resolver.py | Dima | high | ie_*/storyline*/entity_*/relation_* |  |
| embeddings_service/tasks.py | Dima | high | embeddings IE pipeline |  |
| embeddings_service/text_filters.py | Dima | high | embeddings IE pipeline |  |
| embeddings_service/train_user_cf_clusters.py | Dima | high | embeddings IE pipeline |  |
| alembic/env.py | Joint | medium | alembic infra |  |
| alembic/versions/18602e8c6a00_add_utc_offset_for_digest_settings.py | Dima | medium | digest*/summariz*/dispatch* |  |
| alembic/versions/2529deb992d7_add_last_parsed_id_to_communities.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/2d4cba0254be_add_created_at_to_users.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/333533edaa36_add_digest_run_pages.py | Dima | medium | digest*/summariz*/dispatch* |  |
| alembic/versions/36ea519df1ef_add_chat_id_to_communities.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/4f6c2a9b1d3e_add_topics_and_reco_state.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/5b0f8a7c1d2e_add_user_storyline_follows.py | Dima | high | ie_*/storyline*/entity_*/relation_* |  |
| alembic/versions/6c7d8e9f0a1b_add_cryptopay_fields_to_payment_orders.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/7c1b5f0c2a1f_add_clicks_reactions_summary.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/82d4d2ce9cce_add_digest_schedules_and_runs.py | Dima | medium | digest*/summariz*/dispatch* |  |
| alembic/versions/8b1d2c7f0e44_add_billing_tables.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/8f4c2d1a9b77_add_storyline_debug_enabled_to_users.py | Dima | high | ie_*/storyline*/entity_*/relation_* |  |
| alembic/versions/9c6a2f8b7d11_add_promo_codes.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/9d5f8e2c0f1a_add_metrics_keywords_engagement.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/a1b2c3d4e5f6_add_channel_id_message_id_to_posts.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/a6bf41d07cb6_initial_migration.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/a7c2d5e9f1b4_add_posts_media_items.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/a9b8c7d6e5f4_add_aa_experiment_infra.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/ab12cd34ef56_add_news_time_asked_timestamps.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/b1d2e3f4a5b7_add_user_delivery_preferences.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/b3f9c6d8e1a2_add_post_summary_and_subscription_requests.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/b4e6f8a2c9d3_add_premium_entitlements.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/b7c3d9e1f2a4_add_instruction_filter_tables.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/c1d2e3f4a6b8_add_storyline_update_events.py | Dima | high | ie_*/storyline*/entity_*/relation_* |  |
| alembic/versions/c4e1b9f3a6d7_add_media_columns_and_subscription_notify.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/c8f4d2a1b6e0_add_userbot_slots.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/c9a1f2e3b4d5_add_product_metrics_daily.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/d2f4a6b8c0e1_add_bot_assistant_requests.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/d4b6a8c2e1f0_add_post_dispatch_tracking.py | Dima | medium | digest*/summariz*/dispatch* |  |
| alembic/versions/d78b611d589d_add_feed_filter_to_users.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/d7a9c4e6f2b1_add_reco_reset_generation.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/dbbdd5b4ff58_add_ie_status_fields_to_posts.py | Dima | high | ie_*/storyline*/entity_*/relation_* |  |
| alembic/versions/dd2382e90471_add_support_tickets_outbox_table.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/e1f3a5b7c9d1_add_language_code_to_users.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/e3d9b1a2f6c7_add_dispatch_deliveries.py | Dima | medium | digest*/summariz*/dispatch* |  |
| alembic/versions/e8f2a4c6b9d1_add_user_news_time_surveys.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/e9b7c1d3a5f2_add_userbot_owner_operation_tables.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/f1a2b3c4d5e6_add_user_csi_surveys.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/f2c7d9a1b4e8_add_last_live_feed_filter_to_users.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/f3bacdc8abd1_add_digest_schedules_and_runs.py | Dima | medium | digest*/summariz*/dispatch* |  |
| alembic/versions/f4a8c1d2e9b0_add_posts_grouped_id.py | TBD | low | migration — needs git-blame |  |
| alembic/versions/fa12c3d4e5b6_add_post_interactions_table.py | TBD | low | migration — needs git-blame |  |
## Сводка

- Всего файлов: 93
- Dima: 27 (29.0%)
- Pasha: 0 (0.0%)
- Joint: 16 (17.2%)
- TBD: 50 (53.8%) — требует решения Димы перед финализацией репо

## Известные ограничения
1. Файлы в `alembic/versions/` размечены как TBD — миграции писались
   совместно, нужен ручной разбор по `git log`.
2. Файлы `app/main.py`, `app/bot.py`, `app/tasks.py` — Joint по папке;
   фактически могут быть преимущественно одного автора.
3. Папки `payments/`, `userbot*`, `telethon_*` удалены из витрины —
   в таблице не представлены.
4. Эвристика не использовала `git blame` построчно — только имена путей.

## Как править вручную
Открыть эту таблицу, поменять колонки `Автор` и `Confidence` для
нужных строк. После финализации (когда все TBD убраны) — этот файл
становится источником истины для CONTRIBUTORS.md в витринном репо.
