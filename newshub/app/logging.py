import structlog


def configure_logging(service_name: str) -> None:
    def _add_stable_fields(_, __, event_dict):
        event_dict.setdefault("service", service_name)
        event_dict.setdefault("trace_id", None)
        event_dict.setdefault("user_id", None)
        return event_dict

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_stable_fields,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.BoundLogger,
        cache_logger_on_first_use=True,
    )
