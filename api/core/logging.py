import logging


LOG_FOMRAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FOMRAT
    )


def get_logger(name):
    return logging.getLogger(name)
