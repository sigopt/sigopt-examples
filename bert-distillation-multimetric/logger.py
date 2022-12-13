import logging
import os

logger = logging.getLogger(__name__)

os.environ["AWS_DEFAULT_REGION"] = "us-west-2"

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
logger.setLevel(logging.INFO)