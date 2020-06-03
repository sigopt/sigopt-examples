import logging
from logbeam import CloudWatchLogsHandler
import os

logger = logging.getLogger(__name__)

os.environ["AWS_DEFAULT_REGION"] = "us-west-2"

cw_handler = CloudWatchLogsHandler(
    log_group_name='meghana_log_group',
    log_stream_name='meghana_log_stream',
    buffer_duration=10000,
    batch_count=10,
    batch_size=1048576
)

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
logger.setLevel(logging.INFO)
logger.addHandler(cw_handler)
