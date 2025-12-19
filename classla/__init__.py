from classla.pipeline.core import Pipeline
from classla.models.common.doc import Document
from classla.resources.common import download
from classla.resources.installation import install_corenlp, download_corenlp_models
from classla._version import __version__, __resources_version__
from classla.utils.parallel import ParallelPipeline, process_parallel

import logging
logger = logging.getLogger('classla')

# if the client application hasn't set the log level, we set it
# ourselves to INFO
if logger.level == 0:
    logger.setLevel(logging.INFO)

log_handler = logging.StreamHandler()
log_formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s: %(message)s",
                              datefmt='%Y-%m-%d %H:%M:%S')
log_handler.setFormatter(log_formatter)

# also, if the client hasn't added any handlers for this logger
# (or a default handler), we add a handler of our own
#
# client can later do
#   logger.removeHandler(classla.log_handler)
if not logger.hasHandlers():
    logger.addHandler(log_handler)
