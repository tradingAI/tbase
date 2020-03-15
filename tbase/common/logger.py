import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s[%(lineno)d] %(levelname)s %(message)s')

logger = logging.getLogger()

dir_name = os.path.join("/tmp", "tbase")
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

handler = logging.FileHandler(os.path.join(dir_name, "tbase.log"))
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s %(filename)s[%(lineno)d] %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
