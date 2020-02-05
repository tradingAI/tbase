import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s[%(lineno)d] %(levelname)s %(message)s')

logger = logging.getLogger()

handler = logging.FileHandler("/tmp/tbase.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[%(lineno)d] %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
