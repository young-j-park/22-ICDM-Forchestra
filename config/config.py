
import logging

logging.getLogger().setLevel(logging.INFO)

M5_DATASET_PATH = '/Users/youngjinpark/Dropbox (MIT)/forchestra/final/Forchestra_Sep28/data/m5'
NUM_WORKERS = 0
# M5_DATASET_PATH = '/home/gridsan/yjpark/tmp/Forchestra_Sep28/data/m5'
# NUM_WORKERS = 8
logging.info(f'Use {NUM_WORKERS} workers.')

# Dataset
TARGET_KEY = 'order_cnt'
MASK_KEY = 'mask'
DATE_KEY = 'date_list'

# EPS
NORM_EPS = {
    TARGET_KEY: 10,
}
