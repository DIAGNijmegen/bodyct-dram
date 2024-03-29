# Flags
COPY_DATA = False
ON_PREMISE_LOCATION = None

RELOAD_CHECKPOINT = False
IS_CUDA = True

RELOAD_CHECKPOINT_PATH = None

RELOAD_DICT_LIST = ["model"]

DB_PATH = "/mnt/netcache/bodyct/experiments/covid_lesion_wsss_t9195/"
TEST_CSV = "/mnt/netcache/bodyct/experiments/covid_lesion_wsss_t9195/test.csv"
TRAIN_CSV = "/mnt/netcache/bodyct/experiments/covid_lesion_wsss_t9195/wss_train.csv"
VALID_CSV = "/mnt/netcache/bodyct/experiments/covid_lesion_wsss_t9195/val.csv"
DEBUG_PATH = "/mnt/netcache/bodyct/experiments/covid_lesion_wsss_t9195/test_cases/"
MODEL_ROOT_PATH = "/mnt/netcache/bodyct/experiments/covid_lesion_wsss_t9195/models/"

JOB_RUNNER_CLS = "job_runner.LesionSegChunkTrain"
TEST_JOB_RUNNER_CLS = "job_runner.LesionSegTest"



EXP_NAME = "st_dram_ref"

# Training iterations and sizes.
RESAMPLE_MODE = "fixed_size"

NUM_EPOCHS = 200
VAL_EPOCHS = 10
STATE_EPOCHS = 10
NUM_WORKERS = 0
LOG_STEPS = 1

AUG_RATIO = 0.0
BALANCED_LABEL_COUNT = 200
TRAIN_BATCH_SIZE = 10

RESAMPLE_SPACING = 1.0
TEST_RESAMPLE_SPACING = 1.0
RESAMPLE_SIZE = (80, 80, 80)
LOSS_FACTORS = [2.0, 1.0, 0.5, 0.5]

RELABEL_MAPPING = {}
LABEL_NAME_MAPPING = {0: 'background',
                      1: 'emphysema'}
CLASS_WEIGHTS = [0.65, 0.7, 0.7, 0.75, 0.75, 0.8]
# thresholds
PAD_VALUE = -2048
WINDOWING_MAX = -300
WINDOWING_MIN = -1000
NR_CLASS = 1

MODEL = {
    "method": "models.DC3D",
    "n_layers": 3,
    "in_ch_list": [1, 64, 128, 256, 768, 384, 192],
    "base_ch_list": [32, 64, 128, 256, 256, 128, 64],
    "end_ch_list": [64, 128, 256, 512, 256, 128, 64],

    "kernel_sizes": [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
    'stacking': 3,
    "padding_list": [(1, 1), (1, 1), (1, 1), (1, 1),
                     (1, 1), (1, 1), (1, 1)],
    "checkpoint_layers": [0, 1, 0, 1, 0, 1, 0],

    "dropout": 0.0,
    "upsample_ksize": (3, 3, 3),
    "upsample_sf": (2, 2, 2),
    'out_ch': NR_CLASS,
}

TEST_MERGE_PROTOCOLS = [(None, None, None, None)]

INITIALIZER = {
    "method": "models.HeNorm",
    "mode": "fan_in"
}

# OPTIMIZER = {
#    "method" : "adabound.AdaBound",
#    "lr": 0.0001,
#    "final_lr": 0.01
# }
# OPTIMIZER = {
#    "method": "torch.optim.SGD",
#    "momentum": 0.9,
#    "lr": 0.0001,
# }

OPTIMIZER = {
   "method": "torch.optim.Adam",
   "lr": 0.0001,
}



#OPTIMIZER = {
#    "method": "engine.models.optimizer.RAdam",
#    "lr": 0.0001,
#}

SCHEDULER = {
    "method": "torch.optim.lr_scheduler.ExponentialLR",
    "gamma": 0.9
}


LOSS_FUNC = {
    "method": "metrics.IntRegRefineLoss",
    "band_width": 1e-2,
    "smoothing": 0.1
}



# loggers.
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout'
        },
        "file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": "{}/{}/info.log".format(MODEL_ROOT_PATH, EXP_NAME),
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        },
    },
    'loggers': {
        '': {
            'handlers': ['console', "file_handler"],
            'level': 'INFO',
            'propagate': True
        },
    }
}

PROCESSOR_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout'
        },
        "file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": "{}/{}/processor_info.log".format(MODEL_ROOT_PATH, EXP_NAME),
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        },
    },
    'loggers': {
        '': {
            'handlers': ['console', "file_handler"],
            'level': 'INFO',
            'propagate': True
        },
    }
}

# visualization
INSPECT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout'
        },
        "file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": "{}/{}/inspect_info.log".format(MODEL_ROOT_PATH, EXP_NAME),
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        },
    },
    'loggers': {
        '': {
            'handlers': ['console', "file_handler"],
            'level': 'INFO',
            'propagate': True
        },
    }
}

# visualization


VISUALIZATION_COLOR_TABLE = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (100, 0, 0),
    (100, 100, 0),
    (100, 100, 100),
    (50, 200, 0),
    (50, 200, 200),
    (50, 50, 200),
    (200, 50, 200),
    (50, 200, 50),
]

VISUALIZATION_ALPHA = 0.2
VISUALIZATION_SPARSENESS = 150
VISUALIZATION_PORT = 6012


INSPECT_PARAMETERS= {
    "watch_layers": {
        "unet1.bg": {"input": True, "stride": 1},
        "unet2.bg": {"input": False, "stride": 1},

    },
}

