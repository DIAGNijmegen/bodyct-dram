import sys
import os

import matplotlib
matplotlib.use('Agg')

from utils import Settings
from job_runner import LesionSegTest

def run_testing_job():
    setting_module_path = os.path.dirname(__file__) + '/exp_settings/st_dram_ref_att.py'
    settings = Settings(setting_module_path)
    input_image_path = '/input/images/ct/'
    input_lobe_path = '/input/images/pulmonary-lobes/'

    output_path = '/output/images/'
    settings.MODEL_ROOT_PATH = output_path
    settings.DEBUG_PATH = output_path
    settings.RELOAD_CHECKPOINT_PATH = 'best.pth'
    # settings.LOGGING['handlers']['file_handler']['filename'] = '/output/process.log'
    settings.LOGGING['handlers']['file_handler']['filename'] = \
        r'D:\workspace\datasets\COPDGene\v5\derived\emphysema-heatmap/process.log'
    ct = LesionSegTest(input_image_path, input_lobe_path,
                       output_path,
                    settings, 'best.pth')
    ct.run()


if __name__ == "__main__":
    print("Docker start running testing job.")
    run_testing_job()