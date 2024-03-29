import sys
import os
os.environ["DGLBACKEND"] = "pytorch"
from argparse import ArgumentParser



from utils import Settings, get_callable_by_name
import torch

def run_training_job(args):
    if args.smp is None:
        setting_module_path = os.path.dirname(__file__) + '/exp_settings/st_dram_ref.py'
    else:
        setting_module_path = args.smp
    settings = Settings(setting_module_path)
    settings.OPTIMIZER['lr'] = args.lr

    settings.RELOAD_CHECKPOINT_PATH = args.ckp_path
    settings.RELOAD_CHECKPOINT = True if args.pretrain > 0 else False
    settings.TRAIN_BATCH_SIZE = args.batch_size
    runner_cls = get_callable_by_name(settings.JOB_RUNNER_CLS)
    ct = runner_cls(settings_module=settings)
    ct.run()


if __name__ == "__main__":
    print("Docker start running training job.")
    parser = ArgumentParser()
    parser.add_argument('pretrain', type=int, nargs='?',
                        default=0,
                        help="if use pretrained model.")
    parser.add_argument('lr', type=float, nargs='?',
                        default=0.001,
                        help="set up learning rate.")
    parser.add_argument('--batch_size', type=int, nargs='?',
                        default=1,
                        help="set up scan path.")
    parser.add_argument('--smp', type=str, nargs='?',
                        default=None,
                        help="set up scan path.")
    parser.add_argument('--ckp_path', type=str, default=None,
                        help='set checkpoint path.')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    args = parser.parse_args()
    run_training_job(args)
