import time

import logging.config
import logging
from torch.utils.data.dataloader import DataLoader
from utils import *
from tensorboardX import SummaryWriter
from data_sampler import LobeChunkCTSSSampler
from dataset import *
from data_transforms import *
import pandas as pd
from enum import Enum
import inspect, sys
from torchvision import transforms
import glob
import torch
import json
import traceback
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
import seaborn as sn
from sklearn.metrics import confusion_matrix, accuracy_score
from pandas import DataFrame


def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = []
    for c in df_cm.columns:
        sum_col.append(df_cm[c].sum())
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append(item_line[1].sum())
    df_cm['sum_lin'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['sum_col'] = sum_col


def get_new_fig(fn, figsize=[9, 9]):
    """ Init graphics """
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()  # Get Current Axis
    ax1.cla()  # clear existing plot
    return fig1, ax1


def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
    """
      config cell text and colors
      and return text elements to add and to dell
      @TODO: use fmt
    """
    text_add = [];
    text_del = [];
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:, col]
    ccl = len(curr_column)

    # last line  and/or last column
    if (col == (ccl - 1)) or (lin == (ccl - 1)):
        # tots and percents
        if (cell_val != 0):
            if (col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif (col == ccl - 1):
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif (lin == ccl - 1):
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ['%.2f%%' % (per_ok), '100%'][per_ok == 100]

        # text to DEL
        text_del.append(oText)

        # text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fz)
        text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
        lis_txt = ['%d' % (cell_val), per_ok_s, '%.2f%%' % (per_err)]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy();
        dic['color'] = 'g';
        lis_kwa.append(dic);
        dic = text_kwargs.copy();
        dic['color'] = 'r';
        lis_kwa.append(dic);
        lis_pos = [(oText._x, oText._y - 0.3), (oText._x, oText._y), (oText._x, oText._y + 0.3)]
        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
            # print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
            text_add.append(newText)
        # print '\n'

        # set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if (col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr

    else:
        if (per > 0):
            txt = '%s\n%.2f%%' % (cell_val, per)
        else:
            if (show_null_values == 0):
                txt = ''
            elif (show_null_values == 1):
                txt = '0'
            else:
                txt = '0\n0.0%'
        oText.set_text(txt)

        # main diagonal
        if (col == lin):
            # set color of the textin the diagonal to white
            oText.set_color('w')
            # set background color in the diagonal to blue
            facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        else:
            oText.set_color('r')

    return text_add, text_del


def pretty_plot_confusion_matrix(df_cm, annot=True, cmap="Oranges", fmt='.2f', fz=11,
                                 lw=0.5, cbar=False, figsize=[13, 13], show_null_values=0, pred_val_axis='y',
                                 save_path=None):
    """
      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        annot          print text in each cell
        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fz             fontsize
        lw             linewidth
        pred_val_axis  where to show the prediction values (x or y axis)
                        'col' or 'x': show predicted values in columns (x axis) instead lines
                        'lin' or 'y': show predicted values in lines   (y axis)
    """
    if (pred_val_axis in ('col', 'x')):
        xlbl = 'Predicted'
        ylbl = 'Actual'
    else:
        xlbl = 'Actual'
        ylbl = 'Predicted'
        df_cm = df_cm.T

    # create "Total" column
    insert_totals(df_cm)

    # this is for print allways in the same window
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    # thanks for seaborn
    ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
                    cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt, xticklabels=True, yticklabels=True)

    # set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=25, fontsize=10)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.set_visible(False)
        t.set_visible(False)
    for t in ax.yaxis.get_major_ticks():
        t.set_visible(False)
        t.set_visible(False)

    # face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    # iter in text elements
    array_df = np.array(df_cm.to_records(index=False).tolist())
    text_add = [];
    text_del = [];
    posi = -1  # from left to right, bottom to top.
    for t in ax.collections[0].axes.texts:  # ax.texts:
        pos = np.array(t.get_position()) - [0.5, 0.5]
        lin = int(pos[1]);
        col = int(pos[0]);
        posi += 1
        # print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

        # set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    # remove the old ones
    for item in text_del:
        item.remove()
    # append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    # titles and legends
    ax.set_title('Confusion matrix')
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    plt.tight_layout()  # set layout slim
    if save_path:
        plt.savefig(save_path)
    # plt.show()


#

def plot_confusion_matrix_from_data(y_test, predictions, labels=None, save_path=None, columns=None,
                                    annot=True, cmap="Oranges",
                                    fmt='.2f', fz=11, lw=0.5, cbar=False, figsize=[13, 13], show_null_values=0,
                                    pred_val_axis='lin'):
    """
        plot confusion matrix function with y_test (actual values) and predictions (predic),
        whitout a confusion matrix yet
    """

    # data
    if (not columns):
        # labels axis integer:
        ##columns = range(1, len(np.unique(y_test))+1)
        # labels axis string:
        from string import ascii_uppercase
        columns = ['class %s' % (i) for i in list(ascii_uppercase)[0:len(labels)]]

    confm = confusion_matrix(y_test, predictions, labels=labels)
    show_null_values = 2
    df_cm = DataFrame(confm, index=columns, columns=columns)
    pretty_plot_confusion_matrix(df_cm, fz=fz, save_path=save_path, cmap=cmap, figsize=figsize,
                                 show_null_values=show_null_values, pred_val_axis=pred_val_axis)
    plt.clf()


class MODEL_STATUS(Enum):
    UN_INIT = 0
    RANDOM_INITIALIZED = 1
    RELOAD_PRETRAINED = 2
    TRAINING = 3


def load_pretrained_model(cpk_path, reload_objects, state_keys, ignored_keys=[], device='cuda'):
    def reload_state(state, reload_dict, overwrite=False, ignored_keys=[]):
        current_dict = state.state_dict()
        if not overwrite:
            saved_dict = {k: v for k, v in reload_dict.items() if k in current_dict}

            # check in saved_dict, some tensors may not match in size.
            matched_dict = {}
            for k, v in saved_dict.items():
                cv = current_dict[k]
                if k in ignored_keys:
                    print(f"ignore key:{k}")
                    continue
                if isinstance(cv, torch.Tensor) and v.size() != cv.size():
                    print(
                        "in {}, saved tensor size {} does not match current tensor size {}"
                            .format(k, v.size(), cv.size()))
                    continue
                matched_dict[k] = v
        else:
            matched_dict = {k: v for k, v in reload_dict.items()}
        current_dict.update(matched_dict)
        state.load_state_dict(current_dict)

    if device == 'cpu':
        saved_states = torch.load(cpk_path, map_location='cpu')
    else:
        saved_states = torch.load(cpk_path)

    min_len = min(len(reload_objects), len(state_keys))
    for n in range(min_len):
        if state_keys[n] in saved_states.keys():
            if state_keys[n] == "metric":
                reload_state(reload_objects[n], saved_states[state_keys[n]], True, ignored_keys)
            else:
                reload_state(reload_objects[n], saved_states[state_keys[n]], False, ignored_keys)
    return saved_states


class JobRunner:
    class ModelMetricState:

        def __init__(self, **kwargs):
            self._state_dict = copy.deepcopy(kwargs)

        def state_dict(self):
            return self._state_dict

        def load_state_dict(self, new_dict):
            self._state_dict.update(new_dict)

    @staticmethod
    def fix_random_seeds(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def __init__(self, setting_module_file_path, settings_module=None, **kwargs):
        if setting_module_file_path is None:
            file_path = Path(inspect.getfile(self.__class__)).as_posix()
            setting_module_file_path = os.path.join(file_path.rpartition('/')[0], "settings.py")

        if settings_module is not None:
            self.settings = settings_module
        else:
            self.settings = Settings(setting_module_file_path)
        self.model_status = MODEL_STATUS.UN_INIT
        # config loggers
        [os.makedirs(x.rpartition('/')[0]) for x in
         get_value_recursively(self.settings.LOGGING, 'filename') if not os.path.exists(x.rpartition('/')[0])]
        logging.config.dictConfig(self.settings.LOGGING)
        self.logger = logging.getLogger(self.settings.EXP_NAME)

        self.exp_path = os.path.join(self.settings.MODEL_ROOT_PATH, self.settings.EXP_NAME) + '/'
        self.debug_path = os.path.join(self.settings.DEBUG_PATH, self.settings.EXP_NAME) + '/'
        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path)
        if not os.path.exists(self.debug_path):
            os.makedirs(self.debug_path)
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.exp_path, "summary"))

        def runner_excepthook(excType, excValue, traceback):
            self.logger.error("Logging an uncaught exception",
                              exc_info=(excType, excValue, traceback))

        self.model_metrics_save_dict = JobRunner.ModelMetricState()
        sys.excepthook = runner_excepthook
        # self.fix_random_seeds(33 if not hasattr(self.settings, 'RANDOM_SEED') else self.settings.RANDOM_SEED)

        with open(self.exp_path + '/settings.txt', 'wt', newline='') as fp:
            fp.write(str(self.settings))

    def device_data(self, *args):
        pass

    def print_model_parameters(self, iter):
        for name, param in self.model.named_parameters():
            name = name.replace(".", "_")
            if param.requires_grad:
                p = param.clone().cpu().data.numpy()
                self.summary_writer.add_histogram(name, p, global_step=iter)
                self.summary_writer.add_scalar("mean_{}".format(name), np.mean(p), global_step=iter)
                self.summary_writer.add_scalar("std_{}".format(name), np.std(p), global_step=iter)

    def init(self):
        # create model, initializer, optimizer, scheduler for training
        #  according to settings

        cls = get_callable_by_name(self.settings.INITIALIZER.pop('method'))
        self.parameter_initializer = cls(**self.settings.INITIALIZER)
        cls = get_callable_by_name(self.settings.MODEL.pop('method'))
        self.model = cls(**self.settings.MODEL)

        if not hasattr(self.model, 'is_cuda'):
            setattr(self.model, 'is_cuda', torch.cuda.is_available())
        self.is_cuda = self.settings.IS_CUDA & torch.cuda.is_available()
        if self.is_cuda:
            self.model = self.model.cuda()
        # plot MAC and memory
        # print("version 12.2.0rc")
        # spatial_size = self.settings.RESAMPLE_SIZE if hasattr(self.settings, 'RESAMPLE_SIZE') else (80, 80, 80)
        # macs, params = get_model_complexity_info(self.model, (self.model.in_ch_list[0], *spatial_size),
        #                                          as_strings=True,
        #                                          print_per_layer_stat=True, verbose=True)
        #
        # self.logger.info(f"macs: {macs}, params: {params}")

        if not isinstance(self.model, torch.nn.DataParallel):
            self.model.init(self.parameter_initializer)
        else:
            self.model.module.init(self.parameter_initializer)
        # create an optimizer wrapper according to settings
        cls = get_callable_by_name(self.settings.OPTIMIZER.pop('method'))
        if 'groups' in self.settings.OPTIMIZER.keys():
            self.settings.optimizer_groups_settings = self.settings.OPTIMIZER.pop('groups')
            rest_parameters = [{'params': [param for name, param in self.model.named_parameters()
                                           if not any(
                    key in name for key in self.settings.optimizer_groups_settings.keys())]}]
            self.optimizer = cls(
                [{'params': list(getattr(self.model, key).parameters()), **self.settings.optimizer_groups_settings[key]}
                 for key in self.settings.optimizer_groups_settings.keys()] + rest_parameters,
                **self.settings.OPTIMIZER)
        else:
            self.optimizer = cls(self.model.parameters(), **self.settings.OPTIMIZER)

        # create a loss wrapper according to settings
        cls = get_callable_by_name(self.settings.LOSS_FUNC.pop('method'))
        self.loss_func = cls(**self.settings.LOSS_FUNC)

        # create a scheduler wrapper according to settings
        cls = get_callable_by_name(self.settings.SCHEDULER.pop('method'))
        self.scheduler = cls(self.optimizer, **self.settings.SCHEDULER)

        self.model_status = MODEL_STATUS.RANDOM_INITIALIZED
        if hasattr(self.settings, 'USE_GRAD_SCALER'):
            self.scaler = torch.cuda.amp.GradScaler()
        self.logger.info("amp is None, Full 32 mode.")
        self.amp_module = None
        self.logger.debug("init finished, with full config = {}.".format(self.settings))
        self.current_iteration = 0
        self.epoch_n = 0
        self.saved_model_states = {}

    def generate_batches(self, *args):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def run_job(self):
        try:
            self.run()
        except:
            self.logger.exception("training encounter exception.")

    def reload_model_from_cache(self):
        if self.settings.RELOAD_CHECKPOINT:
            if self.settings.RELOAD_CHECKPOINT_PATH is not None:
                cpk_path = self.settings.RELOAD_CHECKPOINT_PATH
            else:
                # we find the checkpoints from the model output path, we reload whatever the newest.
                list_of_files = glob.glob(self.exp_path + '/*.pth')
                if len(list_of_files) == 0:
                    raise RuntimeError("{} has no checkpoint files with pth extensions."
                                       .format(self.exp_path))
                cpk_path = max(list_of_files, key=os.path.getctime)
            self.logger.info("reloading model from {}.".format(cpk_path))

            saved_model_states = torch.load(cpk_path)

            reload_dict = {"model": self.model}
            reload_keys = self.settings.RELOAD_DICT_LIST
            for reload_item in reload_keys:
                reload = reload_dict[reload_item]
                reload_state(reload, saved_model_states[reload_item])
                self.logger.info("=> loaded {}".format(reload_item))

            self.epoch_n = saved_model_states['epoch'] \
                if 'epoch' in saved_model_states.keys() else 0
            self.current_iteration = saved_model_states['iteration'] \
                if 'iteration' in saved_model_states.keys() else 0
        else:
            self.epoch_n = 0
            self.current_iteration = 0

    def update_model_state(self, **kwargs):
        self.saved_model_states['iteration'] = self.current_iteration
        self.saved_model_states['epoch'] = self.epoch_n
        self.saved_model_states['model_dict'] = self.model.state_dict()
        self.saved_model_states['optimizer_dict'] = self.optimizer.state_dict()
        self.saved_model_states['scheduler_dict'] = self.scheduler.state_dict()
        self.saved_model_states['metric'] = self.model_metrics_save_dict.state_dict()
        self.saved_model_states.update(kwargs)

    def save_model(self, **kwargs):
        self.update_model_state(**kwargs)
        cpk_name = os.path.join(self.exp_path, "{}.pth".format(self.current_iteration))
        time.sleep(10)
        torch.save(self.saved_model_states, cpk_name)
        self.logger.info("saved model into {}.".format(cpk_name))

    def archive_results(self, results):
        raise NotImplementedError


def reload_state(state, reload_dict, overwrite=False, ignored_keys=[]):
    current_dict = state.state_dict()
    if not overwrite:
        saved_dict = {k: v for k, v in reload_dict.items() if k in current_dict}

        # check in saved_dict, some tensors may not match in size.
        matched_dict = {}
        for k, v in saved_dict.items():
            cv = current_dict[k]
            if k in ignored_keys:
                print(f"ignore key:{k}")
                continue
            if isinstance(cv, torch.Tensor) and v.size() != cv.size():
                print("in {}, saved tensor size {} does not match current tensor size {}"
                      .format(k, v.size(), cv.size()))
                continue
            matched_dict[k] = v
    else:
        matched_dict = {k: v for k, v in reload_dict.items()}
    current_dict.update(matched_dict)
    state.load_state_dict(current_dict)


class LesionSegChunkTrain(JobRunner):

    def __init__(self, settings_module):
        super(LesionSegChunkTrain, self).__init__(None, settings_module)
        self.init()
        self.metrics = JobRunner.ModelMetricState()
        self.reload_model_from_cache()

        self.trace = False
        self.reset_data()
        self.logger.info(f"running on v1.0, batchsize:{self.settings.TRAIN_BATCH_SIZE}, "
                         f"input_resize:{self.settings.RESAMPLE_SIZE}")

        if os.path.exists(self.exp_path + '/records.csv'):
            self.train_records = pd.read_csv(self.exp_path + '/records.csv')
        else:
            self.train_records = pd.DataFrame(columns=['epoch'])

    def reload_model_from_cache(self):
        if self.settings.RELOAD_CHECKPOINT:
            if self.settings.RELOAD_CHECKPOINT_PATH is not None:
                cpk_path = self.settings.RELOAD_CHECKPOINT_PATH
            else:
                # we find the checkpoints from the model output path, we reload whatever the newest.
                list_of_files = glob.glob(self.exp_path + '/*.pth')
                if len(list_of_files) == 0:
                    raise RuntimeError("{} has no checkpoint files with pth extensions."
                                       .format(self.exp_path))
                cpk_path = max(list_of_files, key=os.path.getctime)
            self.logger.info("reloading model from {}.".format(cpk_path))

            saved_model_states = torch.load(cpk_path)

            reload_dict = {"model": self.model, "optimizer": self.optimizer, 'metrics': self.metrics}
            reload_keys = self.settings.RELOAD_DICT_LIST
            for reload_item in reload_keys:
                reload = reload_dict[reload_item]
                reload_state(reload, saved_model_states[reload_item])
                self.logger.info("=> loaded {}".format(reload_item))

            self.epoch_n = saved_model_states['epoch'] \
                if 'epoch' in saved_model_states.keys() else 0
            self.current_iteration = saved_model_states['iteration'] \
                if 'iteration' in saved_model_states.keys() else 0
        else:
            self.epoch_n = 0
            self.current_iteration = 0

    def ensemble_scan_augmentation(self):
        self.logger.info("LesionSegChunkTrain ensemble_scan_augmentation is called!")
        if not hasattr(self.settings, "AUG_RATIO"):
            aug_ratio = 0
        else:
            aug_ratio = self.settings.AUG_RATIO
        self.logger.info("AUG_RATIO is set to {}.".format(aug_ratio))

        class _T(object):

            def __init__(self, aug_ratio):
                self.aug_ratio = aug_ratio

            transform_pool = [
                GaussianBlur((0.3, 0.5), "random"),
                RandomMaskOut(region_range=((0.2, 0.8), (0.2, 0.8), (0.2, 0.8)),
                              region_size=((0.01, 0.05), (0.01, 0.05), (0.01, 0.05))),
                RandomFlip(3),
                RandomRotate90(3),
                GaussianAddictive((0.01, 0.02), None),
            ]

            def aug_sampling(self, aug_list):
                return [x for x in aug_list if np.random.randint(0, 10) < (10 * self.aug_ratio)]

            def __call__(self, sample):
                all_p = list(permutations(_T.transform_pool, len(_T.transform_pool)))
                p = list(random.sample(all_p, 1)[0])
                p = self.aug_sampling(p)
                for _c in p:
                    sample = _c(sample)
                return sample

        return _T(aug_ratio)

    def ensemble_scan_augmentation_wrapper(self):
        return [self.ensemble_scan_augmentation()]

    def preprocessing(self):
        window_max = self.settings.WINDOWING_MAX
        window_min = self.settings.WINDOWING_MIN
        resample_spacing = self.settings.RESAMPLE_SPACING
        resample_size = self.settings.RESAMPLE_SIZE
        resample_mode = self.settings.RESAMPLE_MODE
        return [Windowing(max=window_max, min=window_min),
                Resample(mode=resample_mode,
                         factor=resample_spacing,
                         size=resample_size
                         ),
                ]

    def val_preprocessing(self):
        resample_spacing = self.settings.RESAMPLE_SPACING
        resample_size = self.settings.RESAMPLE_SIZE
        return [
            Resample(mode='fixed_spacing',
                     factor=resample_spacing,
                     size=resample_size
                     )
        ]

    def post_preprocessing(self):
        return [ToTensor(), RemoveMeta()]

    def get_data_transforms(self, is_train):
        if is_train:
            data_transforms = transforms.Compose(self.preprocessing() +
                                                 self.ensemble_scan_augmentation_wrapper() +
                                                 self.post_preprocessing())
        else:
            data_transforms = transforms.Compose(self.val_preprocessing())
        return data_transforms

    def reset_data(self):
        self.logger.info("************Here we are at LesionSegCTSSLobeChunkTrain reset data schedule!**************")
        tr_uids = RadboudCOVIDLobeVesselChunk.get_series_uids(self.settings.DB_PATH + '/memo.csv')

        tr_dataset = RadboudCOVIDLobeVesselChunk(self.settings.DB_PATH,
                                                tr_uids,
                                                transforms=self.get_data_transforms(True))

        train_sampler = LobeChunkCTSSSampler(self.logger, tr_dataset, self.settings.TRAIN_BATCH_SIZE,
                                             balance_label_count=self.settings.BALANCED_LABEL_COUNT)

        self.ctss_frequency_map = train_sampler.ctss_frequency_map
        self.class_weights = train_sampler.class_weights
        # self.class_weights = self.settings.CLASS_WEIGHTS
        self.logger.info(f"class weights:{self.class_weights}.")
        self.tr_loader = DataLoader(tr_dataset, sampler=train_sampler, drop_last=True,
                                    batch_size=self.settings.TRAIN_BATCH_SIZE,
                                    collate_fn=collate_func_dict_fix,
                                    num_workers=self.settings.NUM_WORKERS)
        self.logger.info("Train steps {}.".format(len(self.tr_loader)))
        self.num_steps = len(self.tr_loader)
        self.val_dataset = RadboudCOVID(self.settings.DB_PATH,
                                             COPDGeneSubtyping.get_series_uids(self.settings.VALID_CSV),
                                             transforms=self.get_data_transforms(False),
                                             keep_sorted=True)

        self.logger.info("************Finished reset data schedule!**************")

    def train(self):
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_record = AverageMeter()
        end = time.time()
        for step_idx, batch_data in enumerate(self.tr_loader):
            data_time.update(time.time() - end)
            with torch.set_grad_enabled(True):
                images = batch_data["#image"].float().cuda().unsqueeze(1)
                lobes = batch_data["#lobe_reference"].float().cuda().unsqueeze(1)
                lesions = batch_data["#pseudo_lesion_reference"].float().cuda().unsqueeze(1)
                ctss = batch_data["meta"]["cle"]
                metas = batch_data['meta']
                self.optimizer.zero_grad()
                loss_tuple = self.loss_func(self.model, images, lobes, lesions,
                                            ctss,
                                            obj=self,
                                            metas=metas)
                loss = torch.stack([l * w for l, w in zip(loss_tuple,
                                                          self.settings.LOSS_FACTORS[:len(loss_tuple)])]).sum()
                loss.backward()
                self.optimizer.step()
                loss_values = [f"{l.item():.5f}" for l in loss_tuple]
                self.current_iteration += 1
                loss_record.update(loss.item(), images.size(0))
                batch_time.update(time.time() - end)
                end = time.time()
                if self.current_iteration % self.settings.LOG_STEPS == 0:
                    self.logger.info(
                        f"Epoch: [{self.epoch_n}][{step_idx}], Time {batch_time.val:.3f} ({data_time.avg:.3f})"
                        f"Loss {loss_record.val:.6f} ({loss_record.avg:.6f}), losses: {loss_values}")
        return {'tr_loss': loss_record.avg, 'tr_data_time': data_time.avg, 'tr_batch_time': batch_time.avg}

    def run(self):
        self.logger.info("start running epochs from {} to {}. "
                         .format(self.epoch_n, self.settings.NUM_EPOCHS))

        start_epoch = self.epoch_n
        for epoch_n in range(start_epoch, self.settings.NUM_EPOCHS):
            self.epoch_n = epoch_n
            self.reset_data()
            tr_metric_dict = self.train()
            if self.epoch_n % self.settings.VAL_EPOCHS == 0 or self.epoch_n == self.settings.NUM_EPOCHS - 1 \
                    or self.epoch_n < 15:
                val_metric_dict = self.validate()
                if len(val_metric_dict.keys()) > 0:
                    self.metrics.load_state_dict(val_metric_dict)
                    self.metrics.load_state_dict(tr_metric_dict)
                    self.summary_writer.add_scalars("val_metrics", val_metric_dict, global_step=self.epoch_n)
                    self.summary_writer.add_scalars("tr_metrics", tr_metric_dict, global_step=self.epoch_n)
                    _d = {'epoch': self.epoch_n, 'iteration': self.current_iteration,
                          'learning_rate': self.optimizer.param_groups[0]["lr"]}
                    _d.update(self.metrics.state_dict())
                    self.train_records = self.train_records.append(_d, ignore_index=True)
                    self.train_records.to_csv(self.exp_path + '/records.csv', index=False)

                self.scheduler.step()
            if self.epoch_n % self.settings.STATE_EPOCHS == 0 or self.epoch_n == self.settings.NUM_EPOCHS - 1:
                save_dict = {
                    "epoch": self.epoch_n,
                    "iteration": self.current_iteration,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "metrics": self.metrics.state_dict()
                }
                torch.save(save_dict, os.path.join(self.exp_path, f"{self.epoch_n}.pth"))
                self.logger.info(f"Saved epoch {self.epoch_n}/ {self.settings.NUM_EPOCHS}.")

        self.logger.info(f"Training stops at epoch {self.epoch_n}/ {self.settings.NUM_EPOCHS}.")

    def evaluate_scan(self, scan_data):
        scan = scan_data['#image']
        metadata = scan_data['meta']
        lobe = scan_data['#lobe_reference']
        uid = metadata['uid']
        now = time.time()
        epoch_debug_path = os.path.join(self.debug_path, str(self.epoch_n))
        if not os.path.exists(epoch_debug_path):
            os.makedirs(epoch_debug_path)
        with torch.no_grad():
            htp = np.zeros(scan.shape, dtype=np.float32)
            for lobe_label in np.unique(lobe)[1:]:
                lobe_binary = (lobe == lobe_label)
                lobe_crop_slices = find_crops(lobe_binary, metadata["spacing"], 5)
                lobe_chunk = lobe_binary[lobe_crop_slices]
                scan_chunk = copy.deepcopy(scan[lobe_crop_slices])
                crop_size = lobe_chunk.shape
                scan_chunk[lobe_chunk == 0] = -2048
                ret = {
                    "#image": scan_chunk.astype(np.int16),
                    "#lobe_reference": lobe_chunk.astype(np.uint8),
                    "meta":
                        {
                            "size": scan_chunk.shape,
                            "spacing": metadata['spacing'],
                            'original_spacing': metadata['spacing'],
                            'original_size': scan_chunk.shape,
                            "origin": metadata['origin'],
                            "direction": metadata['direction'],
                        }
                }
                t_ret = transforms.Compose(self.preprocessing() + self.post_preprocessing())(ret)
                t_scan_chunk = expand_dims(t_ret["#image"], 5).float().cuda()
                t_lobe_chunk = expand_dims(t_ret["#lobe_reference"], 5).float().cuda()
                if self.trace:
                    v_scan = squeeze_dims(t_scan_chunk, 3).cpu().numpy()
                    v_lobe = squeeze_dims(t_lobe_chunk, 3).cpu().numpy()
                    draw_mask_tile_single_view(windowing(v_scan, from_span=(0, 1)),
                                               [[(v_lobe > 0).astype(np.uint8)]],
                                               v_lobe > 0, 5,
                                               epoch_debug_path + f'/{uid}_{lobe_label}',
                                               colors=[(0, 0, 255)],
                                               thickness=[-1], coord_axis=0,
                                               alpha=0.3, titles=["lobe"])
                _, dense_outs = self.model(t_scan_chunk, t_lobe_chunk)
                probs = F.sigmoid(dense_outs)
                probs = squeeze_dims(F.interpolate(probs, size=crop_size, mode='trilinear',
                                                   align_corners=True), 3).cpu().numpy()
                dense_lobe_mask = ret["#lobe_reference"] > 0
                htp[lobe_crop_slices][dense_lobe_mask] = probs[
                    dense_lobe_mask]
            scan_cls_target = int(float(metadata["cle"]))
            pred_lesion_ratio = (htp * (lobe > 0)).sum() / (lobe > 0).sum()
            reg_cls_pred = self.loss_func.ratio_to_label([pred_lesion_ratio])[0]

            self.logger.info(f"val scan {uid}, "
                             f", reg_cls_pred: {reg_cls_pred}, scan_cls_target: {scan_cls_target}.")

        end = time.time()
        return reg_cls_pred, scan_cls_target, end - now

    def validate(self):
        self.logger.info(
            "\r\n************At {}, we validate {} scans.**************\r\n"
                .format(self.epoch_n, len(self.val_dataset)))

        self.model.eval()
        val_time = AverageMeter()
        all_reg_cls_preds = []
        all_cls_targets = []
        for scan_idx, data in enumerate(self.val_dataset):
            reg_cls_preds, cls_targets, elapse = self.evaluate_scan(data)
            val_time.update(elapse, 1)
            all_reg_cls_preds.append(reg_cls_preds)
            all_cls_targets.append(cls_targets)
            self.logger.info(f"Validation step {scan_idx + 1}/{len(self.val_dataset)}.")

        epoch_debug_path = os.path.join(self.debug_path, str(self.epoch_n)) + '/'
        if not os.path.exists(epoch_debug_path):
            os.makedirs(epoch_debug_path)

        all_reg_cls_preds = np.asarray(all_reg_cls_preds)
        all_cls_targets = np.asarray(all_cls_targets)
        v_metrics = {'val_time': val_time.avg}

        val_acc_cls = accuracy_score(all_cls_targets, all_reg_cls_preds)
        plot_confusion_matrix_from_data(all_cls_targets, all_reg_cls_preds,
                                        labels=list(range(0, 6)), save_path=epoch_debug_path + f'cm_reg_cls')
        v_metrics.update({f'val_acc_reg_cls': val_acc_cls})

        self.logger.info(f"val_metrics: {v_metrics}")
        return v_metrics


class LesionSegTest(JobRunner):

    def __init__(self, settings_module=None, scan_path=None
                 , output_path=None, task_name='test'):
        super(LesionSegTest, self).__init__(None, settings_module)

        self.scan_path = scan_path
        self.settings_module = settings_module
        self.output_path = output_path
        self.task_name = task_name
        resample_size = self.settings.RESAMPLE_SIZE
        test_spacing = self.settings.TEST_RESAMPLE_SPACING

        self.test_set = RadboudCOVID(self.settings.DB_PATH, RadboudCOVID.get_series_uids(self.settings.TEST_CSV),
                                     task=task_name, keep_sorted=True,
                                     transforms=transforms.Compose([
                                         Resample(mode="fixed_spacing",
                                                  factor=test_spacing,
                                                  size=resample_size
                                                  ),

                                     ]))

        self.settings.RELOAD_CHECKPOINT = True
        self.init()
        self.reload_model_from_cache()

    def preprocessing(self):
        window_max = self.settings.WINDOWING_MAX
        window_min = self.settings.WINDOWING_MIN
        resample_spacing = self.settings.RESAMPLE_SPACING
        resample_size = self.settings.RESAMPLE_SIZE
        resample_mode = self.settings.RESAMPLE_MODE
        return [Windowing(max=window_max, min=window_min),
                Resample(mode=resample_mode,
                         factor=resample_spacing,
                         size=resample_size
                         )
                ]

    def post_preprocessing(self):
        return [ToTensor()]

    def archive_results(self, scan, lobe, heatmap, pred, post_pred, ref, meta):
        output_path = os.path.join(self.output_path, self.task_name)
        post_path = os.path.join(output_path, "post")
        if not os.path.exists(post_path):
            os.makedirs(post_path)

        heatmap_path = os.path.join(output_path, "heatmap")
        if not os.path.exists(heatmap_path):
            os.makedirs(heatmap_path)

        screenshots_path = os.path.join(output_path, "screenshots")
        if not os.path.exists(screenshots_path):
            os.makedirs(screenshots_path)
        series_uid = meta['uid']
        heatmap_w = windowing(heatmap, from_span=(0, 1)).astype(np.uint8)
        # label_name_mapping = self.settings.LABEL_NAME_MAPPING
        write_array_to_mha_itk(output_path, [pred.astype(np.uint8)],
                               [series_uid], type=np.uint8,
                               origin=meta["origin"][::-1],
                               direction=np.asarray(meta["direction"]).reshape(3, 3)[
                                         ::-1].flatten().tolist(),
                               spacing=meta["original_spacing"][::-1])
        write_array_to_mha_itk(heatmap_path, [heatmap_w.astype(np.uint8)],
                               [series_uid], type=np.uint8,
                               origin=meta["origin"][::-1],
                               direction=np.asarray(meta["direction"]).reshape(3, 3)[
                                         ::-1].flatten().tolist(),
                               spacing=meta["original_spacing"][::-1])

        write_array_to_mha_itk(post_path, [post_pred.astype(np.uint8)],
                               [series_uid], type=np.uint8,
                               origin=meta["origin"][::-1],
                               direction=np.asarray(meta["direction"]).reshape(3, 3)[
                                         ::-1].flatten().tolist(),
                               spacing=meta["original_spacing"][::-1])

        labels = np.unique(ref)
        self.logger.debug("archive results generate screenshots with unique labels {} in prediction."
                          .format(labels))

        draw_mask_tile_singleview_heatmap(windowing(scan).astype(np.uint8),
                                          [[(pred * 255).astype(np.uint8)],
                                           [(post_pred * 255).astype(np.uint8)],
                                           [(ref * 255).astype(np.uint8)],
                                           [windowing(heatmap, from_span=(0, 1)).astype(np.uint8)]],
                                          np.logical_or(pred > 0, ref > 0) > 0, 5,
                                          screenshots_path + f'/{series_uid}/',
                                          titles=["pred_lesion", "pred_lesion_post", "lesion", "pred_cam"])

    def run(self):
        self.logger.info(f"total {len(self.test_set)} files to be processed from {self.scan_path}.")
        self.model.eval()
        if self.output_path is None:
            epoch_n = self.saved_model_states['epoch']
            current_iteration = self.saved_model_states['iteration']
            metrics = self.saved_model_states['metrics']
            self.output_path = os.path.join(self.exp_path, "{:d}_{:d}_{:.5f}"
                                            .format(epoch_n,
                                                    current_iteration, metrics["val_iou"]))

        output_path = os.path.join(self.output_path, self.task_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            with open(output_path + '/settings.txt', 'wt', newline='') as fp:
                fp.write(str(self.settings))
        uids = []
        for uid in self.test_set.uids:
            scan_path = output_path + '/{}.mha'.format(uid)
            if os.path.exists(scan_path):
                self.logger.warn("We have already archived results for scan {}".format(uid))
            else:
                self.logger.info("We add scan {} because {} does not exist.".format(uid, scan_path))
                uids.append(uid)
        if os.path.exists(output_path + '/records.csv'):
            self.scan_records = pd.read_csv(output_path + '/records.csv')
        else:
            self.scan_records = pd.DataFrame(columns=['uid'])
        self.test_set.uids = uids
        self.logger.info("Start {} scans after exclusion."
                         .format(len(self.test_set)))
        average_time = 0
        try:
            scan_cls_preds = []
            scan_cls_targets = []
            with torch.no_grad():
                for scan_idx, scan_data in enumerate(self.test_set):
                    try:
                        start = time.time()
                        scan = scan_data['#image']
                        metadata = scan_data['meta']
                        lobe = scan_data['#lobe_reference']
                        lesion = scan_data['#lesion_reference']
                        vessel = scan_data['#vessel_reference']
                        uid = metadata['uid']
                        htp = np.zeros(scan.shape, dtype=np.float32)
                        scan_accs = []

                        for lobe_label in range(1, 6):
                            lobe_binary = (lobe == lobe_label)
                            cls_target = int(metadata['patient_meta'][self.test_set.metric_k_mapping[lobe_label]])
                            if lobe_binary.sum() < 1e-7:
                                scan_cls_preds.append(cls_target)
                                scan_cls_targets.append(cls_target)
                                continue
                            lobe_crop_slices = find_crops(lobe_binary, metadata["spacing"],
                                                          self.test_set.crop_border)
                            lobe_chunk = lobe_binary[lobe_crop_slices]
                            scan_chunk = copy.deepcopy(scan[lobe_crop_slices])
                            crop_size = lobe_chunk.shape
                            scan_chunk[lobe_chunk == 0] = -2048
                            ret = {
                                "#image": scan_chunk.astype(np.int16),
                                "#lobe_reference": lobe_chunk.astype(np.uint8),
                                "meta":
                                    {
                                        "size": scan_chunk.shape,
                                        "spacing": metadata['spacing'],
                                        'original_spacing': metadata['spacing'],
                                        'original_size': scan_chunk.shape,
                                        "original_origin": metadata['original_origin'],
                                        "original_direction": metadata['original_direction'],
                                        "origin": metadata['origin'],
                                        "direction": metadata['direction'],
                                    }
                            }
                            t_ret = transforms.Compose(self.preprocessing() + self.post_preprocessing())(ret)
                            t_scan_chunk = expand_dims(t_ret["#image"], 5).float().cuda()
                            t_lobe_chunk = expand_dims(t_ret["#lobe_reference"], 5).float().cuda()
                            _, dense_outs = self.model(t_scan_chunk, t_lobe_chunk)
                            pool_outs = self.model.pooling_dense_features(dense_outs, t_lobe_chunk)

                            cls_pred = torch.max(pool_outs, dim=-1)[-1].item()
                            scan_cls_preds.append(cls_pred)

                            scan_cls_targets.append(cls_target)
                            scan_accs.append(cls_pred == cls_target)
                            dense_outs = F.interpolate(dense_outs, size=crop_size, mode='trilinear',
                                                       align_corners=True).squeeze(0)
                            dense_outs = F.relu(dense_outs)
                            dense_out = dense_outs[cls_pred]
                            dense_out = dense_out / dense_out.max()

                            if cls_pred < 1e-7:
                                dense_out.zero_()

                            dense_lobe_mask = ret["#lobe_reference"] > 0
                            htp[lobe_crop_slices][dense_lobe_mask] = dense_out.cpu().numpy()[
                                dense_lobe_mask]

                        max_norm_htp = htp

                        _, th = binary_cam(max_norm_htp[lobe > 0])
                        lesion_pred = max_norm_htp > th

                        w_scan = windowing(scan, to_span=(0, 1))
                        _, th = binary_cam(w_scan[lobe > 0], 0.75)
                        lesion_pred_post = np.logical_and(np.logical_and(lesion_pred, w_scan > th),
                                                          np.logical_not(vessel > 0)).astype(np.uint8)
                        lesion_pred = lesion_pred.astype(np.uint8)
                        # resample and compute metrics
                        original_spacing = np.asarray(metadata['original_spacing']).flatten().tolist()
                        original_size = np.asarray(metadata['original_size']).flatten().tolist()
                        spacing = np.asarray(metadata['spacing']).flatten().tolist()
                        lesion_pred, _ = resample(lesion_pred, spacing, factor=2, required_spacing=original_spacing,
                                                  new_size=original_size, interpolator='nearest')
                        lesion_pred_post, _ = resample(lesion_pred_post, spacing, factor=2,
                                                       required_spacing=original_spacing,
                                                       new_size=original_size, interpolator='nearest')
                        lesion, _ = resample(lesion, spacing, factor=2, required_spacing=original_spacing,
                                             new_size=original_size, interpolator='nearest')
                        scan, _ = resample(scan, spacing, factor=2, required_spacing=original_spacing,
                                           new_size=original_size, interpolator='linear')
                        max_norm_htp, _ = resample(max_norm_htp, spacing, factor=2, required_spacing=original_spacing,
                                                   new_size=original_size, interpolator='linear')
                        m_iou = IOU(lesion_pred > 0, lesion > 0, 1e-5)
                        m_iou_post = IOU(lesion_pred_post > 0, lesion > 0, 1e-5)
                        m_acc = np.mean(scan_accs)
                        m_dice = Dice(lesion_pred > 0, lesion > 0, 1e-5)
                        m_dice_post = Dice(lesion_pred_post > 0, lesion > 0, 1e-5)

                        _d = {
                            "uid": uid,
                            "iou": m_iou,
                            "iou_post": m_iou_post,
                            "dice": m_dice,
                            "dice_post": m_dice_post,
                            "acc": m_acc
                        }

                        self.archive_results(scan, None, max_norm_htp, lesion_pred, lesion_pred_post, lesion, metadata)
                        self.scan_records = self.scan_records.append(_d, ignore_index=True)
                        self.logger.info(f"val scan {uid}, iou:{m_iou}, iou_post:{m_iou_post}, acc:{m_acc}.")
                        if scan_idx % 5 == 0 or scan_idx == (len(self.test_set) - 1):
                            self.scan_records.to_csv(output_path + '/records.csv', index=False)
                        end = time.time()
                        self.logger.info("Finished {}, in {} seconds."
                                         .format(scan_idx, end - start))
                    except StopIteration:
                        raise StopIteration
                    except Exception:
                        track = traceback.format_exc()
                        self.logger.error("Cannot process {} test scans with uid {}, {}."
                                          .format(scan_idx, uid, track))

            plot_confusion_matrix_from_data(scan_cls_targets, scan_cls_preds,
                                            labels=list(range(0, 6)), save_path=output_path + '/cm')
            pd.DataFrame({"target": scan_cls_targets, "pred": scan_cls_preds}) \
                .to_csv(output_path + '/lobewise.csv')
        except StopIteration:
            average_time /= len(self.test_set)
            self.logger.info("Finished testing, average time = {}".format(average_time))

