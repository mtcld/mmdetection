import argparse
import copy
import sys
import os
import os.path as osp
import time
import hypertune
import json
import warnings
import shutil
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import init_dist
from mmcv.utils import get_git_hash
from google.cloud import storage
from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/etc/credentials.json"
print('Started')

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument(
        '--job-dir',  # Handled automatically by AI Platform
        help='GCS location to write checkpoints and export models'
        #required=True
        )
    parser.add_argument(
        '--lr', default=0.02, 
        type=float, 
        help='Learning rate parameter')
    parser.add_argument('--MOMENTUM',  # Specified in the config file
        type=float,
        default=0.9,
        help='SGD momentum (default: 0.5)')
    parser.add_argument(
        '--INIT_STEP', default=8, 
        type=int, 
        help='initial step')
    parser.add_argument(
        '--STEP_RANGE', default=3, 
        type=int, 
        help='step range')
    parser.add_argument(
        '--PRE_NMS_TOP_N_TRAIN', default=2000, 
        type=int, 
        help='PRE_NMS_TOP_N_TRAIN')
    parser.add_argument(
        '--PRE_NMS_TOP_N_TEST', default=1000, 
        type=int, 
        help='PRE_NMS_TOP_N_TEST')
    parser.add_argument(
        '--POST_NMS_TOP_N_TRAIN', default=2000, 
        type=int, 
        help='POST_NMS_TOP_N_TRAIN')
    parser.add_argument(
        '--POST_NMS_TOP_N_TEST', default=1000, 
        type=int, 
        help='POST_NMS_TOP_N_TEST')
    parser.add_argument(
        '--NMS_THRESH_TRAIN', default=0.7, 
        type=float, 
        help='NMS_THRESH_TRAIN')
    parser.add_argument(
        '--NMS_THRESH_TEST', default=0.7, 
        type=float, 
        help='NMS_THRESH_TEST')
    parser.add_argument(
        '--POS_FRACTION_RPN', default=0.7, 
        type=float, 
        help='POS_FRACTION_RPN')

#    parser.add_argument(
#        '--job-dir',  # Handled automatically by AI Platform
#        help='GCS location to write checkpoints and export models'
#        #required=True
#        )
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args

def save_model(job_dir, model_name):
    """Saves the model to Google Cloud Storage"""
    job_dir = job_dir.replace('gs://', '')  
    bucket_id = job_dir.split('/')[0]
    bucket_path = job_dir.lstrip('{}/'.format(bucket_id))
    #plot_names=glob.glob('./*.png')
    #plot_names.extend([model_name,dice_dict_name])
    all_files=[model_name]
    for f in all_files:
        bucket = storage.Client().bucket(bucket_id)
        if f==model_name:
            blob = bucket.blob('{}/{}'.format(
                bucket_path,
                f[f.rfind('/')+1:]))
        else:
            blob= bucket.blob('{}/{}'.format(
                bucket_path,
                f))
        blob.upload_from_filename(f)




def main():
    print('started')
    #os.system('gsutil cp gs://hptuning2/scratch_latest.zip .')
    #os.system('gsutil cp gs://hptuning2/scratch_latest_mask.zip .')
    #os.system('unzip scratch_latest_mask.zip')
    #os.system('unzip scratch_latest.zip')

    print('point a')
    args = parse_args()
    #job_dir = args.pop('job_dir')
    print('args '*100)
    cfg = Config.fromfile(args.config)
    print(cfg)
    print(args.config)
    cfg['optimizer']['lr']=float(args.lr)
    cfg['optimizer']['momentum']=float(args.MOMENTUM)
    cfg['lr_config']['step']=[args.INIT_STEP,args.INIT_STEP+args.STEP_RANGE]
    config_name=args.config
    config_name=config_name[config_name.rfind('/')+1:config_name.rfind('.')]
    print(config_name)
    print('#'*100)
    print(cfg)
    print('#'*100)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        rm_dir=osp.join('./work_dirs',osp.splitext(osp.basename(args.config))[0])
        cfg.work_dir = rm_dir
    #print(os.getcwd())
    #print(cfg.work_dir)
    #sys.exit()
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)
    print('Training completed')
    
    with open('final_epoch.json') as f:
        epoch_data=json.load(f)
    with open('map_data.json') as f:
        map_data=json.load(f)
    for md in map_data['map']:
        if md['epoch'] == int(epoch_data['epoch']):
            map_25=md['bbox_mAP_25']

    model_path='/mmdetection/work_dirs/'+config_name+'/epoch_'+str(epoch_data['epoch'])+'.pth'
    print(model_path)
    
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='map25', metric_value=map_25,
                                            global_step=1)

    save_model(args.job_dir,model_path)

if __name__ == '__main__':
    main()
