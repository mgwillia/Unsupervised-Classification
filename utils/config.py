"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import yaml
from easydict import EasyDict
from utils.utils import mkdir_if_missing

def create_config(config_file_env, config_file_exp):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']
   
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    cfg = EasyDict()
   
    # Copy
    for k, v in config.items():
        cfg[k] = v

    # Set paths for pretext task (These directories are needed in every stage)
    base_dir = os.path.join(root_dir, cfg['train_db_name'])
    pretext_dir = os.path.join(base_dir, 'pretext')
    mkdir_if_missing(base_dir)
    mkdir_if_missing(pretext_dir)
    cfg['pretext_dir'] = pretext_dir
    cfg['pretext_checkpoint'] = os.path.join(pretext_dir, 'checkpoint.pth.tar')
    cfg['pretext_model'] = os.path.join(pretext_dir, 'model.pth.tar')
    cfg['topk_neighbors_train_path'] = os.path.join(pretext_dir, 'topk-train-neighbors.npy')
    cfg['topk_neighbors_val_path'] = os.path.join(pretext_dir, 'topk-val-neighbors.npy')
    cfg['topk_strangers_train_path'] = os.path.join(pretext_dir, 'topk-train-strangers.npy')
    cfg['topk_strangers_val_path'] = os.path.join(pretext_dir, 'topk-val-strangers.npy')
    cfg['centroid_indices_train_path'] = os.path.join(pretext_dir, 'centroid-train-indices.npy')

    # If we perform clustering or self-labeling step we need additional paths.
    # We also include a run identifier to support multiple runs w/ same hyperparams.
    if cfg['setup'] in ['scan', 'selflabel', 'linearprobe']:
        base_dir = os.path.join(root_dir, cfg['train_db_name'])
        linearprobe_dir = os.path.join(base_dir, 'linearprobe')
        scan_dir = os.path.join(base_dir, 'scan')
        scanf_dir = os.path.join(base_dir, 'scanf')
        scanc_dir = os.path.join(base_dir, 'scanc')
        scankl_dir = os.path.join(base_dir, 'scankl')
        selflabel_dir = os.path.join(base_dir, 'selflabel') 
        mkdir_if_missing(base_dir)
        mkdir_if_missing(linearprobe_dir)
        mkdir_if_missing(scan_dir)
        mkdir_if_missing(scanf_dir)
        mkdir_if_missing(scanc_dir)
        mkdir_if_missing(scankl_dir)
        mkdir_if_missing(selflabel_dir)
        cfg['linearprobe_dir'] = linearprobe_dir
        cfg['linearprobe_checkpoint'] = os.path.join(linearprobe_dir, 'checkpoint.pth.tar')
        cfg['linearprobe_model'] = os.path.join(linearprobe_dir, 'model.pth.tar')
        cfg['scan_dir'] = scan_dir
        cfg['scan_checkpoint'] = os.path.join(scan_dir, 'checkpoint.pth.tar')
        cfg['scan_model'] = os.path.join(scan_dir, 'model.pth.tar')
        cfg['scanf_dir'] = scanf_dir
        cfg['scanf_checkpoint'] = os.path.join(scanf_dir, 'checkpoint.pth.tar')
        cfg['scanf_model'] = os.path.join(scanf_dir, 'model.pth.tar')
        cfg['scanc_dir'] = scanc_dir
        cfg['scanc_checkpoint'] = os.path.join(scanc_dir, 'checkpoint.pth.tar')
        cfg['scanc_model'] = os.path.join(scanc_dir, 'model.pth.tar')
        cfg['scankl_dir'] = scankl_dir
        cfg['scankl_checkpoint'] = os.path.join(scankl_dir, 'checkpoint.pth.tar')
        cfg['scankl_model'] = os.path.join(scankl_dir, 'model.pth.tar')
        cfg['selflabel_dir'] = selflabel_dir
        cfg['selflabel_checkpoint'] = os.path.join(selflabel_dir, 'checkpoint.pth.tar')
        cfg['selflabel_model'] = os.path.join(selflabel_dir, 'model.pth.tar')

    return cfg 
