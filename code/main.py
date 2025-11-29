
import logging
import sys
import pickle
import os
from pathlib import Path
import re
import argparse

import numpy as np
import torch
import torch.nn as nn
import datautils

import yaml
import pathlib
import shutil
import inspect

from ultralytics import YOLO

params = {
    'task': 'train',

    'batch': 0.90,
    'classes': None,
    'data': './datasets/data',
    'device': 0,
    'imgsz': 640,

    
    'model': 'yolo12n.pt',
    'model_name': 'yolo12n',
    'noload': False,
    
    
    'resume': False,
    'epochs': 10,
    'patience': 100,
    'save': True,
    'save_period': 10, # checkpoint
    'cache': False,
    'workers': 8, 
    'single_cls': False,
    'freeze': None,
    'box': 7.5,
    'cls': 0.5,
    'dropout': 0.0,

    'log_level': 'DEBUG',
    'log_file': 'yolo.log',
    
    'save_file': None,
    'exist_ok': False,
}

def init_model():
    cuda_num = torch.cuda.device_count()
    LOG.debug(f'CUDA available: {torch.cuda.is_available()}')
    LOG.debug(f'CUDA num: {cuda_num}')
    LOG.debug(f'Using device: {params["device"]}')

    model_path = params["model"]
    if not os.path.exists(model_path):
        LOG.info(f'Model not found: {model_path}')
        
    if not os.path.exists(model_path) or params['noload']:
        model = YOLO(f"{params['model']}")
        LOG.info(f'Created new model: {params["model"]}')
    else:
        model = YOLO(model_path)
        LOG.info(f'Loaded existing model: {params["model"]}')

    return model

def train(model):
    data_config_path = os.path.join(params['data'], 'data.yaml')
    LOG.info(f'Data config: {data_config_path}')

    LOG.info(f'Starting training')
    # results = model.train(
    #     data=data_config_path,
    #     epochs=params['epochs'], 
    #     patience=params['patience'],
    #     batch=params['batch'],
    #     imgsz = params['imgsz'],
    #     save=True,
    #     save_period=params['save_period'],
    #     cache=params['cache'],
    #     device=params['device'],

    #     project='runs/train',
    #     name=f"{params['save_file']}",

    #     exist_ok=params['exist_ok'],
    #     single_cls=params['single_cls'],
    #     resume=params['resume'],
    #     freeze=params['freeze'],
    #     box=params['box'],
    #     cls=params['cls'],
    #     dropout=params['dropout'],

    #     plots=True,
    #     hsv_s=params['hsv_s'] if 'hsv_s' in params else 0.7,
    #     hsv_v=params['hsv_v'] if 'hsv_v' in params else 0.4,
    #     )
    # results = model.train(cfg=config_file,
    #                       data=data_config_path,
    #                       save=True, 
    #                       project='runs/train', 
    #                       name=f"{params['save_file']}", 
    #                       plots=True)
    # valid = set(inspect.signature(model.train).parameters.keys())
    train_params_path = './config/all_train_params.yaml' 
    with open(train_params_path, 'r') as f:
        train_params = yaml.safe_load(f)
    filtered_params = {k: v for k, v in params.items() if k in train_params}
    filtered_params.update({
        'data': data_config_path,
        'project': 'runs/train',
        'name': f"{params['save_file']}",
        'plots': True })
    # print(filtered_params)
    # train_params.update(params)
    results = model.train(**filtered_params)
    LOG.info('Training complete')
    return results

def export(model):
    path = model.export(format='onnx')
    LOG.info(f'Model exported to {path}')
    return path
    

def val(model):
    LOG.info('Validating model')
    data_config_path = os.path.join(params['data'], 'data.yaml')
    LOG.info(f'Data config: {data_config_path}')

    metrics = model.val(
        data=data_config_path,
        imgsz=params['imgsz'],
        batch=0.8,
        save_json=True, 
        # conf=0.001,
        # iou=0.7,
        max_det=20,
        device=None,
        plots=True, 
        classes=params['classes'], 
        split=params['split'] if 'split' in params else 'test',

        project='runs/val', 
        name=f"{params['save_file']}", 
        
        workers=params['workers'],
        visualize=True,
        exist_ok=params['exist_ok'],
        )
    LOG.info(f'Valid mAPs50-95: {metrics.box.maps}')
    LOG.info(f'Valid speed: {metrics.speed}')
    LOG.info('Validation complete')
    return metrics

def predict(model):
    LOG.info('Predicting')
    if os.path.exists(params['data']):
        LOG.info('Found predict data')
    else :
        LOG.info('No data found')
    source = params['data']
    results = model.predict(
        imgsz=1280,
        source=source, 
        save=True, 
        project='runs/predict', 
        name=f"{params['save_file']}", 
        save_txt=True,
        exist_ok=params['exist_ok'],
        # stream=True,
        )
    LOG.info('Prediction complete')
    return results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Config file', type=str, required=False, default='car_train.yaml')
    # parser.add_argument('--train', help='Task: train', action='store_true')
    parser.add_argument('--task', help='Task: train, val, predict', type=str, required=True)
    parser.add_argument('--resume', help='Resume training', action='store_true')
    parser.add_argument('--noload', help='Do not load saved model', action='store_true')
    # parser.add_argument('--val', help='Task: valuate', action='store_true')
    # parser.add_argument('--predict', help='Task: predict', action='store_true')
    
    # parser.add_argument('-d', '--data', help='data folder in the datasets.',
    #                     type=str, required=False, default='./data')
    # parser.add_argument('-m', '--model', help='Model type', 
    #                     type=str, required=False, default=None)
    # parser.add_argument('-l', '--log_level', help='Log level',
    #                     type=str, required=False, default='DEBUG') 
    # parser.add_argument('-e', '--epochs', help='Number of epochs',
    #                     type=int, required=False, default=10)
    
    args = parser.parse_args()

    

    global config_file
    config_file = pathlib.Path(f'config/{args.config}')
    if not config_file.exists():
        print(f'Config file not found: {config_file}, using default parameters.')
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    if config:
        params.update(config)


    LOG = datautils.init_logging(params['log_file'], logging.DEBUG)
    LOG.info('---------------------------')
    LOG.info('---------------------------')
    LOG.info(f'Task: {params["task"]: <6}'+'-'*15)
    LOG.info('----------STARTED----------')
    LOG.info('---------------------------')
    LOG.info('---------------------------')


    if 'version' in params:
        LOG.info(f'Model version: {params["version"]}')

    if 'comment' in params:
        cmt_path = Path(f'./runs/{params["task"]}/{params["save_file"]}/comments.txt')
        if not cmt_path.parent.exists():
            cmt_path.parent.mkdir(parents=True)
        with open(cmt_path, 'w') as f:
            f.write(params['comment'])
    

    params['task'] = args.task
    params['resume'] = args.resume
    params['config_file'] = args.config
    
    if not os.path.exists(params['data']):
        LOG.error(f'Data folder not found: {params["data"]}')
        exit(1)

    params['noload'] = args.noload
    if params['device'] == None:
        params['device'] = -1 if torch.cuda.is_available() else 'cpu'




    # LOG.info(f'Task: {params["task"]}')
    LOG.info(f'Input Args: {args}\n')
    LOG.info(f'Config: {params}\n')



    model = init_model()
    # model.add_callback('on_train_epoch_start', datautils.on_train_epoch_start)
    model.add_callback('on_fit_epoch_end', datautils.on_fit_epoch_end)

    if params['task'] == 'train':
        train(model)
        path = Path(export(model))
        path = path.with_suffix('.pt')
        if not 'model_name' in params:
            params['model_name'] = params['version'] if 'version' in params else 'unnamed'
            
        if not os.path.exists(f'./models/{params["model_name"]}'):
            os.makedirs(f'./models/{params["model_name"]}')
        shutil.copy(path, f'./models/{params["model_name"]}/best.pt')
        # shutil.copy(model.best, f'./models/{params["model_name"]}/best.pt')
        if 'comment' in params:
            cmt_path = Path(f'./models/{params["model_name"]}/comments.txt')
            if not cmt_path.parent.exists():
                cmt_path.parent.mkdir(parents=True)
            with open(cmt_path, 'w') as f:
                f.write(params['comment'])
        

    if params['task'] == 'val':
        val(model)

    if params['task'] == 'predict': 
        predict(model)

