
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
    results = model.train(
        data=data_config_path,
        epochs=params['epochs'], 
        patience=params['patience'],
        batch=params['batch'],
        imgsz = params['imgsz'],
        save=True,
        save_period=params['save_period'],
        cache=params['cache'],
        device=params['device'],

        project='runs/train',
        name=f"{params['save_file']}",

        exist_ok=params['exist_ok'],
        single_cls=params['single_cls'],
        resume=params['resume'],
        freeze=params['freeze'],
        box=params['box'],
        cls=params['cls'],
        dropout=params['dropout'],

        plots=True,
        )
    LOG.info('Training complete')
    return results

def export(model):
    path = model.export(format='onnx')
    LOG.info(f'Model exported to {path}')
    return path
    

def val(model):
    LOG.info('Validating model')
    metrics = model.val(
        imgsz=params['imgsz'],
        batch=16,
        save_json=True, 
        # conf=0.001,
        # iou=0.7,
        max_det=20,
        device=None,
        plots=True, 
        classes=params['classes'], 
        split='test',

        project='runs/val', 
        name=f"{params['save_file']}", 
        
        workers=params['workers'],
        visulize=True,
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
        task='detect',
        save=True, 
        project='runs/predict', 
        name=f"{params['save_file']}", 
        save_txt=True,
        exist_ok=params['exist_ok'],
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

    

    file = pathlib.Path(f'config/{args.config}')
    if not file.exists():
        print(f'Config file not found: {file}, using default parameters.')
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
    if config:
        params.update(config)
    
    

    LOG = datautils.init_logging(params['log_file'], logging.DEBUG)
    LOG.info('---------------------------')
    LOG.info('---------------------------')
    LOG.info('---------------------------')
    LOG.info('----------STARTED----------')
    LOG.info('---------------------------')
    LOG.info('---------------------------')
    

    params['task'] = args.task
    params['resume'] = args.resume
    params['config_file'] = args.config
    
    if not os.path.exists(params['data']):
        LOG.error(f'Data folder not found: {params["data"]}')
        exit(1)

    params['noload'] = args.noload
    if params['device'] == None:
        params['device'] = -1 if torch.cuda.is_available() else 'cpu'
    # params['data'] = os.path.join('./datasets', args.data)
    # params['data_root'] = re.search(r'[^./\\]+', args.data).group(0)
    # params['model'] = args.model if args.model else params['model']
    # model_path = Path(params['model'])
    # params['model_name'] = model_path.stem
    
    # params['epochs'] = args.epochs
    # params['save_period'] = 10 if params['epochs'] > 50 else -1
    # params['log_level'] = args.log_level.upper()



    LOG.info(f'Task: {params["task"]}')
    LOG.info(f'Input Args: {args}\n')
    LOG.info(f'Config: {params}\n')



    model = init_model()
    model.add_callback('on_train_epoch_start', datautils.on_train_epoch_start)
    model.add_callback('on_train_epoch_end', datautils.on_train_epoch_end)

    if params['task'] == 'train':
        train(model)
        export(model)

    if params['task'] == 'val':
        val(model)

    if params['task'] == 'predict': 
        predict(model)

