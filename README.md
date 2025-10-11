修改run.ps1文件:

- options:
    -   -h, --help            show this help message and exit
    - --train               Task: train
    - --resume              Resume training
    - --val                 Task: valuate
  - --predict             Task: predict
  - --noload              Do not load saved model

  - -d DATA_ROOT, --data_root DATA_ROOT
                        data folder in the datasets.
  - -mp MODEL_DIR, --model_dir MODEL_DIR
                        Path to saved model
  - -m MODEL, --model MODEL
                        Model type
  - -l LOG_LEVEL, --log_level LOG_LEVEL
                        Log level

加载模型:

`-m`输入模型地址，或输入如yolo12n.pt等预训练模型名称，自动加载预训练模型

训练模型时自动保存在`runs/train/weights/best.pt`中
export导出到`runs/train/weights/best.onnx`中
val模式 自动使用val数据集，保存到`runs/val`中
predict模式 输入以`dataset`为根目录的完整数据路径

当train,val,predict模式重复运行时，会自动新建如train2,val3,predict4等目录，并保存到对应目录中
