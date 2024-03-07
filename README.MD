


## Install Using Conda
```
conda create --name DKE python=3.8
conda activate DKE
conda install pytorch torchvision cudatoolkit=YOUR_VERSION -c pytorch
python -m pip install -r requirement.txt
```


## Data Preparation

refer to repo:https://github.com/WenmuZhou/OCR_DataSet to download related datasets

The annotation file formats supported by our model are as follows, separated by "\t":

```
" Image file name             Image annotation information encoded by json.dumps"
ch4_test_images/img_61.jpg    [{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}, {...}]
```




## Train



1. config the `dataset['dataset']['train']['data_path']'`,`dataset['dataset']['validate']['data_path']`in [config/fintune/total_r50.yaml](config/fintune/total_r50.yaml)
* Single gpu train
```
CUDA_VISIBLE_DEVICES=0 python tools/train.py config/fintune/total_r50.yaml --output_dir WHERE_TO_STORE_YOUR_LOG_AND_WEIGHTS
```
* Multi-gpu training
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 tools/train.py config/fintune/total_r50.yaml  --output_dir WHERE_TO_STORE_YOUR_LOG_AND_WEIGHTS
```
## Test

[tools/eval.py](tools/eval.py) is used to generate the results for testing


```bash
CUDA_VISIBLE_DEVCIES=0 python tools/eval.py  config/fintune/total_r50.yaml --model_path MODEL_PATH --dataset DATASET_NAME
```
## Evaluation
```bash
cd eval/
./eval_tt.sh
```

