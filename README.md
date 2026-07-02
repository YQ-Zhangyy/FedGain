# FedGain

This is the official implementation of the following paper:

Yuqing Zhang, Changli Zhou, Binghuang Huang, Hui Tian. _FedGain: Toward Negative-Gain-Free Client Collaboration in Federated
Learning_.
ICML 2026.

## Introduction

**FedGain** is a method that defines the Effective Federated Capacity (EFC) metric through distribution shift and data volume scale, quantifies data utility in heterogeneous environments, and achieves heterogeneous scaling laws, thereby avoiding the problem of negative gains in federated learning. 

## Requirements

- torch==1.11.0
- torchvision==0.12.0
- cudatoolkit==10.2.89
- cudnn==7.6.5
- numpy==1.18.5
- tqdm==4.65.0
- matplotlib==3.7.1

## Run

Here we provide an example script for experiments with FedAvg.

1. data_prepare

```bash
python data_prepare.py \
  --dataset cifar10 \
  --num_clients 20 \
  --partition stratified \
  --subsample clusterexp_2_2.0 \
  --seed 0 
```

2. distance_matrix

```bash
python distance_matrix.py \
  --dataset rotated-cifar10 \
  --partition_config client_20_partition_stratified_subsample_clusterexp_2_2.0_seed_0 \
  --contrastive_rounds 20 \
  --contrastive_lr 0.001 \
  --projection_dim 128 \
  --temperature 0.1 \
  --batch_size 32 \
  --use_valid \
  --cuda \
  --seed 0 
```

The estimated pairwise divergence will be saved to `./distance`.

3. gain_structure

```bash
python gain_structure.py \
  --dataset rotated-cifar10 \
  --partition_config client_20_partition_stratified_subsample_clusterexp_2_2.0_seed_0 \
  --divergence_config contrastive_seed_0 \
  --C 10.0 \
  --seed 0 
```

The solved collaboration structure will be saved to `./gain`.

4. Run federated learning

You can run the `Gain`, `Global`, and `Local` settings in parallel using the bash script below.

```bash
cd ../../src || exit

gpu=0
dataset='rotated-cifar10'
num_client=20
partition='stratified'
subsample='clusterexp_2_2.0'
data_seed=0

divergence='contrastive'
divergence_seed=0
C=10.0
solver_seed=0

partition_config="client_${num_client}_partition_${partition}_subsample_${subsample}_seed_${data_seed}"
divergence_config="${divergence}_seed_${divergence_seed}"
structure_config_gain="${divergence_config}_soft_C_${C}_seed_${solver_seed}"

model='cnn'
algorithm='fedavg'
lr=0.01
seed=0

structures=("${structure_config_gain}" 'global' 'local')
shortcuts=('gain' 'global' 'local')

# ----------------- Parallel Training Loop -----------------
for i in {0..2}; do
  {
    CUDA_VISIBLE_DEVICES=${gpu} python main.py \
      --dataset "${dataset}" \
      --partition_config "${partition_config}" \
      --structure_config "${structures[i]}" \
      --model "${model}" \
      --algorithm "${algorithm}" \
      --gm_rounds 200 \
      --lm_opt sgd \
      --lm_lr ${lr} \
      --lm_epochs 1 \
      --seed ${seed} \
      --cuda \
      --history_path "../fed/${dataset}/${algorithm}_${shortcuts[i]}.pkl"
  } &
done
```

The training history will be saved to `./fed`. 

5. Output result (Acc, NGP, RSD)

```bash
python output.py \
  --history_path ../fed/rotated-cifar10/fedavg_gain.pkl \
  --ref_history_path ../fed/rotated-cifar10/fedavg_local.pkl \
  --ref
```

```bash
python output.py \
  --history_path ../fed/rotated-cifar10/fedavg_global.pkl
  --ref_history_path ../fed/rotated-cifar10/fedavg_local.pkl
  --ref
```

```bash
python output.py \
  --history_path ../fed/rotated-cifar10/fedavg_local.pkl
```

Three types of statistics will be printed.


## Citation

If you find this paper or repository helpful in your research, please consider giving a star ⭐️ and citing our paper:

```text
@inproceedings{zhang2026fedgain,
  author  = {Zhang, Yuqing and Zhou, Changli and Huang, Binghuang and Tian, Hui},
  title   = {FedGain: Toward Negative-Gain-Free Client Collaboration in Federated Learning},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning (ICML'26)},
  year    = {2026}
}
```
