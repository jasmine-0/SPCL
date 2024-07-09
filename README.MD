<!-- 服务器地址： 211.81.55.149, ssh连接 校外要挂天大VPN
用户名和密码都是姓名全拼

环境：tensorflow-gpu 2.0

运行：
python lifelong_train.py --imp_method PLL --gpu_id 3 --overlap_rate 1

参数：
imp_method： 方法
gpu_id： 选用哪块GPU
overlap_rate： 重叠率 [0，0.1，...，1] -->


# Parallel Lifelong Learning

## Dependencies

```
tensorflow-gpu 2.x
```


## Create Label Set and Time Line

```
python create_labelset_timeline.py --dataset CIFAR09 --batch_size 32 --labelset_num 1 --output_file CIFAR09.yml
python create_labelset_timeline.py --dataset CIFAR04 --batch_size 32 --labelset_num 1 --output_file CIFAR04.yml
python create_labelset_timeline.py --dataset CIFAR59 --batch_size 32 --labelset_num 1 --output_file CIFAR59.yml
python create_labelset_timeline.py --dataset CIFARTOY --task_num 2 --min_class 5 --max_class 5 --batch_size 32 --labelset_num 1 --output_file CIFARTOY.yml
python create_labelset_timeline.py --dataset CIFAR --task_num 20 --min_class 2 --max_class 15 --batch_size 32 --labelset_num 10 --timeline_num 10 --output_file CIFAR.yml
python create_labelset_timeline.py --dataset CUB
python create_labelset_timeline.py --dataset VD
```

This command will generate 10 different random label sets, each of which has 10 different time lines.
And we set 5 different random seeds.
Thus, the total run times are `10*10*5=500`.
Also, you can set your own labelset and timeline using your personal yaml file, the only thing you need to consider is the correct end time for each time. An example yaml of CIFAR-100 with 20 different tasks is 


## Toy experiments on CIFAR-10

```
python lifelong_train.py --imp_method AGA --gpu_id 3 --overlap_rate 1
```

## Main experiments

```
python lifelon_train.py --dataset CIFAR04
```

## Hyper-Parameters

|Dataset|PS-CIFAR-10|PS-CIFAR-100|PS-EMNIST|PS-Imagenet-TINY|
|--|--|--|--|--|
|Backbone|base|res18|mlp|res18|
|Batch Size|32|128|128|128|
|Learning Rate|0.002|0.0004|0.003|0.0005|
|Memory Size (Per class)|30|30|5|15|
|Task Num|2|20|5|20|

