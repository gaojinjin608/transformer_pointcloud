## 一种改进的点云分析Transformer
这个开源库是我们论文《一种改进的点云分析Transformer》的实现.

### 软件环境
python >= 3.7

pytorch >= 1.6

h5py

scikit-learn

以及 pointnet2_ops_lib

```shell script
pip install  pointnet2_ops_lib/.
```
你可以从以下位置获取pointnet2_ops_lib https://github.com/erikwijmans/Pointnet2_PyTorch https://github.com/WangYueFt/dgcnn and https://github.com/MenghaoGuo/PCT

### 数据集
模型的训练需要ModelNet40(http://modelnet.cs.princeton.edu/) 数据集。

### 训练和验证
```shell script
# train
python main.py --exp_name=train --num_points=1024 --use_sgd=True --batch_size 32 --epochs 250 --lr 0.0001

# test
python main.py --exp_name=test --num_points=1024 --use_sgd=True --eval=True --model_path=checkpoints/best/models/model.t7 --test_batch_size 8

```
