## 一、整体结构框架

代码分为几个主要部分：

1. **导入依赖**：包括 PyTorch、torchvision、numpy、matplotlib 等常用库。
2. **工具函数 (Utils)**：

   * `set_seed`：设定随机种子，保证实验可复现。
   * `accuracy_from_logits`：从模型输出的 logits 计算准确率。
3. **模型定义 (Model)**：

   * `SmallCIFARConvNet`：一个简洁的 CNN，用于 CIFAR-10 分类。
   * 包含三个卷积块 (`block1, block2, block3`) 和一个全连接分类头 (`head`)。
4. **训练与验证循环 (Train/Eval Loops)**：

   * `train_one_epoch`：单轮训练，包含前向传播、损失计算、反向传播、梯度裁剪和优化器更新。
   * `evaluate`：在验证/测试集上评估模型，返回 loss 和准确率。
5. **主函数 (main)**：

   * 参数解析（`argparse`）
   * 数据增强、加载 CIFAR-10 数据集
   * 模型、优化器、学习率调度器初始化
   * 训练与验证循环，保存最佳模型
   * 绘制 Loss 和 Accuracy 曲线图

---

## 二、功能说明

该脚本的主要功能是 **在 CIFAR-10 数据集上训练一个卷积神经网络，并保存最佳模型及训练曲线**。
主要流程：

1. 读取参数，设定随机种子和设备（CPU/GPU）。
2. 加载 CIFAR-10 数据并进行数据增强。
3. 定义模型和优化器。
4. 训练多轮 epoch，记录训练/验证的 loss 和 accuracy。
5. 保存验证集上表现最好的模型。
6. 输出并保存训练过程中的损失和精度曲线。

---

## 三、各个参数作用

在 `main()` 中通过 `argparse` 定义的参数有：

* **`--data_dir`**
  默认 `./Train_CIFAR10/data`
  CIFAR-10 数据集存放的路径。

* **`--epochs`**
  默认 `30`
  最大训练轮数。

* **`--target_acc`**
  默认 `95`
  目标测试集准确率（百分数形式），达到后提前停止训练。

* **`--batch_size`**
  默认 `512`
  每个 batch 的样本数。

* **`--lr`**
  默认 `1e-3`
  学习率。

* **`--weight_decay`**
  默认 `5e-4`
  权重衰减系数（L2 正则化，用于防止过拟合）。

* **`--seed`**
  默认 `42`
  随机种子，用于保证结果可复现。

* **`--num_workers`**
  默认 `4`
  DataLoader 预取数据的工作线程数。

* **`--device`**
  默认 `cuda`（如果 GPU 可用），否则 `cpu`
  训练设备。

---

## 四、模型结构简析

`SmallCIFARConvNet` 的结构大致为：

* **Block1**: Conv(3→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool(2) → Dropout(0.2)
* **Block2**: Conv(64→128) → BN → ReLU → Conv(128→128) → BN → ReLU → MaxPool(2) → Dropout(0.3)
* **Block3**: Conv(128→256) → BN → ReLU → Conv(256→256) → BN → ReLU → MaxPool(2) → Dropout(0.4)
* **Head**: Flatten → Linear(4096→256) → ReLU → Dropout(0.5) → Linear(256→10)

特点：层数不深，采用批归一化 (BN) 和 Dropout 来提高泛化能力。

---

## 五、训练细节

* **损失函数**：交叉熵 (`F.cross_entropy`)。
* **优化器**：AdamW（比 Adam 多了权重衰减参数）。
* **学习率调度器**：CosineAnnealingLR，余弦退火策略。
* **梯度裁剪**：`clip_grad_norm_`，防止梯度爆炸。
* **早停条件**：如果验证集准确率达到 `target_acc`，则提前停止训练。
* **输出**：最佳模型权重（`.pt` 文件）、loss 曲线图、accuracy 曲线图。

