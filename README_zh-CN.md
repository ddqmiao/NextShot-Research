# NextShot

**面向人像摄影的、保留源区域身份的结构化构图目标预测**

[English](README.md) · [方法与代码](docs/METHOD.md) · [论文结果](docs/RESULTS.md) · [数据构建](docs/DATA.md) · [发布范围](docs/RELEASE.md)

NextShot 接收人像照片及准备好的区域观察，联合预测各区域在目标画幅中的位置、可见大小，以及目标画幅比例和人物全身／半身取景。人物和环境区域的身份保持不变，使每项预测能够对应到原图内容。

本仓库公开 **4B 主模型、监督损失、部分区域匹配算法及论文图表与结果**。当前为部分研究代码发布，不包含训练权重、完整训练调度或从任意照片直接运行的应用。

## 方法

![区域关系编码、多模态融合与取景条件化几何解码](assets/architecture.png)

*论文图 3：区域与画框锚点经过关系编码，与图像和文字观察共同进入 Qwen3-VL；预测的目标画幅和人物取景进一步约束区域几何。*

- **监督构建**：通过部分区域对应，将参考图几何映射到源区域身份；多个参考各自保留为完整方案。
- **关系与多模态融合**：显式建模区域—区域、区域—画框关系，结合冻结的 Qwen3-VL-4B 与语言层 query/value LoRA。
- **取景条件化解码**：先预测画幅和人物粗取景，再预测区域框、质心与可见面积。
- **美学辅助学习**：利用 PortraitCraft 的属性评分和理由训练共享表示；布局推理时移除辅助头。

## 论文主要结果

| 证据 | 报告结果 | 评价范围 |
| --- | --- | --- |
| 区域几何 | Box IoU 为 **0.6373**，NoChange 为 **0.5639** | 781 张具有有效参考几何的源图 |
| 相对 AR-8B 的布局偏好 | Qwen / Flash 为 **79.5% / 81.0%** | 固定 200 张测试源图，属于模型评价 |
| 指导重拍 | 人工偏好率 **78.0%** | 100 对照片、5 位评价者，390/500 次判断 |

数字来自论文，不是本次发布重新运行所得。[完整结果](docs/RESULTS.md)保留消融、评价分母及资源统计口径。完整模型并非每一项几何指标都最优。

## 数据与监督

![数据筛选、区域证据提取、部分对应与训练监督](assets/supervision.png)

*论文图 2：AesRecon-CT 提供配对几何监督，PortraitCraft-Aux 提供美学属性与理由监督。*

发布的区域匹配模块接收区域描述和**预先计算的外观／上下文相似度矩阵**，执行语义兼容检查、部分二分匹配、拒配和歧义判断。它不包含完整感知模型或一键数据重建流程。[数据说明](docs/DATA.md)列出处理步骤、字段和各实验集合的区别。

## 核心实现

| 论文模块 | 对应文件 |
| --- | --- |
| 区域关系、画框锚点、Qwen 融合、LoRA | [backbone.py](nextshot/backbone.py) |
| 取景条件化几何解码 | [decoder.py](nextshot/decoder.py)、[framing.py](nextshot/framing.py) |
| 主模型与美学辅助监督 | [model.py](nextshot/model.py)、[auxiliary.py](nextshot/auxiliary.py) |
| 完整参考选择及损失 | [geometry_loss.py](nextshot/geometry_loss.py)、[losses.py](nextshot/losses.py) |
| 只含源观察的输入组织 | [source.py](nextshot/source.py) |
| 区域对应与语义约束 | [matching.py](data_processing/matching.py)、[semantics.py](data_processing/semantics.py) |

更多说明见[方法与代码映射](docs/METHOD.md)和[输入输出协议](docs/MODEL_IO.md)。

## 安装与检查

打包环境使用 Python 3.10 或以上。已有 CPU 测试只需 PyTorch、NumPy、SciPy 和 Pillow，不下载模型，也不加载预训练骨干。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

仓库已补齐[模型结构配置](configs/nextshot_4b.json)，消除了原核心测试依赖缺失配置的问题。该配置只覆盖主模型构造，不构成完整训练配方。[验证说明](docs/VERIFICATION.md)区分本次检查和原实验运行环境。

## 范围与论文状态

输入需要准备好的区域及特征；输出是二维目标构图，不是生成照片、裁剪变换或相机运动参数。人物取景只到全身／半身，细粒度身体部位输出仍为 unknown。

本包不包含原始照片、完整数据清单、权重、训练／评价总控脚本或交互演示。所依据的匿名论文标注为 ICLR 2027 在审稿件，此处不作录用声明。引用、第三方来源和许可状态分别见[发布说明](docs/RELEASE.md)、[第三方致谢](THIRD_PARTY_NOTICES.md)和[权利说明](RIGHTS.md)。
