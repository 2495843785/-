# 项目一：基于 NumPy 的图像编码器

本项目使用 NumPy 实现 OpenCLIP ViT-B-32 图像编码器的前向计算。输入经过预处理的图像，输出每张图像对应的 512 维归一化特征。模型参数由课程提供且保持固定，本项目不训练模型。

## 项目功能

将每张 224 × 224 RGB 图像切分为 49 个 32 × 32 图像块。
根据图像块、CLS token 和位置编码构造图像 token。
 实现 12 个 Transformer block，包括多头注意力、MLP 和残差连接。
 为每张图像生成形状为 `[B, 512]` 的 `float32` 特征。
 使用课程提供的参考结果进行分阶段数值检查。

## 项目结构

```text
project1-numpy-student/
├── image_encoder.py          # 学生补全的图像编码器
├── preprocess_image.py       # 图像缩放、中心裁剪和归一化
├── helpers.py                # LayerNorm、GELU、注意力及向量辅助函数
├── runtime.py                # 模型参数和图像加载
├── run.py                    # 命令行推理程序
├── check.py                  # 分阶段检查程序
├── config.json               # 模型配置
├── requirements.txt          # Python 依赖清单
├── assets/
│   ├── images/               # 课程提供的测试图片
│   └── reference_outputs.npz # 本地数值检查用参考结果
└── weights/
    └── image_encoder.npz     # 课程提供的视觉模型参数
```

压缩包还包含课堂示例、练习题和课件代码索引。运行程序时，请保持课程提供的数据和权重文件位于预期目录。若课程要求不重复提交较大的权重或参考数据文件，请不要将它们上传到提交仓库。

## 环境要求

- Python 3.9 或更高版本
- NumPy
- Pillow
- `imageio`：仅运行课堂图像示例时需要

在项目目录中安装依赖：

```bash
python -m pip install -r requirements.txt
```

图像编码器不需要 GPU、PyTorch、OpenCLIP 软件包或网络连接。

## 运行检查

请在项目目录中按阶段运行，若有错误可以较容易地定位最先失败的阶段：

```bash
python check.py --stage assets
python check.py --stage preprocess
python check.py --stage patches
python check.py --stage tokens
python check.py --stage block0
python check.py --stage last_tokens
python check.py --stage output
python check.py --stage batch
```

也可以一次运行全部检查：

```bash
python check.py --stage all
```

检查程序会验证数组形状、数值是否有限、数据类型是否为 `float32`，并将计算结果与课程提供的参考结果比较。参考数组只用于比对和调试，不能作为编码器输入。编码器还应能处理其他图像和不同的批量大小。

## 提取图像特征

处理课程提供的 10 张图片，并保存图片 ID 和特征：

```bash
python run.py --output output_tokens.npz
```

处理自己的图片：
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz``
生成的 `.npz` 文件包含 `ids` 和 `output_tokens` 两个数组。`output_tokens` 的形状为 `[图片数量, 512]`，数据类型为 `float32`。

查看结果文件中各数组的名称、形状和数据类型：

```bash
python -c "import numpy as np; d=np.load('output_tokens.npz'); print(d.files); [print(k, d[k].shape, d[k].dtype) for k in d.files]"
```

## 实现约定

- 图像块大小为 32；每个图像块按通道、高、宽顺序展平。
- 注意力和 MLP 权重的存储形状为 `[输出维, 输入维]`，计算时需要转置。
- 最终投影矩阵的形状已经是 `[768, 512]`，不要转置。
- 使用项目提供的归一化、GELU、注意力和单位向量辅助函数。
- 中间计算需保持 `float32`，以满足数值检查要求。

## 检查结果

课程提供的测试图片已通过分阶段检查，包括 `patches`、`tokens`、`block0`、`last_tokens`、`output` 和 `batch`。运行脚本成功生成了 10 张图片的特征，结果形状为 `(10, 512)`，数据类型为 `float32`。修改代码后应重新运行检查；不同代码版本或运行环境的结果可能不同。

## 提交说明

按课程要求提交补全后的 `image_encoder.py`、本 README 和实验报告。报告应说明任务与方法、运行环境和步骤、个人贡献、使用过的协助、结果证据、各阶段数值误差，以及分析与反思。
