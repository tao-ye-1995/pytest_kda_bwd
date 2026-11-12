# KDA backward 独立精度测试

项目位于 `/workspace/pytest_kda_bwd`。CPU、AscendC、FLA 各自放在独立文件中。
每个数据用例只生成一次 PyTorch CPU FP64 真值和前向缓存，所有设备 backward 使用同一份缓存，经 torch 转换布局与数值精度后比较七个梯度。

## 开始使用

CPU 测试需要 pytest、PyTorch。设备测试还需要 torch_npu、Triton Ascend、fla_npu，以及本地 FLA 仓。
以下命令在同一个 Bash 终端执行：

```bash
cd /workspace/pytest_kda_bwd

# CPU 数学公式自检：4 项，不使用 NPU
python -m pytest -q --impl cpu --case reference

# 设备环境：当前验证版本为 CANN 9.1.0 / BiSheng 1.2.0
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PATH="$ASCEND_HOME_PATH/tools/bishengir/bin:$PATH"

# 先测试全部边界用例和 CPU 数学自检：25 项
python -m pytest -q --impl all --case edge,reference --device 4

# 再测试全部真实形状、边界用例和 CPU 数学自检：73 项
python -m pytest -q --impl all --case all --device 4
```

`--device 4` 指逻辑 NPU 编号，可以改成要使用的设备。CPU 默认使用 4 个线程，可用 `--cpu-threads 1` 调整。
完整真实形状需要较多 CPU 内存和计算时间；不会缩短或截断 token。

## 怎样选择要测的实现和数据

两个参数相互独立：**`--impl` 选择实现，`--case` 选择数据**。
例如 `--impl fla --case 6` 表示“用 FLA backward 测第 6 组真实形状”，只运行 1 项；CPU 仍会生成这项的真值和缓存。

先列出可选实现、精确用例名、token 数、序列数和输入模式：

```bash
python -m pytest --list-cases
```

`--impl` 可填 `cpu`、`ascendc`、`fla`，多个实现用逗号分隔；`all` 表示目录中的所有实现，默认值为 `all`。

`--case` 的取值如下，多个取值也用逗号分隔：

| 取值 | 选择的数据 |
| --- | --- |
| `6` 或 `shape_demo_6` | shape_demo 第 6 行，T=131042，19 段；两种写法都精确选择这一组 |
| `1,6` | 第 1、6 行两组真实形状 |
| `shape` | 全部 16 组真实形状 |
| `edge` | 全部 7 组边界数据 |
| `edge_zero_do` | 只测零输出梯度；其他边界名称见 `--list-cases` |
| `reference` | 4 项短序列 CPU 公式自检，须选中 `cpu`，不使用真实形状 |
| `edge,reference` | 边界数据加 CPU 公式自检 |
| `all` | 所有真实形状、边界数据；选中 CPU 时再加 4 项公式自检，默认值 |

**数字编号从 1 开始，对应 [shape_demo.txt](shape_demo.txt) 的行号。** 该文件保留了原始 shape_demo 的全部 16 行累计边界。
选择 `1` 只会测第 1 行。每个“实现 × 数据”组合算 1 项；CPU 公式自检额外算 4 项。

可直接复制的例子如下，设备命令需先准备上面的 CANN 环境：

| 想测什么 | 完整命令 | 项数 |
| --- | --- | --- |
| 第 6 组的 FLA | `python -m pytest -q --impl fla --case 6 --device 4` | 1 |
| 第 6 组的两个设备实现 | `python -m pytest -q --impl ascendc,fla --case 6 --device 4` | 2 |
| 第 1、6 组的三个版本 | `python -m pytest -q --impl all --case 1,6 --device 4` | 6 |
| FLA 的全部数据 | `python -m pytest -q --impl fla --case all --device 4` | 23 |
| 三个版本的全部真实形状 | `python -m pytest -q --impl all --case shape --device 4` | 48 |
| 三个版本的全部边界数据 | `python -m pytest -q --impl all --case edge --device 4` | 21 |
| CPU 全部数据和公式自检 | `python -m pytest -q --impl cpu --case all` | 27 |

项数按默认三个实现和 16 行 shape 文件计算。添加实现后，`all` 的项数随之增加。
运行前也可只收集目标用例，确认选择结果，不做计算：

```bash
python -m pytest --collect-only -q --impl fla --case 6
```

## 项目结构与新实现

```text
pytest_kda_bwd/
├── README.md                 使用说明
├── pytest.ini                pytest 配置
├── conftest.py               --impl / --case 等参数和精确选择
├── cases.py                  用例定义与固定随机输入
├── shape_demo.txt            原始 16 组累计边界
├── test_kda_bwd.py            共享缓存、精度断言、报告、CPU 自检
└── implementations/
    ├── __init__.py           共享设备缓存的一次 dtype 转换
    ├── cpu.py                PyTorch FP64 数学前向、autograd 真值
    ├── ascendc.py            AscendC backward 及 torch 布局转换
    └── fla.py                直接 import 本地 FLA backward
```

AscendC 调用安装的 `fla_npu.ops.ascendc.npu_chunk_kda_bwd`。
FLA 直接从 `--fla-root` 指定的仓 import `fla.ops.kda.chunk_bwd.chunk_kda_bwd`，并校验派发后端为 `triton_ascend`；
默认仓为 `/workspace/flash-linear-attention`。两个适配器都不执行设备前向、不重算前向缓存。

接入新设备实现时，在 `implementations/my_impl.py` 定义 `backward(batch)`，再运行：

```bash
python -m pytest -q --impl my_impl --case edge --device 4
python -m pytest -q --impl my_impl --case all --device 4
```

实现文件会自动出现在可选列表里，无需修改注册表。只有选中的设备实现才会被导入。
`implementations/` 中每个独立 `.py` 文件应代表一个实现；通用辅助函数放在 `__init__.py`。

`batch` 包含相同的共享设备缓存和选项，契约如下：

- 张量：`q/k/v/beta/raw_beta/g/gk/A_log/dt_bias/q_rstd/k_rstd/Aqk/Akk/w/u/qg/kg/v_new/h/grad_output`。
- 选项：`scale/chunk_size/packed/normalize/fused_beta/fla_root`。
- metadata：host list `cu_seqlens/chunk_indices` 和 NPU int64 `cu_seqlens_device/chunk_indices_device`；设备 chunk 索引 `[NC,2]`，dense 时均为 None。

启用 normalize 时，q/k 是归一化缓存；关闭时为直接输入。beta 是有效 beta，raw_beta 是原始值。
g 是原始 gate 输入，gk 是块内累计自然 gate 除以 ln(2)；host chunk_indices 按 `(sequence, chunk)` 扁平存储。
共享 token 张量为 `[B,T,H,D]`，beta/rstd 为 `[B,T,H]`，Aqk/Akk 为 `[B,T,H,64]`；
h 在 packed 下为 `[1,total_chunks,H,128,128]`，dense 下为 `[B,T/64,H,128,128]`。
矩阵缓存和 dO 为 BF16，gate、beta、参数、归一化统计量为 FP32。

返回原始输入梯度 `(dq,dk,dv,dbeta,dg,dA,dbias)`：前三项 BF16，后四项 FP32；
token 梯度 `[B,T,H,D]`，dbeta `[B,T,H]`，dA `[H]`，dbias `[H,128]`。
须补齐 L2norm/beta sigmoid 等链式梯度，不得修改共享输入或另行执行前向。

## 真值、阈值和报告

[cpu.py](implementations/cpu.py) 用纯 PyTorch FP64 递推的等价分块公式、三角求解和 autograd 产生真值；
不调用 AscendC 或 FLA。各段初始状态为零，H=4、K=V=128、chunk_size=64。
CPU 自检对比分块与逐 token 公式的输出、梯度，并做 finite-difference gradcheck。
CPU 参数项检查数学基准；设备精度由各设备实现与 CPU 真值的比较项验证。

每个非零参考梯度同时要求相对 L2 误差 ≤ 3%、相对参考峰值的最大绝对误差 ≤ 8%。
数学零参考用绝对容差 1e-7；零 dO 要求七个梯度严格全零。这些是当前 BF16 数据场景的工程阈值。

每次运行自动创建独立的 `results/cpu/run_…/` 或 `results/device_N/run_…/`，
生成 `pytest_kda_math_metrics.json`，并在终端显示实际路径。
也可指定目录并额外生成 JUnit XML：

```bash
python -m pytest -q --impl all --case edge,reference --device 4 \
  --report-dir results/smoke --junitxml=results/smoke/pytest.xml
```

JSON 保存实际累计边界、七个梯度误差、通过状态、CANN/实际编译器路径和版本、设备型号及缓存目录。
`failed` 时查看 traceback 或 JSON 的 `error/metrics`；`skipped` 不代表精度通过。
零参考的相对误差为 `null`，查看 `max_abs/zero_atol`。单独的 `reference` 自检或只收集用例不生成误差 JSON。

pytest 自动将选定 CANN 的 BiSheng 放在 PATH 前面并核验 Triton 实际解析出的路径。
未设置 `TRITON_CACHE_DIR` 时自动建立新缓存；只有确认 CANN、编译器和设备型号一致时才复用已有缓存。
环境升级后重新选择 CANN 并使用新缓存；加 `-s` 可以显示设备执行前的环境核验输出。

## 验证记录

2026-11-12，Ascend A3 逻辑设备 4，CANN 9.1.0 / BiSheng 1.2.0：
完整 **73 passed**，全部 16 组真实累计边界与输入文件逐项一致，耗时 7 分 37 秒。
AscendC 最大相对 L2 为 1.0919%，FLA 为 1.2720%；七个零 dO 梯度均严格为零。
另外验证了精确选择、自动报告目录、新编译缓存和新实现文件的自动接入；临时错误实现的七个梯度均被精度检查拒绝。

完整结果：[误差 JSON](results/full/pytest_kda_math_metrics.json)、
[JUnit XML](results/full/pytest.xml)、[运行日志](results/full/pytest.log)。
