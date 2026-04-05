# Research 阶段交付文档

## 1. 项目整体架构与技术栈说明

### 1.1 项目定位
本仓库是基于 Ultralytics 体系深度改造的 YOLO-Master（YOLO-MoE）工程，核心价值是将 Mixture-of-Experts (MoE) 与 YOLO 主干/颈部模块深度融合，实现按输入复杂度动态分配计算（compute-on-demand），在实时检测场景中取得更优精度-时延平衡。

项目同时覆盖多任务能力：检测（detect）、分割（segment）、分类（classify）、姿态（pose）、旋转框（obb），并提供训练、验证、推理、导出、跟踪、benchmark 的统一接口与 CLI 入口。

### 1.2 总体分层架构
- 应用层
  - `app.py`：Gradio WebUI 推理与模型管理示例（模型扫描、热切换、推理结果展示）。
  - `ultralytics/cfg/__init__.py`：CLI 解析、任务/模式映射、参数校验与配置聚合。
- 引擎层（核心流程编排）
  - `ultralytics/engine/model.py`：统一 Model API（`train/val/predict/export/track/benchmark`）。
  - `ultralytics/engine/trainer.py`：训练调度、DDP、AMP、优化器/学习率、回调、检查点。
  - `ultralytics/engine/exporter.py`：多格式导出（ONNX/TensorRT/OpenVINO 等）。
  - `ultralytics/engine/results.py`：推理结果对象与数据导出（DataFrame/JSON/CSV）。
- 模型与算子层
  - `ultralytics/nn/tasks.py`：模型构图、任务头、损失与前向主流程。
  - `ultralytics/nn/modules/moe/*`：MoE 路由器、专家、损失、分析、剪枝。
- 数据与工具层
  - `ultralytics/data/*`：数据集加载、校验、增强、dataloader。
  - `ultralytics/utils/*`：日志、检查、设备、分布式、可视化、配置、导出工具。
- 质量与文档层
  - `tests/*`：单测/集成测试。
  - `docs/*`：文档与站点构建。
  - `docker/*`：多平台容器构建（GPU/CPU/ARM/Jetson/Python/Conda）。

### 1.3 技术栈与依赖
- 语言与运行时
  - Python >= 3.8
- 深度学习框架
  - PyTorch（训练/推理主框架）
  - torchvision
- 核心依赖（来自 `pyproject.toml` / `requirements.txt`）
  - numpy, scipy, opencv-python, pillow, pyyaml, matplotlib
  - psutil, polars, ultralytics-thop
  - requests
- 可选扩展
  - 导出链路：onnx, onnxslim, openvino, tensorflow, coremltools 等
  - LoRA：peft
  - 记录与可视化：wandb, tensorboard, mlflow
- 工程标准工具
  - pytest, coverage
  - ruff / isort / yapf / docformatter / codespell（配置已在 `pyproject.toml`）

---

## 2. 核心模块功能与业务逻辑拆解

### 2.1 统一任务入口与执行闭环
- Python API 入口
  - `from ultralytics import YOLO`，通过 `Model` 类统一封装训练、验证、推理、导出、跟踪。
- CLI 入口
  - `yolo TASK MODE ARGS`（如 `yolo train ...`, `yolo predict ...`, `yolo export ...`）。
  - TASK 支持：`detect/segment/classify/pose/obb`。
  - MODE 支持：`train/val/predict/export/track/benchmark`。

### 2.2 训练主链路（engine.trainer）
- 配置解析：`get_cfg` 合并默认配置与 overrides。
- 设备/分布式：自动识别 CPU/GPU、多卡 DDP 启动与清理。
- 稳定性策略：AMP 检查、EMA、EarlyStopping、NaN 恢复逻辑。
- 优化策略：线性/余弦 LR、冻结层机制、自动批大小。
- 可扩展能力
  - LoRA：训练前应用 `apply_lora`。
  - 回调体系：支持多事件注入。

### 2.3 MoE 主体能力（nn.modules.moe）
- 模块化 MoE 体系
  - 路由器：`UltraEfficientRouter` / `EfficientSpatialRouter` / `AdaptiveRoutingLayer` 等。
  - 专家：`SimpleExpert` / `GhostExpert` / `InvertedResidualExpert` 等。
  - MoE 形态：`UltraOptimizedMoE`, `ModularRouterExpertMoE`, `ES_MOE` 等。
- 训练稳定性
  - 辅助损失：Load Balancing + Z-Loss（`loss.py`），抑制路由坍塌与 logit 爆炸。
  - Shared Expert 分支提升收敛稳定性。
- 部署导向优化
  - Top-K 稀疏计算、批处理专家并行、GroupNorm 方案、可配置条件计算阈值。

### 2.4 推理结果对象与下游数据消费（engine.results）
- `Results` 结构统一封装检测框、分割掩码、关键点、分类概率等。
- 支持跨设备迁移（CPU/CUDA）与结构化导出（DataFrame/JSON/CSV）。
- 该设计有利于视觉结果直接流向外部机器人管线（例如 ROI、轨迹、统计）。

### 2.5 导出与部署（engine.exporter）
- 格式覆盖：PyTorch/TorchScript/ONNX/OpenVINO/TensorRT/CoreML/TFLite/NCNN/RKNN 等。
- 参数校验：不同导出格式绑定不同可用参数（dynamic/int8/half/opset 等）。
- 对边缘部署友好：支持 INT8 导出链路与 TensorRT engine 路径。

---

## 3. 现有接口、数据结构与依赖关系说明

### 3.1 外部接口（你可直接调用的能力）
- Python API
  - `YOLO(model).train(...)`
  - `YOLO(model).val(...)`
  - `YOLO(model).predict(...)` 或 `YOLO(model)(...)`
  - `YOLO(model).export(format="onnx"|"engine"|...)`
  - `YOLO(model).track(...)`
- CLI
  - `yolo train/val/predict/export/...`
  - 参数风格：`arg=value`，由配置系统统一校验。

### 3.2 关键数据结构
- 配置结构
  - `get_cfg()` 返回 `SimpleNamespace`，聚合训练/推理/导出参数。
- 推理结果结构
  - `Results`：
    - `boxes`：检测框与类别置信度
    - `masks`：分割掩码（与图像尺寸关联）
    - `keypoints`、`probs`、`obb`
    - `speed`：预处理/推理/后处理耗时
- MoE 训练状态
  - 模块内维护辅助损失统计（aux/balance/z-loss），用于训练监控与诊断。

### 3.3 模块依赖关系（主干）
- `ultralytics/__init__.py`
  - 对外暴露 `YOLO` 等模型类并延迟导入。
- `engine/model.py`
  - 依赖 `nn/tasks.py` 完成模型构建与任务推断。
  - 依赖 `engine/trainer.py` / `engine/exporter.py` 完成训练与导出。
- `engine/trainer.py`
  - 依赖 `data/*` 数据集校验与加载、`utils/*` 分布式/优化工具。
  - 可注入 LoRA 与 MoE 相关能力。
- `nn/tasks.py`
  - 依赖 `nn/modules/*`，包含 MoE 模块注册。

### 3.4 与你当前“苹果采摘机器人分割任务”的映射
- 已具备
  - 二分类语义分割能力可通过 `segment` 任务配置落地。
  - 具备实时推理、导出 ONNX/TensorRT、INT8 路径和 Jetson 友好生态。
  - 推理结果可输出 mask，并可派生轮廓/外接矩形。
- 需定制/补充
  - 任务标签体系需要明确为“果实/背景”二分类。
  - 需建立果园专用数据集、增强策略、评估脚本（mIoU/边界误差/小目标完整度）。
  - 性能参数优先聚焦：分割质量、推理实时性、复杂环境下鲁棒性（光照/遮挡/密集重叠/复杂背景）。
  - 训练过程中可优先复用项目原生能力（如 LoRA、MoE 相关训练配置）进行高效微调与迭代。

---

## 4. 项目开发规范与运行环境要求

### 4.1 开发规范
- 文档与注释
  - 倡导 Google-style docstring。
- 代码质量
  - 使用 ruff/isort/yapf/docformatter 等统一风格。
- 测试要求
  - pytest 为主，慢测需显式 `--slow`。
  - 贡献需保证 CI 通过（单测、格式、质量检查）。
- 变更原则
  - 小步、聚焦、避免重复、避免破坏兼容性。

### 4.2 运行环境要求
- 基础环境
  - Python >= 3.8
  - PyTorch 与 CUDA 版本需匹配硬件
- 安装方式
  - `pip install -e .`（开发）或 pip/conda/docker。
- 容器环境
  - 提供 GPU/CPU/ARM/Jetson 等多 Dockerfile，适配训练和部署。
- 导出环境
  - ONNX/TensorRT/OpenVINO 等依赖按导出目标安装 `ultralytics[export]` 或等价依赖集合。

### 4.3 许可证约束
- 当前仓库遵循 AGPL-3.0。
- 衍生工程若复用核心代码，需满足 AGPL 合规要求（除非使用企业授权）。

---

## 5. 我对项目的完整理解总结与认知对齐说明

### 5.1 我当前的对齐结论
1. 该仓库本质是“多任务 YOLO 工程基座 + MoE 加速增强层”，已经具备从训练到部署的完整工具链。
2. 对你的苹果采摘机器人任务，最适配路径是：基于 `segment` 任务做二分类语义分割专项定制，并引入你定义的鲁棒性与边界精度评估体系。
3. 你提出的需求不是“从零做框架”，而是“在现有 YOLO-MoE 工程上做场景化、指标化、部署化落地”。
4. 该仓库在导出与边缘部署方面基础良好（ONNX/TensorRT/INT8）；本阶段以分割质量、实时性、复杂环境鲁棒性为核心目标推进，不将 ROS 端到端链路与 RGB-D 像素级严格对齐作为当前约束。

### 5.2 针对你需求的关键约束理解（已吸收）
- 视觉输入：与 D435i 深度同步的 1280x720 RGB 连续帧。
- 输出定义：与输入同分辨率、同坐标系的二值 mask；前景仅苹果果实（含果蒂不含果柄）。
- 性能目标：
  - 全场景 mIoU、召回、精确率、像素准确率均有高阈值硬约束。
  - 极端光照、遮挡、密集重叠、复杂背景下需保持鲁棒。
  - 边界偏差、边缘精度、粘连切分是关键验收项。
- 部署目标：Jetson Orin NX、30FPS、INT8 精度退化受限、支持全局/ROI 双模式。
- 训练迭代目标：小样本迁移学习、在线增量学习、难例挖掘与重训闭环。
- 训练策略补充：可使用项目原生 LoRA 参数高效微调能力，降低训练开销并加快场景迭代。

### 5.3 当前阶段边界声明（严格遵循你的流程）
- 本次仅完成 Research 阶段，不进入 Planning 与 Implementation。
- 待你完成人工校验并确认“认知无偏差”后，我再进入下一阶段并输出 `plan.md` + Todo List。
