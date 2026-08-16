# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在本仓库中编写代码时提供指导。

## 项目概述

**拼图助手应用**的 Python 后端。目标场景：1000 片的拼图，其中有大面积近似颜色区域，人工拼装困难。后端：

1. 拍摄拼图盒子封面照片，标定参考网格（M1）。
2. 拍摄已拼部分板面的照片，将其对齐到该网格，并找出所有缺口（M2）。
3. 拍摄单个碎片的照片，分割出碎片，并将其四条边编码为稠密的形状签名（M3）。
4. 将碎片与所有板面缺口进行匹配，评估全部 4 个旋转方向，返回带置信度的 Top-3 候选（M4）。

匹配刻意采用**纯形状**方案——不使用任何颜色/纹理特征，因为目标场景正是大面积近似颜色的区域。目前**没有碎片数据库，也没有基于参考图的颜色匹配**；碎片纯粹依靠边缘几何形状与板面"接收边"进行匹配。

本仓库仅包含 Python 服务；应用（移动端/客户端）是独立项目。

## 环境搭建

- 可用的解释器只有系统 Python **3.9.6**（`/usr/bin/python3`，arm64）。没有 brew/pyenv。
- 虚拟环境位于 `.venv/`（已被 git 忽略）。创建并安装全部依赖（运行 + 开发）的命令：

  `python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt`

## 常用命令

- 启动 API：`.venv/bin/uvicorn main:app --reload` → http://127.0.0.1:8000
- 运行测试：`.venv/bin/python -m pytest`
- 代码检查：`.venv/bin/ruff check .`
- 用真实照片验证流水线的某个阶段：`scripts/verify_*.py --help`（每个脚本接收图片路径、网格/角点参数，并把预览图写入 `--out` 目录）。

## 环境注意事项

- **Python 3.9 限制了依赖版本选择。** 许多新版本要求 ≥3.10（例如 fastapi ≥0.129、numpy ≥2.1）。pip 会静默回退到仍支持 3.9 的最新版本（目前为 fastapi 0.128.8、numpy 2.0.2、opencv-python 5.0.0.93）。添加依赖时，预期会有同样的回退——如果某个包已完全放弃 3.9，则会报 "no matching distribution"。
- 开发工具版本在 `requirements-dev.txt` 中**固定（pinned）**。请保持固定：初次搭建时网络不稳定曾让 pip 把 `pytest` 解析到 3.2.5（2017 年的旧版本），破坏了现代 pytest 配置。
- `opencv-python` 解析到 5.0.0.93 版本线（OpenCV 5 预发布版）。

## 架构

- `main.py` — FastAPI 入口；通过 `uvicorn main:app` 运行。日志挂在 `puzzle` logger 上（级别由 `PUZZLE_LOG_LEVEL` 控制，默认 INFO；DEBUG 级别下匹配器会为每个被评估的缺口输出一行日志）。
- `puzzle/vision/` — 图像处理（OpenCV）：
  - `calibration.py`（M1）— `estimate_grid`（根据片数与宽高比估算行/列数）、`warp_box_photo`（根据 4 个标注角点对盒子封面做透视矫正）、`slice_grid`、`draw_grid_overlay`。
  - `board.py`（M2）— `segment_board`（背景取图像边框像素的中位数，与背景颜色距离超过 `color_distance=40` 的像素视为拼图）、`align_board_to_grid`（把照片和掩膜 warp 到 `CELL_PX=64` 像素/格的固定网格）、`classify_cells`（覆盖率 ≥ `FILL_RATIO=0.5` 判定为已填）、`find_gaps`（与已填格相邻的空格）、`extract_receiving_edges`（在"邻居格 + 半个缺口"窗口内按扫描线取极值填充像素——保留凸入缺口的榫/槽形状）、`render_gap_preview`。
  - `piece.py`（M3）— `segment_piece`（取最大连通分量）、`piece_contour`、`split_sides`（以 `minAreaRect` 角点为切点分成四条边）、`resample_contour`（按弧长重采样为 `SIDE_SAMPLES=64` 个点）、`side_profile`（采样点到弦的有符号距离，除以弦长归一化 → 对平移/旋转/均匀缩放不变）、`build_signature`、`validate_piece_resolution`（拒绝最短边低于配置下限的碎片，`MIN_SIDE_PX` 默认 16px，可用环境变量 `PUZZLE_MIN_SIDE_PX` 覆盖）。
- `puzzle/solver/matcher.py`（M4）— 缺口匹配：
  - `edge_profile` 把缺口接收边编码为与碎片边完全相同的特征。
  - `edge_distance` 取 `min(direct, mirrored)` 平均绝对差——碎片边与其邻居的接收边互为镜像，因此两种朝向都会尝试。
  - `match_gap` 尝试全部 4 个旋转（循环位移边序：`(DIRECTIONS[direction] - k) % 4`），并对缺口各方向取平均。
  - `score_gaps` / `top_candidates` 对所有缺口排序，返回带 softmin 置信度（温度 0.1）的 Top-3。
- `puzzle/db/projects.py` — 轻量**基于文件**的存储（无数据库）：`data/projects/{project_id}/` 存放 `metadata.json` 以及上传/派生出的图片。数据目录默认为 `<repo>/data`，可通过 `PUZZLE_DATA_DIR` 覆盖。
- `scripts/` — `verify_calibration.py`、`verify_board.py`、`verify_piece.py`、`verify_locate.py`：用真实照片对各阶段进行手工验证。
- `tests/` — pytest 测试套件（42 个测试）：`test_smoke`、`test_calibration`、`test_board`、`test_piece`、`test_matcher`、`test_api`（通过 `TestClient` 做端到端测试）。测试用程序化方式合成碎片/板面（无图片 fixture）。

### API

| 方法与路径 | 前置条件 | 用途 |
|---|---|---|
| `GET /health` | — | 存活检查 |
| `POST /api/projects` | 盒子照片、片数、成品尺寸（cm） | 创建项目，估算网格 |
| `PUT /api/projects/{id}/calibration` | 项目存在 | 对盒子照片做 4 角点透视矫正，切分网格 |
| `PUT /api/projects/{id}/board` | 已标定 | 板面照片对齐网格、判定已填格、找出缺口；**源照片每格像素低于配置下限（默认 16px）时返回 422** |
| `POST /api/projects/{id}/locate` | 已标定且已上传板面 | 构建碎片签名、为各缺口打分、返回 Top-3 候选；**碎片最短边低于配置下限（默认 16px）时返回 422** |
| `GET /api/projects/{id}` | 项目存在 | 读取 `metadata.json` |

### 关键不变量（修改代码时请牢记）

- **尺度不变性是设计契约。** 每个特征都经过弦归一化并按弧长重采样为固定的 64 个点，板面也始终 warp 到固定的 64 px/格网格，因此照片的绝对尺寸会被抵消。分辨率守卫（`MIN_SIDE_PX`、`MIN_SOURCE_CELL_PX`，默认 16px）保护着榫/槽细节被量化丢失之下的保真度下限；可用环境变量 `PUZZLE_MIN_SIDE_PX` / `PUZZLE_MIN_CELL_PX` 调严（如 64px）。默认值刻意低于 `CELL_PX` / `SIDE_SAMPLES`（64），以兼容低分辨率拍摄（如模拟器），代价是匹配余量变小。
- **接收边必须携带邻居的榫/槽形状**，而不只是边界线；匹配器依赖这一点（`test_real_chain_ranks_true_gap_first` 是对此的回归测试）。
- **缺口坐标是 1-based**（API 与 `find_gaps` 输出中的行/列）；vision 代码内部会转换为 0-based。
- 在没有重新审视项目概述之前，不要给匹配器引入颜色/纹理特征——纯形状方案是刻意为之。
