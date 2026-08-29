# 拼图助手后端（puzzle-backend）

拼图助手 App 的 Python 服务端：对拼图碎片做形状识别，与板面缺口匹配，返回 Top-N 候选位置与置信度。技术栈：FastAPI + OpenCV + NumPy，轻量文件存储，无数据库。

## 已实现内容

### 匹配流水线（M1~M4）

- **M1 盒面标定**（`puzzle/vision/calibration.py`）：根据片数与成品尺寸估算网格（行 × 列）；对盒面照片四角做透视矫正并切分格子。
- **M2 板面处理**（`puzzle/vision/board.py`）：
  - `segment_board`：背景距离 + Otsu 自适应阈值 + 形态学清理（对纹理背景更鲁棒）；
  - `align_board_to_grid`：把用户框选的拼图四边形矫正到网格；`adaptive_cell_px` 让小拼图以 192px/格（大拼图 64px/格）保留接收边细节；
  - `classify_cells` / `find_gaps`：判定已填格与缺口；
  - `extract_receiving_edges`：按扫描线取邻居极值像素，中值滤波平滑，保留榫/槽形状；
  - `alignment_quality`：返回对齐/分割诊断（`boundary_ratio`、`ambiguous_cells`、`note`），仅对“完全识别不出拼图块”或“极差”做拦截。
- **M3 碎片处理**（`puzzle/vision/piece.py`）：
  - `segment_piece`：**GrabCut 优先**（边框=背景、中心=前景，固定随机种子保证可复现），失败回退颜色距离分割；
  - `_refine_mask_edges`：轮廓沿法线吸附到 Canny 边缘，仅当仍通过形状校验时采用；
  - `split_sides`：minAreaRect 对角支撑点切分四边，签名锚定碎片自身坐标系（对平面旋转不变）；
  - `side_profile`：弦归一化的有符号距离序列；`validate_piece_shape` 拒绝“碎片+纹理/阴影”污染轮廓。
- **M4 缺口匹配**（`puzzle/solver/matcher.py`）：
  - 每个缺口按 4 个旋转方向比较接收边与碎片边（镜像处理）；
  - 缺口分数 = **最优方向 + 对明显差的其他方向惩罚**（`GAP_WORST_THRESHOLD`/`GAP_WORST_PENALTY`）；
  - **颜色邻居信号**（默认开启）：碎片核心颜色与候选缺口相邻已拼块颜色的归一化距离，用于打破形状重复的僵局（`PUZZLE_COLOR_WEIGHT`）；
  - 置信度：softmin（`PUZZLE_CONFIDENCE_TEMPERATURE`）；分数并列时返回最多 6 个候选；
  - **旋转不确定标记**：最优与次优旋转分数接近时返回 `rotation_confident=false`。

### API

| 方法与路径 | 用途 |
|---|---|
| `GET /health` | 存活检查 |
| `POST /api/projects` | 创建项目（盒面照片 + 片数 + 尺寸） |
| `PUT /api/projects/{id}/calibration` | 盒面四角标定，可覆盖 rows/cols |
| `PUT /api/projects/{id}/board` | 上传已拼部分（四角 + 可选 rows/cols），返回缺口与对齐诊断 |
| `POST /api/projects/{id}/locate` | 碎片定位：返回候选（row/col/score/confidence/rotation/region/rotation_confident） |
| `GET /api/projects/{id}/board/preview` | 板面缺口标注图 |
| `GET /api/projects/{id}/locate/preview` | 最近一次定位的候选标注图 |
| `GET /api/projects/{id}/operations` | 读取项目的可回放操作日志 |
| `PUT /api/projects/{id}/placed` | 记录用户已把某块碎片放入 (row, col) |
| `GET /api/projects/{id}` | 项目元数据 |

`locate` 每次从存储的板面照片用当前代码重算掩膜，并生成诊断图（`piece_mask_overlay.jpg`、`board_edges_debug.jpg`）与碎片照片（`piece_photo.jpg`），方便复盘。

每个项目的 `operations.jsonl` 按时间顺序记录 `create` / `calibrate` / `board` / `locate` / `placed` 五类操作，包含序号、时间戳、请求参数、响应摘要，以及每次上传的原始照片（保存在项目目录 `ops/` 下，文件名形如 `003_locate_piece_photo.jpg`），可用于自动化回放与回归测试。

### 测试与验证

- 62 个 pytest 测试（合成数据端到端、API、几何映射、形状校验、旋转不变性）。
- `scripts/verify_*.py`：用真实照片验证各阶段（`--help` 查看参数）。
- 环境变量：`PUZZLE_DATA_DIR`、`PUZZLE_MIN_SIDE_PX`、`PUZZLE_MIN_CELL_PX`、`PUZZLE_LOG_LEVEL`、`PUZZLE_COLOR_WEIGHT`、`PUZZLE_CONFIDENCE_TEMPERATURE`、`PUZZLE_ROTATION_BAND`、`PUZZLE_AMBIGUITY_BAND`、`PUZZLE_CONTINUITY_WEIGHT`。

## 运行

```bash
cd /Users/lxt/Documents/Code/puzzle-backend
.venv/bin/uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

真机联调必须 `--host 0.0.0.0`；Swagger 文档：`http://127.0.0.1:8000/docs`。

## 遗留问题

- **碎片照片透视畸变**：碎片照片存在斜拍时，签名四边长度不均衡（如 483/729/732/779），旋转方向判断不可靠。已通过 `rotation_confident` 诚实提示；彻底解决需要正上方拍摄或透视矫正。
- **板面分割质量**：浅色碎片在浅色背景上分割不完整，会产生假缺口（候选指向已拼格）；接缝对比度低时接收边形状信号弱，真缺口分数被压低。改善板面照片（补光、接缝清晰）是当前最大精度杠杆。
- **单方向缺口本质模糊**：板面填充率低时，多数缺口只有 1 个相邻面可匹配，形状+颜色信号仍可能区分不出；多拼几块再拍可显著提升。
- **盒面参考图未用于匹配**：板面已拼格与盒面参考格的颜色对齐实测不可靠（中位距离 ~80），参考图图案匹配尚未实现；修复盒面标定后可作为第二校验信号。
- **边缘连续性信号默认关闭**：`edge_continuity` 在低对比度照片上相关性不稳定，默认 `PUZZLE_CONTINUITY_WEIGHT=0`，需要更高质量的碎片/板面照片后再启用。
- **参数为经验值**：颜色权重、置信度温度、旋转带宽等基于现有真实案例调优，需要更多样本校准。

## 目录

- `main.py` — FastAPI 入口
- `puzzle/vision/` — 图像处理（calibration / board / piece）
- `puzzle/solver/` — 缺口匹配
- `puzzle/db/` — 基于文件的轻量存储
- `scripts/` — 真实照片阶段验证脚本
- `tests/` — pytest 测试套件
