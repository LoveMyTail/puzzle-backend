# 测试数据集格式

用于批量评估定位算法的标准数据集布局。把数据集放到 `data/test_sets/` 下，用 `scripts/evaluate_dataset.py --test-sets data/test_sets` 一键跑完整流程。

## 目录结构

```
data/test_sets/
  manifest.csv
  set_01/
    board.jpg        # 已拼部分照片（拼图现状）
    piece.jpg        # 单块碎片照片
  set_02/
    board.jpg
    piece.jpg
  ...
```

## manifest.csv 字段

```csv
name,board,piece,rows,cols,corners,expected_row,expected_col
set_01,set_01/board.jpg,set_01/piece.jpg,6,8,"[[0,0],[2731,0],[2731,2048],[0,2048]]",4,7
set_02,set_02/board.jpg,set_02/piece.jpg,6,8,,5,3
```

- `name`：用例名（也作为临时项目名）。
- `board` / `piece`：相对 `data/test_sets/` 的照片路径。
- `rows` / `cols`：拼图网格行列数（**必填**，来自包装盒或工作台）。
- `corners`：板面照片中“整个拼图”四个角的 JSON 数组 `[[x,y],...]`（左上→右上→右下→左下，照片像素坐标）。**留空时使用后端自动检测四边形**（检测不可靠时结果会差，建议尽量提供）。
- `expected_row` / `expected_col`：该碎片的正确位置（**强烈建议提供**，用于计算 hit@1/3/6 与真值排名、置信度）；不知道可以留空。

## 运行

```bash
cd /Users/lxt/Documents/Code/puzzle-backend
.venv/bin/python scripts/evaluate_dataset.py --test-sets data/test_sets
```

脚本会在 `data/test_runs/` 下为每个用例创建临时项目（不污染正式数据），跑完整定位链路（分割 → 缺口接收边 → 形状匹配 → 颜色邻居 → 候选排序），输出：

```
name      candidates                          truth   rank  confidence
set_01    (4,7),(3,7),(4,6)                   (4,7)   1     0.72
set_02    ...

hit@1: 2/5 (40%)
hit@3: 4/5 (80%)
```

## 备注

- 板面照片要求：包含**整个拼图**（所有行列），正上方拍摄，接缝清晰；碎片照片要求：正上方、无阴影、碎片占画面 1/2 以上。
- `corners` 缺失时自动检测仅在拼图块与背景对比明显时可靠；有条件的用例请手动标注四角。
- 每张 `board.jpg`/`piece.jpg` 就是一次独立测试，可以重复组合（同一板面配不同碎片、同一碎片配不同板面）。
