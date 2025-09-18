# sim(3)对齐和逐物体指标计算流程总结

## 概述

新的评估流程现在按照以下步骤进行：
1. **合并网格**: 分别合并GT和预测网格
2. **sim(3)对齐**: 使用GAPS将预测网格对齐到GT网格
3. **变换提取**: 从对齐结果中提取变换矩阵
4. **逐物体计算**: 对每个单独的物体计算CD和IoU

## 详细流程

### Step 1: 合并GT和预测网格

```python
# 分别合并GT和预测网格
if len(gt_meshes) > 1:
    gt_merged = trimesh.util.concatenate(gt_meshes)
else:
    gt_merged = gt_meshes[0]

if len(pred_meshes) > 1:
    pred_merged = trimesh.util.concatenate(pred_meshes)
else:
    pred_merged = pred_meshes[0]
```

### Step 2: sim(3) registration使用GAPS

```python
# 使用GAPS将预测网格对齐到GT网格
aligned_gt_merged, aligned_pred_merged = align_merged_meshes_with_gaps(gt_merged, pred_merged)
```

**过程**:
1. 将合并的GT和预测网格保存为PLY文件
2. 使用GAPS工具将预测网格对齐到GT网格
3. 返回GT网格（不变）和对齐后的预测网格

### Step 3: 提取和应用变换

```python
# 从对齐结果中提取变换并应用到单个预测网格
aligned_pred_meshes = extract_and_apply_transformation(pred_merged, aligned_pred_merged, pred_meshes)
```

**变换提取过程**:
1. 对原始和对齐的合并预测网格进行采样
2. 使用SVD方法计算旋转矩阵
3. 计算缩放因子和平移向量
4. 将sim(3)变换应用到每个单独的预测网格

### Step 4: 逐物体指标计算

```python
# 对每个GT物体，找到最佳匹配的预测物体并计算指标
for i, gt_mesh in enumerate(gt_meshes):
    best_cd = float('inf')
    best_fscore = 0.0
    best_iou = 0.0
    best_pred_idx = -1
    
    for j, pred_mesh in enumerate(aligned_pred_meshes):
        # 计算Chamfer距离
        cd = compute_chamfer_distance(gt_mesh, pred_mesh, num_samples=num_samples//len(gt_meshes))
        # 计算F-score
        fscore = compute_f_score(gt_mesh, pred_mesh, num_samples=num_samples//len(gt_meshes), threshold=0.1)
        # 计算IoU
        iou = compute_IoU(gt_mesh, pred_mesh, num_grids=32, scale=1.5)
        
        if cd < best_cd:
            best_cd = cd
            best_fscore = fscore
            best_iou = iou
            best_pred_idx = j
```

## 优势

1. **全局对齐**: 先将预测网格对齐到GT网格，确保整体一致性
2. **精确匹配**: 对每个物体进行精确的匹配和指标计算
3. **变换一致性**: 所有预测物体使用相同的变换，保持相对位置关系
4. **详细指标**: 提供每个物体的详细指标和场景级别的统计
5. **参考一致性**: GT网格作为参考保持不变，确保评估的一致性

## 输出结果

### 场景级别指标
- `chamfer_distance`: 平均Chamfer距离
- `f_score`: 平均F-score
- `iou`: 平均IoU
- `chamfer_distance_std`: Chamfer距离标准差
- `f_score_std`: F-score标准差
- `iou_std`: IoU标准差

### 逐物体指标
- `per_object_cds`: 每个物体的Chamfer距离
- `per_object_fscores`: 每个物体的F-score
- `per_object_ious`: 每个物体的IoU

### 对齐结果
- `aligned_gt_scene`: 对齐后的GT场景
- `aligned_pred_scene`: 对齐后的预测场景
- `aligned_gt_merged`: 对齐后的合并GT网格
- `aligned_pred_merged`: 对齐后的合并预测网格

## 使用示例

```python
from src.utils.eval_utils import compute_aligned_metrics

# 计算sim(3)对齐后的逐物体指标
metrics = compute_aligned_metrics(gt_meshes, pred_meshes, num_samples=10000)

# 获取场景级别指标
scene_cd = metrics['chamfer_distance']
scene_fscore = metrics['f_score']
scene_iou = metrics['iou']

# 获取逐物体指标
per_object_cds = metrics['per_object_cds']
per_object_fscores = metrics['per_object_fscores']
per_object_ious = metrics['per_object_ious']

# 获取对齐后的网格用于可视化
aligned_gt_scene = metrics['aligned_gt_scene']
aligned_pred_scene = metrics['aligned_pred_scene']
```

## 预期输出

```
GT scene contains 6 objects
Predicted scene contains 6 objects
Step 1: Merging GT and predicted meshes...
GT merged mesh: 15000 vertices, 30000 faces
Pred merged mesh: 12000 vertices, 24000 faces
Step 2: Applying sim(3) registration using GAPS (aligning predicted mesh to GT mesh)...
Successfully aligned predicted mesh to GT mesh
Step 3: Extracting transformation and applying to individual meshes...
Extracted transformation (pred to GT): scale=1.0234, translation=[0.123, 0.456, 0.789]
Applied transformation to 6 predicted meshes
Step 4: Computing per-object metrics...
GT object 0 matched with pred object 2: CD=0.023456, F-score=0.876543, IoU=0.789012
GT object 1 matched with pred object 0: CD=0.034567, F-score=0.765432, IoU=0.678901
...
Final scene metrics: CD=0.028901±0.012345, F-score=0.820987±0.098765, IoU=0.733456±0.123456
```

## 注意事项

1. **GAPS依赖**: 需要正确安装GAPS工具
2. **变换精度**: 变换提取的精度取决于网格质量和采样密度
3. **匹配策略**: 使用Chamfer距离作为匹配标准
4. **内存使用**: 合并大型网格可能需要较多内存
5. **计算时间**: sim(3)对齐可能需要较长时间

## 错误处理

代码包含完善的错误处理：
- GAPS对齐失败时使用原始网格
- 变换提取失败时使用简单平移
- 指标计算失败时返回默认值
- 匹配失败时标记为无效

## 测试

可以使用 `test_sim3_alignment.py` 脚本来测试整个流程：

```bash
python test_sim3_alignment.py
```

这将验证：
1. sim(3)对齐功能
2. 变换提取和应用
3. 逐物体指标计算
4. 整个流程的完整性
