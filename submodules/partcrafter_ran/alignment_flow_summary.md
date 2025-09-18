# GAPS对齐和指标计算流程总结

## 概述

修改后的评估流程现在按照以下步骤进行：
1. **GAPS对齐**: 先对单个网格进行GAPS对齐
2. **网格合并**: 将对齐后的网格合并
3. **指标计算**: 在合并后的对齐网格上计算CD、F-score和IoU

## 详细流程

### Step 1: GAPS对齐 (align_meshes_with_gaps)

```python
# 对每个GT对象，找到最佳匹配的预测对象并进行GAPS对齐
aligned_gt_scene, aligned_pred_scene = align_meshes_with_gaps(gt_meshes, pred_meshes)
```

**过程**:
1. 对每个GT网格，计算与所有预测网格的Chamfer距离
2. 找到最佳匹配的预测网格
3. 使用GAPS工具对匹配的网格对进行对齐
4. 返回对齐后的GT场景和预测场景

### Step 2: 网格合并

```python
# 从对齐场景中提取网格
aligned_gt_meshes = list(aligned_gt_scene.geometry.values())
aligned_pred_meshes = list(aligned_pred_scene.geometry.values())

# 过滤空网格
aligned_gt_meshes = [mesh for mesh in aligned_gt_meshes if mesh is not None and len(mesh.vertices) > 0]
aligned_pred_meshes = [mesh for mesh in aligned_pred_meshes if mesh is not None and len(mesh.vertices) > 0]

# 合并网格
if len(aligned_gt_meshes) > 1:
    aligned_gt_merged = trimesh.util.concatenate(aligned_gt_meshes)
else:
    aligned_gt_merged = aligned_gt_meshes[0] if aligned_gt_meshes else trimesh.Trimesh()

if len(aligned_pred_meshes) > 1:
    aligned_pred_merged = trimesh.util.concatenate(aligned_pred_meshes)
else:
    aligned_pred_merged = aligned_pred_meshes[0] if aligned_pred_meshes else trimesh.Trimesh()
```

### Step 3: 指标计算

```python
# 在对齐后的合并网格上计算指标
cd = compute_chamfer_distance(aligned_gt_merged, aligned_pred_merged, num_samples=num_samples)
fscore = compute_f_score(aligned_gt_merged, aligned_pred_merged, num_samples=num_samples, threshold=0.1)
iou = compute_IoU(aligned_gt_merged, aligned_pred_merged, num_grids=64, scale=2.0)
```

## 优势

1. **更准确的对齐**: 先对单个对象进行精确对齐，再进行合并
2. **更好的指标**: 在对齐后的网格上计算指标，结果更准确
3. **保持一致性**: 确保GT和预测网格在相同的位置和尺度下进行比较

## 输出文件

流程会生成以下文件：
- `aligned_gt_merged.glb`: 对齐后的合并GT网格
- `aligned_pred_merged.glb`: 对齐后的合并预测网格
- 各种指标值：Chamfer Distance、F-score、IoU等

## 使用示例

```python
from src.utils.eval_utils import compute_aligned_metrics

# 计算对齐后的指标
metrics = compute_aligned_metrics(gt_meshes, pred_meshes, num_samples=10000)

# 获取结果
cd = metrics['chamfer_distance']
fscore = metrics['f_score']
iou = metrics['iou']

# 获取对齐后的网格用于可视化
aligned_gt_scene = metrics['aligned_gt_scene']
aligned_pred_scene = metrics['aligned_pred_scene']
aligned_gt_merged = metrics['aligned_gt_merged']
aligned_pred_merged = metrics['aligned_pred_merged']
```

## 注意事项

1. **GAPS依赖**: 需要正确安装GAPS工具
2. **网格质量**: 输入网格应该是有效的3D网格
3. **内存使用**: 合并大型网格可能需要较多内存
4. **计算时间**: GAPS对齐可能需要较长时间，特别是对于复杂网格

## 错误处理

代码包含完善的错误处理：
- GAPS对齐失败时使用原始网格
- 网格合并失败时使用单个网格
- 指标计算失败时返回默认值

## 测试

可以使用 `test_alignment_flow.py` 脚本来测试整个流程：

```bash
python test_alignment_flow.py
```

这将验证：
1. GAPS对齐功能
2. 网格合并功能
3. 指标计算功能
4. 整个流程的完整性
