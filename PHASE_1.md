# Fundus DX 第一阶段交付

## 本阶段目标

基于 `lixiangcog/fundus-dx-ml` 的 ResNet18、FastAPI 与 React 框架，搭建一个可信的中文眼底 AI 筛查工作站底座。

参考 RetiWave-Mamba 的临床工作流与信息层级，但严格遵守现有模型边界：当前处理的是彩色眼底照片，而不是 OCT；当前提供四分类筛查，而不是病灶分割或临床诊断。

## 已完成

- 中文响应式工作站界面（桌面端与 390px 移动端）
- 患者姓名（可选）与自动病例编号
- JPG、PNG、WebP 拖放/选择上传，前后端 12 MB 限制
- 脱敏演示样例快速载入
- 真实 ResNet18 四分类推理
- AMD、白内障、糖尿病视网膜病变、正常的完整概率分布
- 高、中、低置信度分层提示与人工复核建议
- API 在线状态、API 往返耗时、模型推理耗时
- `/health` 健康检查与 `/model-info` 模型范围接口
- 清晰展示“科研教学演示、非医疗器械”的系统边界
- 页面内置第二、三阶段路线图

## 本地运行

后端：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

前端（已安装 Node.js 的环境）：

```powershell
cd frontend
npm install
npm run dev
```

打开 `http://127.0.0.1:5173/`。

## 服务器运行

服务器项目目录：`/data/user/hd66945/fundus-dx-ml`

服务通过 Slurm CPU 分区运行，前端构建产物由 FastAPI 同源托管：

```bash
cd /data/user/hd66945/fundus-dx-ml
mkdir -p logs
sbatch slurm/fundus-dx-serve.sbatch
```

查询作业节点：

```bash
squeue -u "$USER" -n fundus-dx -o '%.18i %.9P %.20j %.8T %.10M %.20R'
```

在本地建立 SSH 隧道时，将 `<计算节点>` 替换为作业所在节点：

```bash
ssh -p 50888 -L 8000:<计算节点>:8000 hd66945@dl01.hpcmaster.com
```

随后打开 `http://127.0.0.1:8000/`。

## 验证结果

- Python：`12 passed`
- 前端：ESLint 通过
- 前端：Vite 生产构建通过
- 浏览器：桌面端真实上传与推理通过
- 浏览器：390px 移动端无横向溢出
- 浏览器：控制台无 error/warning
- 合成样例端到端结果：白内障 52.9%，正确触发低置信度人工复核提示

测试环境中出现一条 Starlette `TestClient` 关于未来 `httpx2` 的弃用警告，不影响当前功能。

## 下一阶段建议：病灶可解释性

1. 将 ResNet18 最后一层卷积特征接入 Grad-CAM。
2. 新增 `/explain` 或在 `/predict?explain=true` 返回热力图。
3. 后端统一输出原图尺寸、热力图尺寸、归一化方式与目标类别。
4. 前端增加原图 / 热力图 / 叠加图三视图和透明度调节。
5. 增加图像质量控制：视野完整度、曝光、模糊度与非眼底图拒绝。
6. 为 Grad-CAM 增加形状、范围、目标类别和确定性测试。

完成第二阶段后，再进入结构化 PDF 报告与病例存档。
