# Graph-Counselor 缺陷与待修复问题清单

> 本文档记录在本地（WSL2/Linux）部署时发现的所有缺陷，按严重程度排序。

---

## 🔴 严重（会导致程序直接崩溃）

---

### ISSUE-01：缺少 `graph.json` 数据文件

**影响范围**：所有数据集，程序启动即崩溃  
**位置**：[code/run.py](Graph-Counselor/code/run.py) 第 57 行  

**问题描述**：  
`run.py` 会加载 `{data_path}/graph.json` 作为知识图谱结构数据，但当前所有数据集目录下只有 `data.json`，完全缺少 `graph.json`。

```python
# run.py
args.graph_dir = os.path.join(args.path, "graph.json")  # 此文件不存在
```

**缺失文件列表**：

```
data/processed_data/amazon/graph.json         ❌
data/processed_data/biomedical/graph.json     ❌
data/processed_data/dblp/graph.json           ❌
data/processed_data/goodreads/graph.json      ❌
data/processed_data/legal/graph.json          ❌
data/processed_data/maple/Biology/graph.json  ❌
data/processed_data/maple/Chemistry/graph.json ❌
data/processed_data/maple/Materials_Science/graph.json ❌
data/processed_data/maple/Medicine/graph.json ❌
data/processed_data/maple/Physics/graph.json  ❌
```

**解决方案**：  
从 README 提供的链接下载完整数据集：  
<https://drive.google.com/drive/folders/1DJIgRZ3G-TOf7h0-Xub5_sE4slBUEqy9?usp=share_link>

---

### ISSUE-02：缺少 Embedding 模型 `all-mpnet-base-v2`

**影响范围**：Retriever 初始化，影响所有运行模式  
**位置**：[code/tools/retriever.py](Graph-Counselor/code/tools/retriever.py) 第 43 行、第 156 行  

**问题描述**：  
`Retriever` 类硬编码了本地路径，但该路径下的模型尚未下载：

```python
# retriever.py - 硬编码路径，无法通过参数覆盖
self.model = sentence_transformers.SentenceTransformer('../model/all-mpnet-base-v2')
```

**解决方案**：

```bash
cd Graph-Counselor/model
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download sentence-transformers/all-mpnet-base-v2 \
    --local-dir ./all-mpnet-base-v2 \
    --local-dir-use-symlinks False
```

---

### ISSUE-03：`tools/` 目录缺少 `__init__.py`

**影响范围**：所有 Agent 文件，导致 `ImportError`  
**位置**：[code/tools/](Graph-Counselor/code/tools/)  

**问题描述**：  
所有 Agent 文件均使用包导入方式：

```python
from tools import graph_funcs, retriever  # 5 个 Agent 文件都有此导入
```

但 `tools/` 目录下没有 `__init__.py`，Python 无法将其识别为包。

**解决方案**：

```bash
touch Graph-Counselor/code/tools/__init__.py
```

---

### ISSUE-04：`run.py` 中模型路径白名单与本地路径不匹配

**影响范围**：本地部署时，`assert` 直接报错退出  
**位置**：[code/run.py](Graph-Counselor/code/run.py) 第 64-66 行  

**问题描述**：  
`run.py` 对 `--llm_version` 参数做了白名单校验，但白名单中的路径全部是原作者集群的绝对路径（`/cpfs02/user/lidong1/...` 和 `/nas/shared/...`），本地路径（`../model/Qwen2.5-7B-Instruct`）并不在其中：

```python
# 白名单中只有原作者集群路径，本地路径会触发 AssertionError
assert args.llm_version in [
    '/cpfs02/user/lidong1/model/Qwen2.5-7B-Instruct',   # 原作者路径
    '/cpfs02/user/lidong1/model/Qwen2.5-72B-Instruct',  # 原作者路径
    # ... 没有 '../model/Qwen2.5-7B-Instruct'
]
```

而脚本传入的是：

```bash
GPT_version=../model/Qwen2.5-72B-Instruct  # 不在白名单中 → AssertionError
```

**解决方案**：  
修改 `run.py` 第 64-66 行，在白名单中追加本地路径：

```python
assert args.llm_version in [
    # ... 原有列表 ...,
    '../model/Qwen2.5-7B-Instruct',
    '../model/Qwen2.5-72B-Instruct',
    '../model/Qwen2.5-7B-Instruct-AWQ',
]
```

---

## 🟡 中等（影响功能，运行时报错）

---

### ISSUE-05：`retriever.py` 中 Embedding 模型路径硬编码，无法通过参数配置

**影响范围**：Retriever 模块  
**位置**：[code/tools/retriever.py](Graph-Counselor/code/tools/retriever.py) 第 43 行  

**问题描述**：  
`--embedder_name` 参数传入了 `sentence-transformers/all-mpnet-base-v2`，但代码忽略了该参数，直接硬编码本地路径：

```python
self.model_name = args.embedder_name  # 正确读取了参数
self.model = sentence_transformers.SentenceTransformer(
    '../model/all-mpnet-base-v2'       # ❌ 但这里没有使用 self.model_name
)
```

**解决方案**：  
将硬编码路径改为使用参数：

```python
self.model = sentence_transformers.SentenceTransformer(self.model_name)
```

---

### ISSUE-06：`GraphAgent_Plan_Reflect_vllm.py` 中 `enc2` 无条件加载 7B 模型

**影响范围**：使用 72B 模型时多余加载 7B tokenizer，浪费时间  
**位置**：[code/GraphAgent_Plan_Reflect_vllm.py](Graph-Counselor/code/GraphAgent_Plan_Reflect_vllm.py) 第 81 行  

**问题描述**：  
无论主模型是什么，都会无条件加载 `Qwen2.5-7B-Instruct` 的 tokenizer：

```python
# 该行无条件执行，与主模型版本无关
self.enc2 = AutoTokenizer.from_pretrained("../model/Qwen2.5-7B-Instruct", trust_remote_code=True)
```

若本地没有 7B 模型或模型路径不同，将报错。

---

### ISSUE-07：脚本中 `server2.log` / `server_test.log` 残留在 git 追踪中

**影响范围**：git 仓库  
**位置**：[scripts/](Graph-Counselor/scripts/)  

**问题描述**：  
日志文件已在 `.gitignore` 中标记忽略，但若曾经被 commit，则仍会被追踪。

**解决方案**：

```bash
git rm --cached Graph-Counselor/scripts/server2.log
git rm --cached Graph-Counselor/scripts/server_test.log
```

---

## 🟢 轻微（建议改进，不影响运行）

---

### ISSUE-08：`data/` 目录下存在 macOS 残留文件 `.DS_Store`

**位置**：`data/.DS_Store`、`data/processed_data/.DS_Store`  
**解决方案**：已在 `.gitignore` 中忽略，执行以下命令清理：

```bash
find . -name ".DS_Store" -delete
```

---

### ISSUE-09：`run.py` 中存在注释掉的调试代码

**位置**：[code/run.py](Graph-Counselor/code/run.py) 第 92 行  

```python
# contents = [contents[80]]  # 调试用，仅处理第80条数据
```

建议删除或改为命令行参数控制。

---

### ISSUE-10：`run_Graph-Counselor.sh` 中 Qwen2.5-72B 需要 4 张高显存 GPU

**位置**：[scripts/run_Graph-Counselor.sh](Graph-Counselor/scripts/run_Graph-Counselor.sh)  

**问题描述**：  
脚本默认配置要求：

- GPU 0：运行 7B 模型（需 ~16GB 显存）
- GPU 1,2,3,4：运行 72B 模型（需 ~160GB 显存，4×40GB）

消费级显卡（≤8GB）无法直接运行，需改用量化模型或云端 API。

---

## 修复优先级汇总

| 编号 | 严重程度 | 问题 | 是否已修复 |
|------|---------|------|---------|
| ISSUE-01 | 🔴 严重 | 缺少 `graph.json` | ❌ 待处理 |
| ISSUE-02 | 🔴 严重 | 缺少 `all-mpnet-base-v2` 模型 | ❌ 待处理 |
| ISSUE-03 | 🔴 严重 | 缺少 `tools/__init__.py` | ❌ 待处理 |
| ISSUE-04 | 🔴 严重 | `run.py` 模型路径白名单不含本地路径 | ❌ 待处理 |
| ISSUE-05 | 🟡 中等 | `retriever.py` 硬编码 Embedding 模型路径 | ❌ 待处理 |
| ISSUE-06 | 🟡 中等 | 无条件加载 7B tokenizer | ❌ 待处理 |
| ISSUE-07 | 🟡 中等 | 日志文件 git 追踪残留 | ✅ 已处理 |
| ISSUE-08 | 🟢 轻微 | `.DS_Store` 文件残留 | ✅ 已忽略 |
| ISSUE-09 | 🟢 轻微 | 调试代码残留 | ❌ 待处理 |
| ISSUE-10 | 🟢 轻微 | 显存需求过高 | ❌ 需适配 |
