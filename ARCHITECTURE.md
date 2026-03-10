# Graph-Counselor 项目架构文档

> ACL 2025 Main Conference Paper  
> 论文：[arXiv:2506.03939](https://arxiv.org/pdf/2506.03939)

---

## 目录

1. [项目概述](#1-项目概述)
2. [目录结构](#2-目录结构)
3. [整体架构](#3-整体架构)
4. [核心模块详解](#4-核心模块详解)
5. [Agent 类型与选择逻辑](#5-agent-类型与选择逻辑)
6. [数据流](#6-数据流)
7. [支持的模型与接入方式](#7-支持的模型与接入方式)
8. [支持的数据集](#8-支持的数据集)
9. [关键运行参数](#9-关键运行参数)

---

## 1. 项目概述

Graph-Counselor 是一个基于**多智能体协同**的图知识推理框架，核心思想是：

- **Planning Agent**：对复杂问题进行分解与规划
- **Execution Agent**：在知识图谱上逐步执行探索动作（ReAct 模式）
- **Reflection Agent**：对执行结果进行多视角自我反思与修正

三类 Agent 协同工作，完成对知识图谱的自适应探索和问答推理。

---

## 2. 目录结构

```
Graph-Counselor/                    ← 仓库根目录
├── README.md
├── environment.yaml                ← conda 完整依赖快照
├── .gitignore
├── eval_Qwen.py                    ← Qwen 系列模型评估脚本
├── eval_Llama.py                   ← Llama 系列模型评估脚本
├── eval.sh                         ← 批量评估入口
├── assets/
│   └── main.PNG                    ← 论文框架图
├── data/
│   └── processed_data/             ← 各数据集预处理后的图数据
│       ├── amazon/
│       ├── biomedical/
│       ├── dblp/
│       ├── goodreads/
│       ├── legal/
│       └── maple/{Biology,Chemistry,...}
└── Graph-Counselor/
    ├── model/                      ← 本地模型权重（不入 git）
    │   └── Qwen2.5-7B-Instruct/
    ├── results/                    ← 实验输出（不入 git）
    ├── scripts/
    │   └── run_Graph-Counselor.sh  ← 主运行脚本
    └── code/                       ← 核心代码
        ├── run.py                  ← 程序入口
        ├── GraphAgent.py           ← 基础 Agent（transformer 推理）
        ├── GraphAgent_vllm.py      ← 基础 Agent（vLLM 推理）
        ├── GraphReflectAgent.py    ← 反思 Agent（transformer 推理）
        ├── GraphReflectAgent_vllm.py   ← 反思 Agent（vLLM 推理）
        ├── GraphAgent_Plan_Reflect_vllm.py  ← 完整三智能体（主力）
        ├── graph_prompts.py        ← 所有 Prompt 模板定义
        ├── graph_fewshots.py       ← Few-shot 示例库
        └── tools/
            ├── graph_funcs.py      ← 图操作工具函数
            └── retriever.py        ← FAISS 向量检索器
```

---

## 3. 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                      run.py（入口）                       │
│  读取数据集 → 根据参数选择 Agent 类型 → 逐问题推理 → 保存结果 │
└─────────────────┬───────────────────────────────────────┘
                  │ 实例化
        ┌─────────▼──────────┐
        │   Agent 调度层       │
        │  compound_strategy  │
        │  reflexion_strategy │
        │  llm_way            │
        └──┬──────────┬───────┘
           │          │
   ┌───────▼──┐  ┌────▼─────────────────────────────┐
   │ 基础模式  │  │     Plan + Reflect 模式（主力）      │
   │GraphAgent│  │   GraphAgent_Plan_Reflect_vllm    │
   └──────────┘  └──────┬──────────┬────────────────┘
                        │          │
              ┌─────────▼──┐  ┌────▼──────────┐
              │  执行层      │  │   反思层        │
              │  ReAct 循环  │  │  Reflection   │
              │  max_steps  │  │  max_reflect  │
              └──────┬──────┘  └───────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
   ┌────▼────┐  ┌────▼────┐  ┌───▼──────┐
   │ 图操作   │  │ 向量检索  │  │  LLM 推理 │
   │graph_   │  │Retriever│  │ vLLM API │
   │funcs    │  │ (FAISS) │  │ port8020 │
   └─────────┘  └─────────┘  └──────────┘
```

---

## 4. 核心模块详解

### 4.1 `run.py` — 程序入口

负责：

- 解析命令行参数（数据集、模型路径、策略等）
- 加载 `data.json` 数据集
- 根据 `compound_strategy` × `reflexion_strategy` × `llm_way` 三个维度选择 Agent
- 驱动推理循环，统计 correct / incorrect / halted 结果
- 将结果写入 `results.jsonl`

### 4.2 `GraphAgent_Plan_Reflect_vllm.py` — 主力 Agent

项目最完整的实现，包含三个协同角色：

| 角色 | 功能 | 使用模型 |
|------|------|---------|
| **Execution Agent** | ReAct 模式在图上逐步执行动作 | `api_url`（大模型，port 8020） |
| **Eval Agent** | 判断当前答案是否正确 | `api_url2`（小模型，port 8010） |
| **Reflection Agent** | 分析失败原因，生成改进建议 | `api_url3`（大模型，port 8020） |

**执行流程（单问题）：**

```
输入问题
  │
  ▼
[Planning] 生成子问题分解计划
  │
  ▼ 循环 max_reflect 次
[Execution] ReAct 循环（最多 max_steps 步）
  ├── Thought：推理下一步
  ├── Action：调用工具（check_neighbours / check_nodes / lookup / finish）
  └── Observation：工具返回结果
  │
  ▼
[Evaluation] 判断答案是否正确
  ├── 正确 → 输出结果
  └── 错误 → [Reflection] 生成反思 → 重新执行
```

### 4.3 `tools/graph_funcs.py` — 图操作工具

Agent 可调用的原子操作：

| 函数 | 功能 |
|------|------|
| `check_neighbours(node, neighbor_type)` | 查询节点的邻居列表 |
| `check_nodes(node, feature)` | 查询节点的属性特征 |
| `check_degree(node, neighbor_type)` | 查询节点的度数 |
| `check_all_neighbour(node_q)` | 反向查询：谁的邻居包含此节点 |

### 4.4 `tools/retriever.py` — FAISS 向量检索器

- 使用 `sentence-transformers/all-mpnet-base-v2` 对所有图节点文本进行 Embedding
- 构建 FAISS 索引，支持 GPU 加速
- 支持本地缓存（`cache-all-mpnet-base-v2.pkl`），避免重复编码
- 提供 `lookup(query)` 接口，通过语义搜索定位起始节点

### 4.5 `graph_prompts.py` — Prompt 模板

定义了所有场景下的提示词：

| 变量名 | 用途 |
|--------|------|
| `graph_agent_prompt` | 基础 ReAct 提示 |
| `graph_compound_and_plan_prompt` | Plan + Compound 策略提示 |
| `graph_reflect_prompt` | 反思提示（multiple 模式） |
| `graph_reflect_prompt_base` | 反思提示（base 模式） |
| `graph_eval_prompt` | 答案评判提示 |
| `GRAPH_DEFINITION` | 图结构说明（注入所有提示） |

### 4.6 `graph_fewshots.py` — Few-shot 示例库

按数据集分类存储 few-shot 示例，键结构：

```python
EXAMPLES[dataset]           # 基础执行示例
PLAN_EXAMPLES[dataset]      # 带规划的执行示例
REFLECT_EXAMPLES_BASE[dataset]    # 反思示例（base）
PLAN_SHORT_REFLECT_EXAMPLES[dataset]  # 反思示例（plan模式）
PLAN_SHORT_EVAL_EXAMPLES[dataset]     # 评判示例
```

---

## 5. Agent 类型与选择逻辑

`run.py` 根据以下三个参数组合选择 Agent：

```
compound_strategy:   None | compound | plan_compound | plan
reflexion_strategy:  None | Last_attempt | Reflexion | Last_attempt_and_Reflexion
llm_way:             transformer | vllm
```

| compound_strategy | reflexion_strategy | llm_way | 实例化的 Agent |
|-------------------|--------------------|---------|--------------|
| None | None | transformer | `GraphAgent` |
| None | None | vllm | `GraphAgent_vllm` |
| None | Reflexion 等 | transformer | `GraphReflectAgent` |
| None | Reflexion 等 | vllm | `GraphReflectAgent_vllm` |
| plan_compound / plan | Reflexion 等 | vllm | `GraphAgent_Plan_Reflect_vllm` ← **推荐** |

---

## 6. 数据流

### 输入数据格式（`data.json`，jsonlines）

```json
{
  "question": "Which papers cite both A and B?",
  "answer": ["paper_123"],
  "graph": "dblp"
}
```

### 图数据格式（`graph.json`）

```json
{
  "paper_nodes": {
    "p001": {
      "features": {"title": "Attention is All You Need"},
      "neighbors": {"cites": ["p002", "p003"], "written_by": ["a001"]}
    }
  },
  "author_nodes": { ... }
}
```

### 输出格式（`results.jsonl`）

```json
{
  "question": "...",
  "answer": ["p001"],
  "pred": ["p001"],
  "correct": true,
  "trajectory": ["Thought: ...", "Action: check_neighbours[p001, cites]", ...]
}
```

---

## 7. 支持的模型与接入方式

### vLLM 本地部署（推荐）

| 服务 | 端口 | 默认模型 | 角色 |
|------|------|---------|------|
| server2 | 8010 | Qwen2.5-7B-Instruct | Eval Agent（判断答案） |
| server_test | 8020 | Qwen2.5-72B-Instruct | Execution + Reflection Agent |

### OpenAI API

直接设置 `--llm_version gpt-4` 等，无需本地部署。

### 百度千帆 API

设置 `--llm_version ERNIE-Speed-8K` 并配置 `QIANFAN_AK` / `QIANFAN_SK`。

---

## 8. 支持的数据集

| 数据集 | 描述 | 节点类型 |
|--------|------|---------|
| `dblp` | 学术论文引用图 | paper, author, venue |
| `amazon` | 电商商品图 | item, brand |
| `biomedical` | 生物医学知识图谱 | Disease, Gene, Compound 等11类 |
| `legal` | 法律判例图 | opinion, court, docket 等 |
| `goodreads` | 图书社交网络 | book, author, publisher, series |
| `maple` | 多学科学术图 | paper, author, venue（分5个子领域） |

---

## 9. 关键运行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | dblp | 使用的数据集 |
| `--max_steps` | 10 | 单次推理最大步数（ReAct 循环上限） |
| `--max_reflect` | 2 | 最大反思轮次 |
| `--compound_strategy` | plan_compound | 策略：None/compound/plan_compound/plan |
| `--reflexion_strategy` | Reflexion | 反思策略 |
| `--llm_way` | vllm | 推理后端：vllm 或 transformer |
| `--judge_correct` | llm | 答案判断方式：llm 或 groundtruth |
| `--reflect_prompt` | multiple | 反思 Prompt 类型：base/multiple/short_multiple |
| `--embedder_name` | all-mpnet-base-v2 | 向量检索模型 |
| `--api_url` | :8020/v1/completions | 主模型 API（Execution/Reflection） |
| `--api_url2` | :8010/v1/completions | 评判模型 API（Eval） |
