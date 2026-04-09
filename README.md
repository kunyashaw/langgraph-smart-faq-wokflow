# LangGraph Smart FAQ Workflow

基于 LangGraph + MiniMax 的智能问答工作流，支持多轮对话、路由分发、Agent 调用和 LangSmith 调用链追踪。

## 功能特性

- **多轮对话**：基于 `add_messages` reducer 自动累积对话历史
- **智能路由**：自动判断走通用问答 Agent 还是代码处理 Agent
- **双 Agent 架构**：prompt_agent（通用问答）+ code_agent（代码处理）
- **LangSmith 追踪**：完整的调用链监控

## 快速开始

### 环境准备

```bash
# 安装依赖
uv add langchain-openai langgraph python-dotenv
uv add langchain-core langchain-community

# 配置 API Key
cp .env.example .env
# 编辑 .env 填入 MINIMAX_API_KEY 和 SMITH_API_KEY
```

### 运行

```bash
# 查看工作流图
uv run python run_workflow.py --show-graph

# 单轮问答
uv run python run_workflow.py --user-input "你好，介绍一下自己"

# 多轮对话
uv run python run_workflow.py
```

## 项目结构

```
demos/
├── agents/              # Agent 定义
│   ├── code_agent.py    # 代码处理 Agent
│   └── prompt_agent.py   # 通用问答 Agent
├── core/                # 核心工具
│   ├── llm.py           # LLM 初始化
│   └── tracing.py       # LangSmith 配置
├── tools/               # 工具集
├── workflow/            # 工作流
│   ├── nodes/           # 节点实现
│   ├── graph/           # 图定义
│   ├── routes/          # 路由逻辑
│   └── states/          # 状态定义
└── run_workflow.py      # 入口脚本
```

## 工作流图

```
START → router_node → [prompt_agent_node / code_agent_node] → END
```

详细文档：[blog_post.md](blog_post.md)

## License

MIT
