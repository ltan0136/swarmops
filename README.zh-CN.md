# SwarmOps

**跑在云原生基础设施上的 Agent 原生蜂群。** Agent 负责决定“做什么”，平台负责让它“跑得稳”。

[English](README.md)

- **Agent 原生层：** 基于 [LangGraph](https://github.com/langchain-ai/langgraph) 的三智能体蜂群（planner、executor、reviewer），沿显式定义的边互相交接控制权（handoff）。
- **云原生层：** 异步 FastAPI 服务；Postgres checkpointer，保证每段对话在重启后依然存在；带健康检查的容器；CI 会把整套服务真正启动起来做冒烟测试。
- **不需要 API key 也能跑：** 默认使用一个确定性的离线模型驱动完整运行时。改两个环境变量即可切换到 Ollama、DeepSeek、Qwen 或 OpenAI。

架构图见 [英文 README](README.md#architecture)。

## 快速开始（Docker，无需 API key）

```bash
cp .env.example .env
docker compose up -d --build --wait
```

调用一次：

```bash
curl -s -X POST http://localhost:8000/threads/demo/runs \
  -H 'Content-Type: application/json' \
  -d '{"message": "What is (12 + 3) * 4?"}'
```

返回里的 `answer` 是最终答案，`steps` 是这一轮里每个 agent 的每一步（规划、工具调用、交接、评审）。

以 SSE 流式查看每一步：

```bash
curl -N -X POST http://localhost:8000/threads/demo/runs/stream \
  -H 'Content-Type: application/json' \
  -d '{"message": "And 2 ** 10?"}'
```

重启 API 后对话仍然存在：

```bash
docker compose restart api && docker compose up -d --wait
curl -s http://localhost:8000/threads/demo
```

交互式接口文档：<http://localhost:8000/docs>。用 `docker compose down -v` 停止并清理。

## 切换到真实模型

在 `.env` 里设置以下变量，然后重新执行 `docker compose up -d --build --wait`。任何兼容 OpenAI chat-completions 接口、并支持工具调用的服务都可以。

| 提供方 | `LLM_BASE_URL` | `LLM_MODEL` 示例 |
|---|---|---|
| 本机 Ollama | `http://host.docker.internal:11434/v1` | `qwen2.5:7b` |
| compose 内的 Ollama（`--profile local-llm`） | `http://ollama:11434/v1` | `qwen2.5:7b` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 通义千问（DashScope 国际站） | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |

```dotenv
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
LLM_API_KEY=sk-...
```

模型名称会随时间变化，以各家文档为准。务必选支持工具调用的模型，因为 agent 之间的交接本身就是一次工具调用。

## 本地开发（不用 Docker）

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

swarmops "What is (12 + 3) * 4?"        # 命令行跑一次
uvicorn swarmops.api.app:app --reload   # 或者在 :8000 启动 API
pytest -q                                # 测试使用离线模型和内存状态
```

不设置 `DATABASE_URL` 时，状态保存在内存里。要运行 Postgres 集成测试，把 `SWARMOPS_TEST_DATABASE_URL` 指向任意一个 Postgres（例如 `docker compose up postgres` 启动的那个）。

## 一次运行的过程

1. 蜂群把消息路由给该 thread 的 `active_agent`（新 thread 默认是 planner）。
2. **planner** 写一个简短计划，交接给 **executor**。
3. **executor** 调用工具，再交接给 **reviewer**。
4. **reviewer** 给出最终答案，或者把工作退回 executor 重做。
5. 每一步都以 thread id 为键写入 Postgres。此时 reviewer 是活跃 agent，所以下一条消息先到它手里，它会把新请求交回 planner。

## 安全说明

- `.env` 已被 git 忽略，仓库里只有不含真实 key 的 `.env.example`。API key 只放在本地 `.env` 里；CI 使用离线模型，不需要任何密钥。
- 设置 `SWARMOPS_API_KEY` 后，除 `/healthz`（容器健康检查用）外的所有接口都要求 `Authorization: Bearer <key>`。默认不设置，docker compose 把 API 和 Postgres 都只绑定在 `127.0.0.1`；没有 key 和 TLS 时不要把 API 暴露到网络上。
- `DELETE /threads/{thread_id}` 删除该 thread 的全部 checkpoint，之后同一 id 会作为新会话重新开始；thread 正在运行时返回 `409`。
- `.env.example`、compose 和 CI 里的 Postgres 账号口令只是本地占位值，任何共享环境都要改掉。
- calculator 用 `ast` 解析表达式而不是 `eval`，模型生成的输入无法执行任意代码。

## 设计决策

- [ADR 0001：限制交接边的蜂群](docs/adr/0001-swarm-with-restricted-handoffs.md)
- [ADR 0002：用 Postgres 做 checkpointer](docs/adr/0002-postgres-checkpointer.md)
- [ADR 0003：用于演示、测试和 CI 的离线脚本模型](docs/adr/0003-scripted-offline-model.md)

## 路线图

- [x] **v0.1** 三智能体蜂群、带 SSE 的异步 API、Postgres checkpointer、Docker、CI
- [ ] **v0.2** 工具拆成独立的 MCP 服务容器；高风险工具走人工审批
- [ ] **v0.3** Kubernetes（kind + Helm）、Redis 任务队列、KEDA 自动扩缩、OpenTelemetry + Langfuse
- [ ] **v0.4** CI 中的评测集；实验运行器比较不同蜂群规模、拓扑和交接规则，包括留一法消融

## 许可证

[MIT](LICENSE)
