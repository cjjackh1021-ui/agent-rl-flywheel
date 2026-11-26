# Agentic Post-Training Flywheel (`agent-rl-flywheel`)

面向多轮工具调用 Agent 的全链路后训练与防坍塌评测闭环，聚焦 **Zero Retokenization Drift（零重分词漂移）**、**跨家族前缀保留序列级 OPD/DPO（`DeepSeek-V4.1-Flash` Teacher）**、**DAPO** 与 **RAGEN-2 模板坍塌诊断 + SNR 分组过滤**，并在设计上专门适配 **消费级 12GB 显存（RTX 4070 SUPER）QLoRA 训练** 与 **Mac 纯 PyTorch 快速回归验证**。

---

## 一、核心架构与 2026 前沿设计亮点

```text
+----------------------------------------------------------------------+
|  Multi-Turn Agent Harness (<think> -> <tool_call> -> <obs> -> <ans>) |
|  - 安全沙箱 SymPy 符号计算（AST 白名单 + 防高次幂炸弹）               |
|  - 混合检索器 HybridRetriever（Okapi BM25 + 向量相似度 + RRF 融合）   |
+----------------------------------+-----------------------------------+
                                   | 仅传 token_ids（零重分词漂移）
                                   v
+----------------------------------------------------------------------+
|  AgentProxyGateway + ExactTokenBuffer                                |
|  - 严格保留采样 token_ids 与 rollout logprobs                         |
|  - 对 System/User/工具 <obs> 强制打 assistant_mask = 0               |
+----------------------------------+-----------------------------------+
                                   | RLVR 复合可验证奖励 + 反作弊门禁
                                   v
+----------------------------------------------------------------------+
|  Post-Training & RAGEN-2 诊断引擎                                     |
|  - Exp1a: Static SFT（DeepSeek-V4.1-Flash 验证轨迹冷启动）           |
|  - Exp1b: Cross-Family Prefix OPD + On-Policy DPO                    |
|           （保留学生出错前真实前缀，DeepSeek 从断点纠错生成偏好对）   |
|  - Exp2a: 标准 GRPO 基线                                             |
|  - Exp2b: DAPO (Clip-Higher + Dynamic Sampling + Token-Level Loss)   |
|           + RAGEN-2 在线互信息代理 (MI Proxy) + SNR-Aware 组过滤      |
+----------------------------------------------------------------------+
```

### 关键技术细节
1. **Zero Retokenization Drift 与严格 `<obs>` Mask**：参考 `Agent Lightning` Proxy 思路，通过 `ExactTokenBuffer` 直接拼接采样 token_ids，绝不对多轮历史文本做二次分词，同时确保工具返回结果零梯度。
2. **RLVR 反 Reward Hacking 门禁**：当模型仅空转工具而最终答案错误时，工具奖励被严格封顶并扣除重复调用惩罚，防止“刷工具分”作弊。
3. **跨词表 `DeepSeek-V4.1-Flash`（`deepseek-flash`）Teacher 蒸馏方案**：针对 DeepSeek 与 Qwen 词表不同的现实约束，实现 **Prefix-Preserving Sequence-Level Correction + On-Policy DPO**（仅在学生失败断点后对教师修正片段计算监督/偏好梯度），并额外预留同词表 `token_level_opd_loss` 接口。
4. **DAPO + RAGEN-2（`arXiv:2604.06268`）防模板坍塌**：
   - `DAPO` 实现非对称裁剪（`clip_eps_low=0.2, clip_eps_high=0.28`）、`Dynamic Sampling` 剔除零方差组、Token 级全局损失归一化与 `Overlong Reward Shaping`；
   - `RAGEN-2` 在传统 `Token Entropy` 之外同步监控跨输入 `MI Proxy`（JS 散度 + 跨 Prompt 余弦差异），及时抓出“熵看似正常但千篇一律套话”的模板坍塌，并通过 `SNR-Aware` 组过滤保护有效梯度。

---

## 二、快速开始

### 1. Mac / 本地纯 PyTorch 环境验证与回归测试
```bash
PYTHONPATH=src pytest -W error -v

# 本地 TinyCausalLM 冒烟跑通全部 4 阶段消融
PYTHONPATH=src python3 scripts/build_dataset.py --out data/generated
PYTHONPATH=src python3 scripts/run_experiment.py --backend tiny --stage smoke_all --data-dir data/generated --out-dir outputs/mac_smoke
```

> **验证边界说明**：Mac 环境使用 2 层 `TinyCausalLM` + UTF-8 字节分词器验证完整损失反传、`<obs>` Mask、DAPO/RAGEN-2 逻辑与 CLI 链路；真实小参数模型精度提升需在具备 CUDA GPU 的 Windows WSL2 4070 端执行。

### 2. Windows WSL2 (Ubuntu) + RTX 4070 SUPER 12GB 实机训练
```bash
# 1. 创建环境并安装 4070 依赖
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-4070.txt
pip install -e .

# 2. 配置 DeepSeek-V4.1-Flash API Key
export DEEPSEEK_API_KEY="your_deepseek_api_key"

# 3. 一键运行 Qwen3.5-2B 4-bit QLoRA 全套消融实验（稳妥备选可传 configs/qwen3_1p7b_4070.yaml）
bash scripts/run_4070_pipeline.sh configs/qwen35_2b_4070.yaml
```

---

## 三、目录结构
- `configs/`：`Qwen3.5-2B` 与 `Qwen3-1.7B` 4070 12GB 配置文件
- `src/agent_rl_flywheel/tools/`：AST 沙箱 SymPy 校验器、Okapi BM25 + 向量 + RRF 混合检索器
- `src/agent_rl_flywheel/env/`：`AgentProxyGateway`、`ExactTokenBuffer`、多轮 Harness、RLVR 验证器
- `src/agent_rl_flywheel/data/`：防泄漏 `group_id` 切分、双工具数据集生成、`DeepSeek-V4.1-Flash` 蒸馏管线
- `src/agent_rl_flywheel/rl/`：`SFT / DPO / Token-OPD / GRPO / DAPO` 纯 PyTorch 实现、`RAGEN-2` 诊断与 `AgentRLTrainer`
- `src/agent_rl_flywheel/backends/`：`TinyCausalLMBackend` 与懒加载 `HFQLoRABackend`（利用 `disable_adapter()` 零额外显存算 Reference Logprobs）
- `src/agent_rl_flywheel/eval/`：ID / OOD 锁定测试集评测与 Markdown 消融表生成
- `tests/`：全覆盖单元测试（在 `-W error` 零警告约束下运行）
