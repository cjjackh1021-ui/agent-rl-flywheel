"""Deterministic dual-tool dataset generator with strict group_id leakage isolation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from agent_rl_flywheel.data.schema import DatasetBundle, TaskItem
from agent_rl_flywheel.tools.hybrid_search import DocumentChunk


def split_by_group_id(
    tasks: list[TaskItem],
    ratios: dict[str, float] | None = None,
    seed: int = 42,
) -> dict[str, list[TaskItem]]:
    """Split tasks by `group_id` so no group spans across any two splits."""
    target_ratios = ratios or {
        "train_coldstart": 0.25,
        "train_rl": 0.50,
        "val": 0.125,
        "locked_test_id": 0.125,
    }
    groups: dict[str, list[TaskItem]] = {}
    for task in tasks:
        groups.setdefault(task.group_id, []).append(task)

    group_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    split_names = list(target_ratios.keys())
    total_groups = len(group_keys)
    counts = [max(1, int(total_groups * target_ratios[name])) for name in split_names[:-1]]
    assigned = sum(counts)
    while assigned >= total_groups and any(c > 1 for c in counts):
        for i in range(len(counts)):
            if assigned < total_groups:
                break
            if counts[i] > 1:
                counts[i] -= 1
                assigned -= 1

    out: dict[str, list[TaskItem]] = {name: [] for name in split_names}
    cursor = 0
    for name, cnt in zip(split_names[:-1], counts):
        for gid in group_keys[cursor : cursor + cnt]:
            for item in groups[gid]:
                out[name].append(
                    TaskItem(
                        task_id=item.task_id,
                        group_id=item.group_id,
                        task_type=item.task_type,
                        split=name,
                        prompt=item.prompt,
                        gold_answer=item.gold_answer,
                        required_doc_ids=list(item.required_doc_ids),
                        requires_tool=item.requires_tool,
                        gold_turns=list(item.gold_turns),
                    )
                )
        cursor += cnt

    last_name = split_names[-1]
    for gid in group_keys[cursor:]:
        for item in groups[gid]:
            out[last_name].append(
                TaskItem(
                    task_id=item.task_id,
                    group_id=item.group_id,
                    task_type=item.task_type,
                    split=last_name,
                    prompt=item.prompt,
                    gold_answer=item.gold_answer,
                    required_doc_ids=list(item.required_doc_ids),
                    requires_tool=item.requires_tool,
                    gold_turns=list(item.gold_turns),
                )
            )

    # Hard leakage assertion
    seen_groups: set[str] = set()
    for name, items in out.items():
        gset = {t.group_id for t in items}
        overlap = seen_groups & gset
        if overlap:
            raise AssertionError(f"Group leakage detected into split {name}: {overlap}")
        seen_groups |= gset

    return out


def _make_math_task(idx: int, rng: random.Random, is_ood: bool) -> TaskItem:
    """Generate a multi-step symbolic/numeric math task verified by SymPy."""
    gid = f"{'ood' if is_ood else 'id'}_math_grp_{idx // 2}"
    variant = idx % 3
    if variant == 0:
        a = rng.randint(2, 9)
        b = rng.randint(3, 12)
        c = rng.randint(5, 19)
        d = rng.randint(1, 7)
        if is_ood:
            expr = f"({a}*{b} + {c}*{d}) * ({a} + {d})"
            ans = str((a * b + c * d) * (a + d))
            prompt = f"计算 ({a}*{b} + {c}*{d}) * ({a} + {d}) 的精确结果，请调用 sympy_eval 验算后给出答案。"
        else:
            expr = f"{a}*{b} + {c} - {d}"
            ans = str(a * b + c - d)
            prompt = f"计算 {a}*{b} + {c} - {d} 的精确数值，请使用 sympy_eval 工具验算。"
    elif variant == 1:
        r1 = rng.randint(2, 11)
        r2 = rng.randint(2, 11)
        if is_ood:
            # Derivative at point
            coef = rng.randint(2, 6)
            pt = rng.randint(2, 5)
            expr = f"diff({coef}*x**3 + {r1}*x, x).subs(x, {pt})" if False else f"3*{coef}*{pt}**2 + {r1}"
            ans = str(3 * coef * (pt**2) + r1)
            prompt = f"已知 f(x) = {coef}*x^3 + {r1}*x，求导数 f'({pt}) 的精确值（可用 sympy_eval 计算 3*{coef}*{pt}^2 + {r1}）。"
        else:
            expr = f"({r1} + {r2})**2 - {r1}*{r2}"
            ans = str((r1 + r2) ** 2 - r1 * r2)
            prompt = f"求表达式 ({r1} + {r2})^2 - {r1}*{r2} 的精确值，请调用 sympy_eval 验算。"
    else:
        p = rng.randint(3, 15)
        q = rng.randint(2, 8)
        expr = f"Rational({p}*{q} + {p}, {q})"
        ans = str(p * q + p) if (p * q + p) % q != 0 else str((p * q + p) // q)
        # Keep exact Rational form using sympy formula
        num = p * q + q
        expr = f"({num}) / {q}"
        ans = str(num // q)
        prompt = f"求解线性方程 {q}*x = {num} 中的 x 精确值，请调用 sympy_eval 计算 ({num})/{q}。"

    gold_turns = [
        {
            "assistant_text": (
                f"<think>需要通过 sympy_eval 精确验算 {expr}。</think>"
                f"<tool_call>{{\"name\": \"sympy_eval\", \"arguments\": {{\"expr\": \"{expr}\"}}}}</tool_call>"
            )
        },
        {
            "assistant_text": f"<think>工具返回精确结果为 {ans}，可以给出最终答案。</think><answer>{ans}</answer>"
        },
    ]
    return TaskItem(
        task_id=f"{'ood' if is_ood else 'id'}_math_{idx:04d}",
        group_id=gid,
        task_type="math",
        split="unassigned",
        prompt=prompt,
        gold_answer=ans,
        required_doc_ids=[],
        requires_tool=True,
        gold_turns=gold_turns,
    )


def _make_rag_task(
    idx: int,
    rng: random.Random,
    is_ood: bool,
) -> tuple[TaskItem, DocumentChunk]:
    """Generate a document chunk and paired retrieval + reasoning task requiring citation."""
    prefix = "ood" if is_ood else "id"
    gid = f"{prefix}_rag_grp_{idx // 2}"
    doc_id = f"doc_{prefix}_{idx:04d}"

    if is_ood:
        domain_names = ["边缘异构推理簇", "量子信道纠错节点", "卫星星间激光链路"]
        domain = domain_names[idx % len(domain_names)]
        base_ms = rng.randint(18, 45)
        scale = rng.randint(2, 5)
        overhead = rng.randint(4, 15)
        total = base_ms * scale + overhead
        code_tag = f"SPEC-OOD-{idx:04d}"
        title = f"{domain} 规范 {code_tag}"
        text = (
            f"文档编号 {doc_id}（规范代号 {code_tag}）：在 {domain} 调度模式下，"
            f"单节点基准延迟为 {base_ms} 毫秒，并发倍率因子为 {scale}，"
            f"附加校验开销为 {overhead} 毫秒。总约束预算按 base*scale + overhead 计算。"
        )
        query = f"{code_tag} {domain} 基准延迟 并发倍率"
        expr = f"{base_ms}*{scale} + {overhead}"
        prompt = (
            f"请先检索代号为 {code_tag} 的 {domain} 规范参数，再用 sympy_eval 计算总约束预算（单位毫秒数值），"
            f"在 <answer> 中给出精确数值并标注来源文档编号（例如 {doc_id}）。"
        )
        task_type = "hybrid_rag_math"
        gold_turns = [
            {
                "assistant_text": (
                    f"<think>先检索规范 {code_tag} 的参数定义。</think>"
                    f"<tool_call>{{\"name\": \"hybrid_search\", \"arguments\": {{\"query\": \"{query}\", \"top_k\": 3}}}}</tool_call>"
                )
            },
            {
                "assistant_text": (
                    f"<think>已从 {doc_id} 查得 base={base_ms}, scale={scale}, overhead={overhead}，用 sympy_eval 计算总额。</think>"
                    f"<tool_call>{{\"name\": \"sympy_eval\", \"arguments\": {{\"expr\": \"{expr}\"}}}}</tool_call>"
                )
            },
            {
                "assistant_text": f"<think>总预算已确认，引用 {doc_id} 输出最终答案。</think><answer>{total} ({doc_id})</answer>"
            },
        ]
        gold_ans = str(total)
    else:
        topics = ["实验缓存前缀阈值", "声纹水印容错参数", "训练梯度累积步数"]
        topic = topics[idx % len(topics)]
        val = rng.randint(25, 95)
        code_tag = f"MANUAL-ID-{idx:04d}"
        title = f"{topic} 手册 {code_tag}"
        text = (
            f"文档编号 {doc_id}（手册代号 {code_tag}）：针对 {topic} 配置，"
            f"系统规定标准阈值参数为 {val}，超出该阈值需触发二级审计。"
        )
        query = f"{code_tag} {topic} 标准阈值"
        prompt = (
            f"请检索手册代号 {code_tag}（{topic}）规定的标准阈值参数数值，"
            f"并在 <answer> 中给出精确数值和引用文档编号（如 {doc_id}）。"
        )
        task_type = "rag"
        gold_turns = [
            {
                "assistant_text": (
                    f"<think>调用 hybrid_search 查询手册 {code_tag} 的标准阈值。</think>"
                    f"<tool_call>{{\"name\": \"hybrid_search\", \"arguments\": {{\"query\": \"{query}\", \"top_k\": 3}}}}</tool_call>"
                )
            },
            {
                "assistant_text": f"<think>在 {doc_id} 中查到标准阈值为 {val}，按要求作答并标注引用。</think><answer>{val} ({doc_id})</answer>"
            },
        ]
        gold_ans = str(val)

    chunk = DocumentChunk(doc_id=doc_id, group_id=gid, title=title, text=text)
    task = TaskItem(
        task_id=f"{prefix}_rag_{idx:04d}",
        group_id=gid,
        task_type=task_type,
        split="unassigned",
        prompt=prompt,
        gold_answer=gold_ans,
        required_doc_ids=[doc_id],
        requires_tool=True,
        gold_turns=gold_turns,
    )
    return task, chunk


def build_benchmark_dataset(
    seed: int = 42,
    total_id: int = 400,
    total_ood: int = 60,
    ratios: dict[str, float] | None = None,
) -> DatasetBundle:
    """Build deterministic ID + OOD dual-tool benchmark with strict group_id split isolation."""
    rng = random.Random(seed)
    id_tasks: list[TaskItem] = []
    corpus: list[DocumentChunk] = []

    half_id = total_id // 2
    for i in range(half_id):
        id_tasks.append(_make_math_task(i, rng=rng, is_ood=False))
    for i in range(total_id - half_id):
        task, chunk = _make_rag_task(i, rng=rng, is_ood=False)
        id_tasks.append(task)
        corpus.append(chunk)

    splits = split_by_group_id(id_tasks, ratios=ratios, seed=seed)

    ood_tasks: list[TaskItem] = []
    half_ood = total_ood // 2
    for i in range(half_ood):
        t = _make_math_task(i, rng=rng, is_ood=True)
        ood_tasks.append(
            TaskItem(
                task_id=t.task_id,
                group_id=t.group_id,
                task_type=t.task_type,
                split="locked_test_ood",
                prompt=t.prompt,
                gold_answer=t.gold_answer,
                required_doc_ids=list(t.required_doc_ids),
                requires_tool=t.requires_tool,
                gold_turns=list(t.gold_turns),
            )
        )
    for i in range(total_ood - half_ood):
        t, chunk = _make_rag_task(i, rng=rng, is_ood=True)
        corpus.append(chunk)
        ood_tasks.append(
            TaskItem(
                task_id=t.task_id,
                group_id=t.group_id,
                task_type=t.task_type,
                split="locked_test_ood",
                prompt=t.prompt,
                gold_answer=t.gold_answer,
                required_doc_ids=list(t.required_doc_ids),
                requires_tool=t.requires_tool,
                gold_turns=list(t.gold_turns),
            )
        )

    splits["locked_test_ood"] = ood_tasks
    return DatasetBundle(
        splits=splits,
        corpus=corpus,
        metadata={
            "seed": seed,
            "total_id": total_id,
            "total_ood": total_ood,
            "split_counts": {k: len(v) for k, v in splits.items()},
        },
    )


def save_dataset_bundle(bundle: DatasetBundle, out_dir: str | Path) -> Path:
    """Persist dataset splits and corpus JSONL files under `out_dir`."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    for split_name, tasks in bundle.splits.items():
        with (root / f"{split_name}.jsonl").open("w", encoding="utf-8") as f:
            for item in tasks:
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    with (root / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for chunk in bundle.corpus:
            f.write(
                json.dumps(
                    {
                        "doc_id": chunk.doc_id,
                        "group_id": chunk.group_id,
                        "title": chunk.title,
                        "text": chunk.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (root / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(bundle.metadata, f, ensure_ascii=False, indent=2)
    return root


def load_dataset_bundle(data_dir: str | Path) -> DatasetBundle:
    """Load dataset splits and corpus JSONL files from `data_dir`."""
    root = Path(data_dir)
    splits: dict[str, list[TaskItem]] = {}
    for path in sorted(root.glob("*.jsonl")):
        if path.name == "corpus.jsonl":
            continue
        split_name = path.stem
        items: list[TaskItem] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(TaskItem.from_dict(json.loads(line)))
        splits[split_name] = items

    corpus: list[DocumentChunk] = []
    corpus_path = root / "corpus.jsonl"
    if corpus_path.exists():
        with corpus_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw: dict[str, Any] = json.loads(line)
                    corpus.append(DocumentChunk(**raw))

    meta: dict[str, Any] = {}
    meta_path = root / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return DatasetBundle(splits=splits, corpus=corpus, metadata=meta)
