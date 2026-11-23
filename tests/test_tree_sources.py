"""Treeの候補だけで回答した場合に本文の取得へ戻せることを確認する。"""

import json
from types import SimpleNamespace

import pytest
from pydantic_ai import ModelRetry
from pydantic_ai.messages import ModelRequest, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from app.agent import tree_agent, tree_citations, validate_tree_sources
from app.retrieval import browse_tree


@pytest.mark.parametrize("citation", [
    "第2回・PDF p.52, 53", "第2回・PDF p.52, p.53", "第2回・PDF p.52-53",
])
def test_citation_formats(citation):
    assert tree_citations(citation) == {(2, 52), (2, 53)}


def test_unopened_citation_is_rejected():
    ctx = SimpleNamespace(messages=[ModelRequest(parts=[ToolReturnPart(
        tool_name="browse_tree", content=json.dumps(browse_tree("lec02-main-p052")), tool_call_id="p52",
    )])])
    with pytest.raises(ModelRetry, match="lec02-main-p053"):
        validate_tree_sources(ctx, "説明（第2回・PDF p.52, 53）")


def test_agent_retries_preview_answer_and_reads_full_pages():
    step = 0

    def respond(messages, info):
        nonlocal step
        step += 1
        if step == 1:
            return ModelResponse(parts=[ToolCallPart(
                "browse_tree", {"node_id": "lec02-main-s045"}, tool_call_id="section",
            )])
        if step == 2:
            return ModelResponse(parts=[TextPart("説明（第2回・PDF p.52, 53）")])
        if step == 3:
            assert any(isinstance(p, RetryPromptPart) for m in messages for p in m.parts)
            return ModelResponse(parts=[
                ToolCallPart("browse_tree", {"node_id": f"lec02-main-p{n:03d}"}, tool_call_id=f"p{n}")
                for n in (52, 53)
            ])
        return ModelResponse(parts=[TextPart("本文確認後の説明（第2回・PDF p.52, 53）")])

    with tree_agent.override(model=FunctionModel(respond)):
        result = tree_agent.run_sync("LoRAとFT・RAGの使い分け")
    assert result.output == "本文確認後の説明（第2回・PDF p.52, 53）"
    assert result.usage.requests == 4


def test_agent_retries_answer_without_any_tool_calls():
    step = 0

    def respond(messages, info):
        nonlocal step
        step += 1
        if step == 1:
            return ModelResponse(parts=[TextPart("未取得の説明（第2回・PDF p.999）")])
        if step == 2:
            assert any(isinstance(p, RetryPromptPart) for m in messages for p in m.parts)
            return ModelResponse(parts=[ToolCallPart(
                "browse_tree", {"node_id": "lec02-main-p052"}, tool_call_id="p52",
            )])
        return ModelResponse(parts=[TextPart("本文確認後の説明（第2回・PDF p.52）")])

    with tree_agent.override(model=FunctionModel(respond)):
        result = tree_agent.run_sync("LoRAとは？")
    assert result.output == "本文確認後の説明（第2回・PDF p.52）"
    assert result.usage.requests == 3
