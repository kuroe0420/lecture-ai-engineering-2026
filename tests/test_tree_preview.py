"""候補の抜粋が説明本文を含み、ページ全文も取得できることを確認する。"""

from app import retrieval


def test_preview_contains_explanation_and_limits_length():
    result = retrieval.browse_tree("lec02-main-s045")
    page = next(p for p in result["children"] if p["id"] == "lec02-main-p052")
    original = retrieval.read_page(page["id"])["content"].strip()
    assert "元の重みは固定" in page["content"]
    assert page["content"] == original[:retrieval.TREE_PREVIEW_CHARS] + "…"
    assert len(page["content"]) < len(original)


def test_page_node_still_returns_complete_body():
    result = retrieval.browse_tree("lec02-main-p052")
    assert result["node"]["content"] == retrieval.read_page("lec02-main-p052")["content"]
    assert result["children"] == []
