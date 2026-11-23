"""前後ページの取得で資料の境界と重複を確認する。APIは呼ばない。"""

from app import retrieval


def sample_pages():
    return [
        {"page_id": f"{doc}-p{n}", "document_id": doc, "page": n,
         "content": f"{doc}の本文{n}", "title": f"ページ{n}"}
        for doc, numbers in [("lec01-main", (1, 2, 3)), ("lec02-main", (1,))]
        for n in numbers
    ]


def test_neighbors_keep_body_and_do_not_cross_documents(monkeypatch):
    pages = sample_pages()
    monkeypatch.setattr(retrieval, "load_pages", lambda: pages)
    hit = {**pages[1], "distance": 0.1}
    results = retrieval.expand_neighbors([hit])
    assert [p["page"] for p in results] == [2, 1, 3]
    assert all(p["document_id"] == "lec01-main" for p in results)
    assert results[0]["distance"] == 0.1
    assert results[0]["retrieval_type"] == "match"
    assert results[1]["content"] == "lec01-mainの本文1"
    assert results[1]["retrieval_type"] == "neighbor"


def test_overlapping_hits_are_unique_and_keep_match_metadata(monkeypatch):
    pages = sample_pages()
    monkeypatch.setattr(retrieval, "load_pages", lambda: pages)
    results = retrieval.expand_neighbors([
        {**pages[1], "distance": 0.1}, {**pages[2], "distance": 0.2},
    ])
    assert len(results) == 3
    assert len({p["page_id"] for p in results}) == 3
    third = next(p for p in results if p["page"] == 3)
    assert third["retrieval_type"] == "match"
    assert third["distance"] == 0.2


def test_boundary_and_empty_results(monkeypatch):
    pages = sample_pages()
    monkeypatch.setattr(retrieval, "load_pages", lambda: pages)
    assert retrieval.expand_neighbors([]) == []
    assert len(retrieval.expand_neighbors([pages[0]])) == 2
    assert len(retrieval.expand_neighbors([pages[2]])) == 2
    assert retrieval.expand_neighbors([pages[3]]) == [{**pages[3], "retrieval_type": "match"}]


def test_neighbor_switch_preserves_the_same_search(monkeypatch):
    pages = sample_pages()
    queries = []

    def search(query):
        queries.append(query)
        return [pages[1]]

    monkeypatch.setattr(retrieval, "search", search)
    monkeypatch.setattr(retrieval, "load_pages", lambda: pages)
    monkeypatch.setattr(retrieval, "RAG_INCLUDE_NEIGHBORS", False)
    assert retrieval.search_with_neighbors("比較する質問") == [pages[1]]
    monkeypatch.setattr(retrieval, "RAG_INCLUDE_NEIGHBORS", True)
    assert len(retrieval.search_with_neighbors("比較する質問")) == 3
    assert queries == ["比較する質問", "比較する質問"]
