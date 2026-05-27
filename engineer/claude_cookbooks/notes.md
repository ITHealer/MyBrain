---
modified: 2026-05-27T05:59:53.109Z
title: Knowledge Graph — Giải pháp theo đúng luồng build
---

# Knowledge Graph — Giải pháp theo đúng luồng build

---

## Luồng tổng quan

```
[1. Extraction] → [2. Entity Resolution] → [3. Graph Assembly] 
→ [4. Summarization] → [5. Querying] → [6. Evaluation] → [7. Scale]
```

---

## 1. Extraction — Trích xuất entities và relations

### Vấn đề: NER truyền thống cần training data per domain

Hệ thống NER cổ điển cần labeled dataset, trained model riêng cho mỗi domain (y tế, tài chính...). Khi domain thay đổi, phải train lại. Relation extraction lại cần thêm một model khác nữa.

**Giải pháp: Structured output thay thế toàn bộ**

Dùng Pydantic schema + `client.messages.parse()` — schema IS the training. Một call duy nhất extract cả entities lẫn relations, không cần training data.

```python
class Entity(BaseModel):
    name: str
    type: Literal["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "ARTIFACT"]
    description: str  # ← BẮT BUỘC: 1 câu grounded trong document này
                      # Dùng ở bước Resolution để disambiguate

class Relation(BaseModel):
    source: str       # phải là entity đã extract trong document này
    predicate: str    # short verb phrase: "commanded", "launched from"
    target: str       # phải là entity đã extract trong document này

class ExtractedGraph(BaseModel):
    entities: list[Entity]
    relations: list[Relation]
```

**Tại sao `description` bắt buộc ngay lúc extract?**

Nếu không có description, bước Resolution sau này không phân biệt được "Armstrong (jazz trumpeter)" vs "Armstrong (astronaut)" — cùng tên, khác người. Description là context duy nhất để LLM arbitrate đúng.

**Rule quan trọng:**
- Chỉ extract entities **central** to document — bỏ incidental mentions
- Predicate phải là short verb phrase, không phải noun ("leadership of" → sai, "led" → đúng)
- Mọi relation phải connect 2 entities đã extract trong cùng document — không invent cross-document relations ở bước này

**Model selection:**
```python
EXTRACTION_MODEL = "claude-haiku-4-5"   # cheap, fast, schema-constrained
SYNTHESIS_MODEL  = "claude-sonnet-4-6"  # dùng cho resolution + summarization
# Không dùng model nano cho KG extraction — không follow JSON format đủ chính xác
```

---

## 2. Entity Resolution — Gộp surface forms về canonical

### Vấn đề: Cùng entity, nhiều surface form → graph bị phân mảnh

Từ 6 documents về Apollo:
```
"NASA" xuất hiện ở doc 1
"National Aeronautics and Space Administration" ở doc 2
→ 2 nodes riêng biệt, không có edge nào nối
→ multi-hop query fail hoàn toàn
```

String similarity (edit distance, Jaccard) bắt được typo nhưng fail với:
```
"Edwin Aldrin" vs "Buzz Aldrin"  ← zero character overlap, cùng người
```

**Giải pháp: LLM clustering với description context**

```python
RESOLVE_PROMPT = """Below are {entity_type} entities extracted from several documents.
Some are different surface forms of the same real-world entity.

<entities>
{entity_list}
# format: "- name: one-sentence description"
</entities>

Cluster them. Rules:
- Each input name must appear in exactly one cluster's aliases list
- Entities genuinely distinct → own single-element cluster  
- Use descriptions to avoid merging entities that share a name only
- Canonical = most complete, unambiguous form"""
```

**Tại sao cần description ở đây?**

```
"Armstrong: first person to walk on the Moon" 
"Armstrong: famous jazz trumpeter"
→ LLM thấy description → KHÔNG merge

Nếu chỉ có tên "Armstrong" → LLM có thể merge sai
```

### Issue 1: Entity bị mất sau resolution

**Vấn đề:** LLM bỏ sót một entity không assign vào cluster nào → `alias_to_canonical` không có key → node biến mất silently.

**Giải pháp: Mandatory fallback**

```python
def resolve_with_fallback(entity_type, entities):
    try:
        clusters = resolve(entity_type, entities)
    except APIError:
        # Hard fallback: mỗi name là 1 cluster
        return [Cluster(canonical=n, aliases=[n]) 
                for n in {e["name"] for e in entities}]
    
    # Verify không có entity nào bị drop
    all_input_names = {e["name"] for e in entities}
    all_clustered = {a for c in clusters for a in c.aliases}
    dropped = all_input_names - all_clustered
    
    for name in dropped:
        # Thêm single-element cluster cho entity bị bỏ sót
        clusters.append(Cluster(canonical=name, aliases=[name]))
    
    return clusters
```

### Issue 2: Over-merging — merge thực thể sai

**Vấn đề:** "Gemini 12" (specific mission) bị fold vào "Project Gemini" (broad program) vì description overlap.

**Giải pháp:**
- Extraction description phải đủ specific: "Gemini 12 — Buzz Aldrin's last spaceflight before Apollo" vs "Project Gemini — NASA's second human spaceflight program"
- Spot-check output sau mỗi resolution run, đặc biệt với entities cùng root word

### Issue 3: Scale — 10,000 entities không fit 1 prompt

**Vấn đề:** Với corpus lớn, không thể dump tất cả PERSON entities vào 1 prompt.

**Giải pháp: Block trước bằng cheap signals**

```python
def block_entities(entities: list[dict]) -> list[list[dict]]:
    """
    Cheap signals để group candidates trước khi gọi LLM:
    1. Same last name
    2. Overlapping significant tokens (Jaccard > 0.3)  
    3. Embedding cosine similarity > 0.85
    
    LLM chỉ arbitrate WITHIN blocks, không cross-block
    Block size target: 50-100 entities
    """
    # Implement blocking logic
    # Sau đó gọi resolve() per block
    pass
```

---

## 3. Graph Assembly — Xây dựng graph từ canonical entities

### Vấn đề: Raw relations dùng surface forms, không dùng canonical names

Sau resolution, `alias_to_canonical` map mọi surface form về canonical. Phải rewrite toàn bộ relation endpoints trước khi add vào graph.

**Giải pháp: Rewrite + validate**

```python
G = nx.MultiDiGraph()
# MultiDiGraph vì:
# - Multi: 2 entities có thể có nhiều predicate khác nhau
#   ("launched from" VÀ "operated by" giữa cùng 2 nodes)
# - Di: direction matters 
#   ("Armstrong commanded Apollo 11" ≠ "Apollo 11 commanded Armstrong")

for r in raw_relations:
    src = alias_to_canonical.get(r["source"])
    tgt = alias_to_canonical.get(r["target"])
    
    # Guard 1: cả 2 endpoint phải có canonical form
    if not src or not tgt:
        continue
    
    # Guard 2: no self-loops
    if src == tgt:
        continue
    
    G.add_edge(src, tgt, predicate=r["predicate"], source_doc=r["source_doc"])
```

**Node metadata cần lưu:**
```python
G.add_node(canonical, 
    type=...,           # PERSON / ORG / ...
    description=...,    # từ extraction
    source_docs=[],     # list documents mention entity này
    mentions=0          # đếm số lần mention cross-doc
)
# Sau khi build: dedup source_docs
G.nodes[n]["source_docs"] = sorted(set(G.nodes[n]["source_docs"]))
```

**Sanity check sau khi build:**
```python
# Graph tốt phải là 1 connected component
# Nhiều components → entity resolution bị miss
components = nx.number_weakly_connected_components(G)
if components > 1:
    # Inspect isolated nodes → tìm aliases chưa được resolve đúng
```

---

## 4. Summarization — Làm giàu hub nodes

### Vấn đề: Hub nodes chỉ có description từ 1 document đầu tiên

Entity như "Apollo program" xuất hiện trong 6 documents, nhưng node chỉ lưu description từ document đầu. Bỏ mất nhiều context.

**Giải pháp: Pool tất cả mentions + graph neighborhood → synthesize**

```python
SUMMARIZE_PROMPT = """Generate a knowledge-graph profile for this entity.

Entity: {name} ({etype})

Source excerpts mentioning this entity:
{excerpts}          # ← TẤT CẢ documents mention entity này

Known relations in the graph:
{relations}         # ← subgraph neighborhood làm context

Rules:
- 2-3 paragraph factual summary, resolve contradictions bằng cách prefer most specific claim
- 3-5 key_facts, mỗi fact traceable về source cụ thể
- time_range: YYYY format, "unknown" hoặc "ongoing" nếu không rõ
- KHÔNG invent facts không có trong excerpts"""

class EntityProfile(BaseModel):
    summary: str
    key_facts: list[str]
    time_range: TimeRange  # {start: str, end: str}
```

**Khi nào cần re-summarize?**
```python
# CHỈ re-summarize khi source_docs set thay đổi materially
# Không re-summarize khi document mới không mention entity này
old_source_docs = set(node["source_docs"])
new_source_docs = old_source_docs | {new_doc_title}
if new_source_docs != old_source_docs:
    profile = summarize_entity(node_name)
```

---

## 5. Querying — Multi-hop reasoning

### Vấn đề: Ungrounded answer — LLM trả lời từ pretraining

**Vấn đề:** LLM biết Apollo program từ training data → có thể trả lời đúng nhưng không traceable, không auditable. Với private corpus (tài liệu nội bộ), LLM không có pretraining knowledge → answer sai hoàn toàn.

**Giải pháp: Serialize subgraph → constrain prompt**

```python
def serialize_subgraph(center: str, hops: int = 2) -> str:
    """
    2-hop neighborhood đủ cho hầu hết queries.
    3-hop bắt đầu noisy và đắt token.
    """
    nodes = {center}
    frontier = {center}
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            nxt |= set(G.successors(n)) | set(G.predecessors(n))
        frontier = nxt - nodes
        nodes |= frontier
    
    sub = G.subgraph(nodes)
    lines = [f"({s}) --[{d['predicate']}]--> ({t})" 
             for s, t, d in sub.edges(data=True)]
    return "\n".join(sorted(set(lines)))

# Prompt bắt buộc cite edges:
QUERY_PROMPT = """Answer using ONLY the knowledge graph below. 
Cite the specific edges that support each claim.
If the graph does not contain enough information, say so explicitly.

<graph>
{subgraph}
</graph>

Question: {question}"""
```

**Tại sao "say so explicitly" quan trọng?**

Không có câu này → LLM fallback về pretraining khi graph không đủ → ungrounded answer → hallucination. Phải hard-stop: "The knowledge graph does not contain information about X."

---

## 6. Evaluation — Đo chất lượng extraction

### Vấn đề: Không có feedback loop → không biết prompt change có cải thiện không

**Giải pháp: Gold set + alias map + Precision/Recall/F1**

```python
def prf(predicted: set, gold: set) -> tuple[float, float, float]:
    tp = len(predicted & gold)
    p  = tp / len(predicted) if predicted else 0.0
    r  = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1
```

**alias_map.json — tại sao cần?**
```json
{
  "neil armstrong": "neil alden armstrong",
  "the moon": "moon",
  "nasa": "nasa"
}
```

Sau resolution, canonical form có thể verbose hơn gold ("Neil Alden Armstrong" vs "Neil Armstrong"). Không có alias_map → cả 2 không match → recall giảm giả tạo. Đây là **scoring artifact**, không phải resolver bug.

**Rule:** Update alias_map.json mỗi khi canonical form mới xuất hiện sau resolution.

**Đo 2 lần song song:**
```
Raw extraction F1    → chất lượng extraction prompt
Resolved recall      → chất lượng resolution step
```

Nếu resolved recall < raw recall → resolver đang over-normalize một số entity, canonical form không match gold.

---

## 7. Scale — Production considerations

### Incremental update (đúng cách)

```
Document mới đến:
  [1] Extract entities + relations từ document mới
  [2] Resolve entities mới against EXISTING canonical set
      (không resolve against each other)
  [3] Add edges mới vào graph
  [4] Re-summarize ONLY nodes có source_docs thay đổi
  
KHÔNG rebuild toàn bộ graph từ đầu
```

### Extraction cost tối ưu

```python
# Prompt caching: cache schema + instructions, pay full chỉ cho document text
# Message Batches API: 50% off cho jobs chờ được 24h
# Haiku cho extraction (volume cao) → Sonnet cho resolution/summarization (quality cần cao)
```

### Storage theo corpus size

```
< 100k edges    → NetworkX in-memory
> 100k edges    → 3 PostgreSQL tables:
    entities(id, name, type, summary)
    relations(source_id, target_id, predicate)  
    aliases(entity_id, alias)
    
    Hoặc Neo4j / Neptune nếu cần graph traversal queries phức tạp
```

**Extraction và resolution code không thay đổi** khi switch storage — chỉ persistence layer thay đổi.

---

## Tóm tắt theo luồng

```
[Extraction]
  └─ Structured output thay NER truyền thống
  └─ description bắt buộc cho mỗi entity
  └─ Model nhỏ (Haiku) đủ dùng nếu schema rõ ràng
  └─ BẮT BUỘC: model đủ lớn (không dùng nano)

[Resolution]
  └─ LLM clustering với description context
  └─ Fallback: unmatched name → single-element cluster
  └─ Spot-check over-merging
  └─ Scale: block entities trước khi resolve

[Assembly]
  └─ MultiDiGraph
  └─ Rewrite endpoints về canonical
  └─ Guard: skip if src/tgt không có canonical
  └─ Sanity check: 1 connected component

[Summarization]
  └─ Pool all mentions + graph neighborhood
  └─ Re-summarize chỉ khi source_docs thay đổi

[Querying]
  └─ Serialize 2-hop subgraph
  └─ Hard-constrain: "answer ONLY from graph, cite edges"
  └─ Explicit "say so" khi graph không đủ

[Evaluation]
  └─ Gold set + alias_map.json
  └─ Đo raw vs resolved recall song song
  └─ Chạy sau mỗi prompt change

[Scale]
  └─ Incremental: resolve new against existing canonical
  └─ Prompt caching + Batch API cho cost
  └─ Switch storage không ảnh hưởng extraction/resolution code
```