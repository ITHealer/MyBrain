---
modified: 2026-05-27T06:47:25.999Z
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


## Bản tổng hợp từ nhiều nguồn
# Tài liệu kỹ thuật: Xây dựng Knowledge Graph cho Agentic RAG

---

## Pipeline tổng quan

```
Documents
    ↓
[1] Chunking
    ↓ — checkpoint —
[2] LLM Extraction          ← thay thế NER + Relation Classifier
    ↓ — checkpoint —
[3] Entity Resolution       ← LLM clustering dùng description
    ↓                         "gọi entity này là gì?" — KHÔNG merge
[4] Entity Deduplication    ← embedding similarity trên full context
    ↓                         "đây có phải cùng entity không?" — merge ở đây
[5] Graph Assembly
    ↓
[6] Hub Summarization
    ↓
[7] Querying
```

**Ba nguyên tắc không được bỏ qua:**
- Resolution và Deduplication là **hai bước khác nhau hoàn toàn** — đây là sai lầm phổ biến nhất
- False merge **corrupt graph silently** — không có error log, query trả kết quả sai mà không biết
- **Defensive default:** khi không chắc → tạo node mới, không merge

---

## Bước 1 — Chunking

**Mục tiêu:** Chia document thành đơn vị xử lý phù hợp cho extraction.

**Vấn đề:**
- Chunk quá lớn → LLM extract quá nhiều incidental entities → graph nhiễu
- Chunk quá nhỏ → relation bị cắt đứt giữa chừng → thiếu context

**Giải pháp:** 512–1024 tokens per chunk, split theo semantic boundary (headers). Mỗi chunk giữ `document_id` để track provenance.

```python
def prepare_chunks(document: dict) -> list[dict]:
    chunks = chunk_by_headers(document["text"])  # header-based split
    return [
        {
            "chunk_id": f"{document['id']}_chunk_{i}",
            "document_id": document["id"],
            "document_title": document["title"],
            "text": chunk,
        }
        for i, chunk in enumerate(chunks)
    ]
```

**Rule cứng:** Một chunk không được span 2 documents. Chunk per-document trước, process sau.

---

## Bước 2 — LLM Extraction

### Bối cảnh: Tại sao cần thay thế pipeline truyền thống?

Pipeline truyền thống cần **hai model riêng biệt**, cả hai đều cần labeled training data:

```
[NER model]              → tag entity spans: (PERSON, ORG, LOC...)
        ↓
[Relation Classifier]    → classify pairs of spans thành relation types
                           VD: (Neil Armstrong, Apollo 11) → "commanded"
```

Khi domain thay đổi (y tế → tài chính), phải train lại cả hai. Chi phí cao, không linh hoạt.

**Giải pháp của cookbook:** Một LLM call với Pydantic schema **thay thế hoàn toàn cả NER lẫn Relation Classifier**. Schema là "training" — không cần labeled data.

### Implementation

```python
from pydantic import BaseModel
from typing import Literal

EntityType = Literal["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "ARTIFACT"]

class Entity(BaseModel):
    name: str
    type: EntityType
    description: str  # BẮT BUỘC: 1 câu grounded trong document này
                      # Thiếu field này → Resolution bước 3 sẽ sai

class Relation(BaseModel):
    source: str       # tên entity đã extract trong chunk này
    predicate: str    # short verb phrase: "commands", "treats", "founded"
    target: str       # tên entity đã extract trong chunk này

class ExtractedGraph(BaseModel):
    entities: list[Entity]
    relations: list[Relation]

EXTRACTION_PROMPT = """Extract a knowledge graph from the document below.

<document>
{text}
</document>

Rules:
- Extract ONLY entities central to this document. Skip incidental mentions.
- Each entity MUST have a 1-sentence description grounded in THIS document.
  This description will be used later to distinguish entities with similar names.
- Predicates must be short verb phrases: "commands", "treats", "founded by".
- Every relation must connect two entities you extracted above.
  Do NOT create relations to entities not in your list."""

def extract(chunk: dict) -> ExtractedGraph | None:
    result = client.messages.parse(
        model=KG_EXTRACTION_MODEL,  # KHÔNG dùng nano — xem issue bên dưới
        messages=[{"role": "user",
                   "content": EXTRACTION_PROMPT.format(text=chunk["text"])}],
        output_format=ExtractedGraph,
    )
    if len(result.entities) == 0:
        logger.warning(f"0 entities from chunk {chunk['chunk_id']}")
        return None
    return result
```

**Tại sao `description` bắt buộc ngay tại đây?**

Bước 3 (Resolution) dùng LLM để cluster entities. LLM cần nhìn thấy description để phân biệt:
- `"Armstrong: first human to walk on the Moon"` vs `"Armstrong: jazz trumpeter"`
- Không có description → LLM không có context → cluster sai

### Issue: Model nhỏ không follow JSON schema → 0 entities, silent failure

```
Log thực tế:
KG extraction produced 0 entities.
Model 'gpt-4.1-nano' may not support entity extraction format.
```

Pipeline chạy thành công nhưng graph rỗng — không có error, không có warning rõ ràng.

**Giải pháp:** Tách model config cho KG extraction, validate output:

```python
# config
LLM_CHAT_MODEL          = "gpt-4.1-nano"    # OK cho chat, rẻ
LLM_KG_EXTRACTION_MODEL = "gpt-4.1-mini"    # riêng cho KG extraction

# Model đủ lớn cho KG extraction:
# ✅ gpt-4.1-mini, gpt-4.1, claude-haiku-4-5, gemini-flash, qwen3:14b
# ❌ nano-class models
```

### Issue: Fail giữa chừng → tốn lại toàn bộ token

Extraction là bước tốn tiền nhất. Nếu fail ở chunk 500/1000, không có checkpoint → chạy lại từ đầu.

**Giải pháp:** Checkpoint per chunk:

```python
def extract_with_checkpoint(chunks: list[dict], checkpoint_dir: str) -> list[dict]:
    Path(checkpoint_dir).mkdir(exist_ok=True)
    results = []

    for chunk in chunks:
        checkpoint_file = Path(checkpoint_dir) / f"{chunk['chunk_id']}.json"

        if checkpoint_file.exists():            # đã có → skip, không gọi LLM lại
            results.append(json.load(open(checkpoint_file)))
            continue

        result = extract(chunk)
        if result:
            data = {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "document_title": chunk["document_title"],
                "entities": [e.model_dump() for e in result.entities],
                "relations": [r.model_dump() for r in result.relations],
            }
            checkpoint_file.write_text(json.dumps(data))
            results.append(data)

    return results
```

---

## Bước 3 — Entity Resolution

> **Định nghĩa chính xác:** "Gọi entity này là gì?" — cập nhật canonical name. **KHÔNG merge nodes.**

### Tại sao string similarity không đủ?

```
"Edwin Aldrin" vs "Buzz Aldrin"
→ edit distance: rất thấp (khác hoàn toàn)
→ Jaccard on tokens: 0 (không có token chung)
→ string similarity: FAIL

→ Nhưng descriptions nói: cả hai đều là
  "Lunar Module Pilot on Apollo 11, second human on Moon"
→ LLM đọc description → hiểu đây là cùng người
```

**Đây là lý do cookbook dùng LLM clustering với description context** — không phải embedding hay fuzzy matching ở bước này.

### Implementation — LLM clustering với descriptions

```python
from pydantic import BaseModel

class Cluster(BaseModel):
    canonical: str        # most complete, unambiguous form
    aliases: list[str]    # tất cả surface forms map về canonical này

class ResolvedClusters(BaseModel):
    clusters: list[Cluster]

RESOLVE_PROMPT = """Below are {entity_type} entities extracted from 
several documents. Some are different surface forms of the same 
real-world entity.

<entities>
{entity_list}
</entities>

Rules:
- Each input name must appear in exactly one cluster's aliases list.
- Entities genuinely distinct → own single-element cluster.
- Use the descriptions to avoid merging entities sharing only a name.
  Example: "Apple (technology company)" ≠ "Apple (fruit)"
- Canonical = most complete, unambiguous form.
- Do NOT merge based on name similarity alone."""

def resolve(entity_type: str, entities: list[dict]) -> list[Cluster]:
    # Dedup by name trước khi gửi LLM
    unique = {}
    for e in entities:
        unique.setdefault(e["name"], e["description"])

    entity_list = "\n".join(
        f"- {name}: {desc}"
        for name, desc in unique.items()
    )

    response = client.messages.parse(
        model=SYNTHESIS_MODEL,   # cần model lớn hơn extraction
        messages=[{"role": "user",
                   "content": RESOLVE_PROMPT.format(
                       entity_type=entity_type,
                       entity_list=entity_list,
                   )}],
        output_format=ResolvedClusters,
    )
    return response.parsed_output.clusters
```

**Lý do chỉ cluster entities cùng type:** "Apple" (ORGANIZATION) không thể match "Apple" (ARTIFACT). Type filtering là guard đầu tiên chống false resolution.

### Issue: Entity bị drop silently

LLM có thể bỏ sót một entity không assign vào cluster nào → `alias_to_canonical` không có key → node biến mất hoàn toàn.

**Giải pháp: Mandatory fallback**

```python
def resolve_with_fallback(entity_type: str,
                          entities: list[dict]) -> dict[str, str]:
    """Returns: alias → canonical map"""
    try:
        clusters = resolve(entity_type, entities)
    except Exception:
        # Hard fallback: mỗi name là cluster riêng
        return {e["name"]: e["name"] for e in entities}

    alias_to_canonical = {}
    for cluster in clusters:
        for alias in cluster.aliases:
            alias_to_canonical[alias] = cluster.canonical

    # Verify không có entity nào bị drop
    all_input = {e["name"] for e in entities}
    all_clustered = set(alias_to_canonical.keys())
    dropped = all_input - all_clustered

    for name in dropped:
        # Single-element cluster cho entity bị bỏ sót
        alias_to_canonical[name] = name
        logger.warning(f"Entity '{name}' not clustered, added as standalone")

    return alias_to_canonical
```

### Issue: Over-merging — "Gemini 12" bị fold vào "Project Gemini"

LLM thấy descriptions overlap → merge nhầm specific mission với broad program.

**Giải pháp:** Extraction description phải đủ specific:
- Sai: `"Gemini 12: a NASA mission"`
- Đúng: `"Gemini 12: Buzz Aldrin's final spaceflight before Apollo 11, last Gemini mission"`

### Issue: Scale — 10,000 entities không fit 1 prompt

**Giải pháp: Block trước bằng cheap signals, LLM arbitrate trong block nhỏ**

```python
def block_and_resolve(entities: list[dict],
                      entity_type: str) -> dict[str, str]:
    """
    Block theo cheap signals trước:
    1. Same last name (PERSON)
    2. Token overlap > 0.3 (Jaccard)
    3. Embedding cosine > 0.85
    
    Mỗi block: 50-100 entities → gửi LLM
    LLM KHÔNG biết entities ở block khác → không cross-block merge
    Target block size: 50-100 entities
    """
    blocks = create_blocks(entities)  # cheap grouping
    alias_to_canonical = {}

    for block in blocks:
        block_result = resolve_with_fallback(entity_type, block)
        alias_to_canonical.update(block_result)

    return alias_to_canonical
```

**Quan trọng:** Resolution chỉ cập nhật canonical name để dùng cho soft matching. **Không merge nodes. Không thay đổi graph.**

---

## Bước 4 — Entity Deduplication

> **Định nghĩa chính xác:** "Đây có phải cùng entity thật không?" — đây là bước duy nhất quyết định merge.

### Tại sao Resolution không đủ để merge?

```
Resolution xong ta biết:
"Edwin Aldrin" và "Buzz Aldrin" → canonical = "Buzz Aldrin"

Nhưng canonical name giống nhau KHÔNG chứng minh cùng entity:
"Jensen Huang" (CEO NVIDIA) vs "Jensen Huang" (bác sĩ ở Đài Loan)
→ Cùng canonical name sau resolution
→ KHÔNG phải cùng người
→ Merge sai → toàn bộ NVIDIA relations gán cho bác sĩ → graph corrupt silently
```

**Nguyên tắc từ image:** `Evidence strength = Permission strength`

```
≥ 0.95 → auto-merge       (strong evidence, đủ tin)
> 0.85 → flag for review   (uncertain, cần human)
≤ 0.85 → new node          (weak evidence, defensive default)
```

### Implementation — 3-tier confidence scoring trên full context

```python
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz

encoder = SentenceTransformer("BAAI/bge-m3")

def build_entity_context(entity: dict) -> str:
    """
    Full context = name + type + description + relations
    Không dùng chỉ name hay chỉ description
    """
    relations_str = "; ".join(entity.get("relations", []))
    return (
        f"Name: {entity['name']}. "
        f"Type: {entity['type']}. "
        f"Description: {entity['description']}. "
        f"Known relations: {relations_str}"
    )

def compute_similarity(new_entity: dict,
                       existing_node: dict) -> float:
    new_ctx = build_entity_context(new_entity)
    existing_ctx = build_entity_context(existing_node)

    # Semantic similarity trên full context (70% weight)
    new_emb = encoder.encode(new_ctx, normalize_embeddings=True)
    existing_emb = encoder.encode(existing_ctx, normalize_embeddings=True)
    semantic_score = float(new_emb @ existing_emb)

    # Fuzzy similarity trên name (30% weight — supporting signal)
    name_score = fuzz.ratio(
        new_entity["name"].lower(),
        existing_node["name"].lower()
    ) / 100

    return 0.7 * semantic_score + 0.3 * name_score

def deduplicate(new_entity: dict,
                same_type_nodes: list[dict]) -> dict:
    """
    Returns one of:
    {"action": "new_node"}
    {"action": "merge", "target_id": str}
    {"action": "flag_review", "candidate_id": str, "score": float}
    """
    if not same_type_nodes:
        return {"action": "new_node"}

    scores = [
        (node, compute_similarity(new_entity, node))
        for node in same_type_nodes
    ]
    best_node, best_score = max(scores, key=lambda x: x[1])

    if best_score >= 0.95:
        return {"action": "merge", "target_id": best_node["id"]}
    elif best_score > 0.85:
        return {"action": "flag_review",
                "candidate_id": best_node["id"],
                "score": best_score}
    else:
        return {"action": "new_node"}  # defensive default
```

### Issue: Embedding là bước đắt — cần cache

```python
import hashlib
import numpy as np

def embed_with_cache(entity: dict, cache_dir: str) -> np.ndarray:
    # Hash theo content, không phải name
    # Cùng name, khác description → hash khác → embed lại đúng
    content = f"{entity['name']}|{entity['description']}"
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    cache_file = Path(cache_dir) / f"{content_hash}.npy"

    if cache_file.exists():
        return np.load(cache_file)

    emb = encoder.encode(content, normalize_embeddings=True)
    np.save(cache_file, emb)
    return emb
```

---

## Bước 5 — Graph Assembly

**Mục tiêu:** Dùng `alias_to_canonical` map từ Resolution để rewrite relation endpoints, load vào graph.

```python
import networkx as nx

def assemble_graph(canonical_entities: dict,
                   raw_relations: list[dict],
                   alias_to_canonical: dict) -> nx.MultiDiGraph:
    """
    MultiDiGraph vì:
    - Multi: 2 nodes có nhiều predicates khác nhau
      VD: Drug → Condition: "treats" VÀ "causes side effect"
    - Directed: quan hệ có chiều
      "Armstrong commanded Apollo 11" ≠ "Apollo 11 commanded Armstrong"
    """
    G = nx.MultiDiGraph()

    for canonical_name, info in canonical_entities.items():
        G.add_node(canonical_name,
                   type=info["type"],
                   description=info["description"],
                   source_docs=[],
                   mentions=0)

    stats = {"added": 0, "skipped": 0}

    for rel in raw_relations:
        src = alias_to_canonical.get(rel["source"])
        tgt = alias_to_canonical.get(rel["target"])

        # Guard 1: endpoint không có canonical → skip
        if not src or not tgt:
            stats["skipped"] += 1
            continue

        # Guard 2: self-loop → skip
        if src == tgt:
            stats["skipped"] += 1
            continue

        G.add_edge(src, tgt,
                   predicate=rel["predicate"],
                   source_doc=rel["source_doc"])
        stats["added"] += 1

    # Sanity check sau khi build
    components = nx.number_weakly_connected_components(G)
    if components > 3:
        logger.warning(
            f"Graph has {components} connected components — "
            "likely indicates entity resolution gaps. Inspect isolated nodes."
        )

    logger.info(f"Assembly: {stats}")
    return G
```

**Fragmented graph (nhiều components)** = dấu hiệu Resolution bị miss — có entity variants chưa được cluster đúng.

---

## Bước 6 — Hub Node Summarization

**Mục tiêu:** Hub nodes (degree cao) xuất hiện trong nhiều documents. Node description chỉ lưu context từ 1 document đầu — không đủ.

```python
SUMMARIZE_PROMPT = """Generate a profile for this entity.

Entity: {name} ({type})

All source excerpts:
{excerpts}

Known graph relations:
{relations}

Output:
- summary: 2-3 paragraphs synthesized from excerpts.
  Resolve contradictions by preferring the most specific claim.
- key_facts: 3-5 atomic facts, each traceable to a specific source.
- time_range: YYYY format. "unknown" or "ongoing" where appropriate.

Do NOT invent facts not supported by the excerpts."""

def summarize_hub_nodes(G: nx.MultiDiGraph,
                        documents: list[dict],
                        min_degree: int = 5):
    for node in G.nodes:
        if G.degree(node) < min_degree:
            continue  # chỉ summarize hub nodes

        source_docs = G.nodes[node]["source_docs"]
        excerpts = "\n\n".join(
            f"[{d['title']}]\n{d['text']}"
            for d in documents
            if d["title"] in source_docs
        )
        relations = "\n".join(
            f"- {node} --{d['predicate']}--> {tgt}"
            for _, tgt, d in G.out_edges(node, data=True)
        )

        profile = client.messages.parse(
            model=SYNTHESIS_MODEL,
            messages=[{"role": "user",
                       "content": SUMMARIZE_PROMPT.format(
                           name=node,
                           type=G.nodes[node]["type"],
                           excerpts=excerpts,
                           relations=relations,
                       )}],
            output_format=EntityProfile,
        )
        G.nodes[node]["profile"] = profile.model_dump()
```

**Rule incremental:** Chỉ re-summarize khi `source_docs` thay đổi. Không re-summarize khi document mới không đề cập entity đó.

---

## Bước 7 — Querying với Multi-hop Reasoning

**Vấn đề:** LLM có pretraining knowledge về famous entities → trả lời "đúng" nhưng không traceable, không auditable. Với private corpus (hồ sơ bệnh nhân, tài liệu nội bộ) → hallucinate.

**Giải pháp:** Serialize subgraph → hard constraint prompt buộc cite edges.

```python
def serialize_subgraph(G: nx.MultiDiGraph,
                       center: str,
                       hops: int = 2) -> str:
    """
    2-hop đủ cho hầu hết queries.
    3-hop bắt đầu noisy và tốn token.
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
    lines = [
        f"({src}) --[{data['predicate']}]--> ({tgt})"
        for src, tgt, data in sub.edges(data=True)
    ]
    return "\n".join(sorted(set(lines)))

QUERY_PROMPT = """Answer using ONLY the knowledge graph below.
For every claim, cite the specific edge(s) supporting it.
If the graph does not contain sufficient information, say:
"The knowledge graph does not contain information about this."
Do NOT use external knowledge.

<graph>
{subgraph}
</graph>

Question: {question}"""
```

**Tại sao "say so explicitly" bắt buộc:** Không có câu này → LLM fill gap bằng pretraining silently → answer fluent nhưng không grounded → audit fail.

---

## Incremental Update

**Pattern sai:** Rebuild toàn bộ graph mỗi khi có document mới.

**Pattern đúng:**

```python
def ingest_new_document(doc: dict, G: nx.MultiDiGraph,
                        canonical_registry: dict) -> nx.MultiDiGraph:
    """
    Quy trình đúng:
    1. Extract từ document mới
    2. Resolve entities mới against EXISTING canonical set
       (không phải against each other)
    3. Deduplicate against existing graph nodes
    4. Add chỉ edges mới
    5. Re-summarize chỉ nodes có source_docs thay đổi
    """
    extracted = extract(doc)
    if not extracted:
        return G

    for entity in extracted.entities:
        # Resolve against existing canonical names của cùng type
        existing_names = list(canonical_registry.get(entity.type, set()))
        canonical = resolve_single_entity(entity.name, entity.description,
                                          existing_names)

        # Dedup against existing nodes cùng type
        same_type_nodes = [
            {"id": n, **G.nodes[n]}
            for n in G.nodes
            if G.nodes[n].get("type") == entity.type
        ]
        action = deduplicate(
            {"name": canonical, "type": entity.type,
             "description": entity.description},
            same_type_nodes
        )

        if action["action"] == "merge":
            target = action["target_id"]
            old_docs = set(G.nodes[target]["source_docs"])
            G.nodes[target]["source_docs"].append(doc["title"])
            G.nodes[target]["mentions"] += 1
            # Re-summarize chỉ khi source_docs thay đổi
            if set(G.nodes[target]["source_docs"]) != old_docs:
                G.nodes[target]["profile"] = summarize_entity(target, G)

        elif action["action"] == "new_node":
            G.add_node(canonical, type=entity.type,
                       description=entity.description,
                       source_docs=[doc["title"]], mentions=1)
            canonical_registry.setdefault(entity.type, set()).add(canonical)

        elif action["action"] == "flag_review":
            review_queue.append({
                "new_entity": entity.model_dump(),
                "candidate": action["candidate_id"],
                "score": action["score"],
                "source_doc": doc["title"],
            })

    return G
```

---

## Evaluation

```python
def evaluate(predicted_entities: set[str],
             gold_entities: set[str],
             alias_map: dict) -> dict:
    """
    alias_map.json bắt buộc — không có thì canonical form verbose
    ("Neil Alden Armstrong") không match gold ("Neil Armstrong")
    → recall giảm giả tạo → metrics misleading

    Đo 2 lần:
    - Raw F1: chất lượng extraction prompt
    - Resolved recall: chất lượng resolution step
    Nếu resolved recall < raw recall → resolver over-normalizing
    """
    def norm(name: str) -> str:
        return alias_map.get(name.lower().strip(), name.lower().strip())

    pred = {norm(e) for e in predicted_entities}
    gold = {norm(e) for e in gold_entities}

    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall    = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "missed": sorted(gold - pred),
    }
```

**Feedback loop bắt buộc:** Chạy eval sau mỗi thay đổi extraction prompt. Không có eval loop = không biết thay đổi có cải thiện hay không.

---

## Tổng hợp issues và giải pháp

| Bước | Issue | Hậu quả | Giải pháp |
|---|---|---|---|
| Extraction | Model nano không follow schema | 0 entities, silent | Tách `KG_EXTRACTION_MODEL`, validate output |
| Extraction | Không có `description` field | Resolution sai | Enforce trong Pydantic schema |
| Extraction | Không checkpoint | Tốn lại toàn bộ token | Checkpoint per chunk |
| Resolution | Entity bị drop | Node biến mất silently | Mandatory fallback single-element cluster |
| Resolution | Over-merging | Mất precision | Description đủ specific khi extract |
| Resolution | Scale > 1000 entities | Không fit 1 prompt | Block by cheap signals, 50-100 per block |
| Deduplication | False merge cùng tên khác người | Graph corrupt silently | 3-tier: 0.95 merge / 0.85 review / default new node |
| Deduplication | Embedding đắt, hay retry | Tốn tiền | Cache by content hash |
| Assembly | Fragmented graph | Multi-hop fail | Check components sau build, inspect orphans |
| Querying | LLM fallback pretraining | Ungrounded answer | Hard-constrain prompt + cite edges |
| Incremental | Rebuild toàn bộ | Không scale | Resolve new against existing canonical set |
| Evaluation | Canonical verbose không match gold | Metrics misleading | `alias_map.json` + update sau mỗi run |