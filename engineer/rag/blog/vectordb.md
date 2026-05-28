# How VectorDBs Work Internally + How To Make The Most Out Of Them

List of Contents
- The Naive Approach – Brute-force search limitations explained clearly
- Distance Concentration in High Dimensions – Why high-dimensional spaces behave strangely + efficient alternatives to linear search
- HNSW Algorithm – Navigable graphs for fast search
- Other Indexing Strategies – IVF, PQ, LSH tradeoffs explored
- Tradeoffs in Vector Databases – Balancing recall, speed, memory usage
- RAG Patterns and VectorDBs – How different pipelines use databases
- Production Realities – Filtering, updates, sharding, latency challenges
- Implications for Applications – Designing systems around query patterns



## 1. HNSW (Hệ thống Thế giới Nhỏ Có Thể Điều Hướng Theo Cấp Bậc)
HNSW là thuật toán hỗ trợ hầu hết các cơ sở dữ liệu vector hiện đại, và nó dựa trên một nhận định xuất sắc: tìm kiếm trong không gian đa chiều giống như điều hướng trong một mạng xã hội.

**Cách thức hoạt động của các lớp**

HNSW xây dựng nhiều lớp đồ thị lân cận. Mỗi lớp là một đồ thị trong đó các nút là các vectơ và các cạnh kết nối các vectơ tương tự. Các thuộc tính chính:
- Lớp 0 (dưới cùng): Chứa tất cả các vectơ, được kết nối chặt chẽ.
- Các lớp cao hơn : Chứa số lượng vectơ ít hơn theo cấp số mũ, hoạt động như "làn đường ưu tiên".
- Mỗi vectơ xuất hiện ở lớp 0 và có thể ở các lớp cao hơn (được chọn theo xác suất).

Khi bạn chèn một vectơ, thuật toán sẽ gán ngẫu nhiên cho nó một lớp (sử dụng phân phối suy giảm theo hàm mũ — hầu hết các vectơ nằm ở lớp 0, chỉ một số ít đạt đến các lớp cao hơn). Sau đó, nó kết nối vectơ đó với các láng giềng gần nhất trong mỗi lớp mà nó chiếm giữ.


