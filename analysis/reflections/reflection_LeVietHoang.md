# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Lê Việt Hoàng  
**Mã số học viên:** 2A202602596  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

Bảng đối chiếu giữa các khái niệm lý thuyết cốt lõi trong bài giảng Production RAG và các hàm/module cụ thể đã triển khai trong bài thực hành:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích chuyên sâu |
|----------------|--------|-------------|------------------------------------|
| **Advanced Chunking (Semantic, Hierarchical, Structure-Aware)** | M1: Chunking | `chunk_semantic()`, `chunk_hierarchical()`, `chunk_structure_aware()` | - Basic chunking cắt đoạn theo ký tự hoặc `\n\n` cố định dẫn tới đứt câu và mất liên kết bảng biểu/tiêu đề.<br>- `chunk_semantic()` dùng cosine similarity giữa embedding của các câu kế tiếp (ngưỡng 0.85) để tách đoạn theo biến chuyển ý nghĩa.<br>- `chunk_hierarchical()` tạo quan hệ Cha (2048 ký tự) - Con (256 ký tự). Khi tìm kiếm, truy vấn so khớp trên chunk Con để đạt độ chính xác từ khóa cao, nhưng trả về bối cảnh Cha cho LLM đọc để không mất ngữ cảnh.<br>- `chunk_structure_aware()` phân tách theo Header Markdown (`#`, `##`, `###`), bảo toàn bảng biểu quy định và đưa header vào `section` metadata. |
| **Hybrid Search (Lexical + Dense Fusion)** | M2: Search | `segment_vietnamese()`, `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | - Dense search (BAAI/bge-m3 qua Qdrant `query_points`) hiểu tốt ngữ nghĩa câu hỏi tự nhiên nhưng dễ trượt các từ khóa chính xác (mã số, số ngày, tên điều khoản).<br>- BM25 (BM25Okapi) bắt từ khóa chính xác nhưng trong tiếng Việt cần tách từ bằng `underthesea` và thay thế ký tự `_` thành khoảng trắng để khớp với câu hỏi người dùng gõ.<br>- Thuật toán RRF ($RRF\_Score(d) = \sum \frac{1}{k + rank + 1}$) kết hợp thứ hạng từ 2 nguồn mà không cần chuẩn hóa thang đo điểm số khác biệt giữa cosine và BM25. |
| **Cross-Encoder Reranking** | M3: Reranking | `CrossEncoderReranker.rerank()`, `_load_model()` | - Tầng Hybrid Search sàng lọc 20 ứng viên tiềm năng bằng Bi-Encoder (nhanh nhưng độc lập câu hỏi và văn bản).<br>- Cross-Encoder (`BAAI/bge-reranker-v2-m3`) nhận đồng thời cặp `(query, document)` qua cơ chế self-attention đầy đủ của Transformer, nắm bắt quan hệ logic sâu sắc giữa câu hỏi và văn bản, sắp xếp lại và trích xuất top-3 chính xác nhất.<br>- Giảm thiểu số lượng tokens đưa vào prompt của LLM, giảm độ trễ tạo sinh và hiện tượng "lost in the middle". |
| **Automated Evaluation with RAGAS & Diagnostic Tree** | M4: Evaluation | `evaluate_ragas()`, `failure_analysis()` | - Đánh giá tự động hệ thống RAG qua bộ 4 chỉ số chuẩn công nghiệp: Faithfulness, Answer Relevancy, Context Precision, Context Recall.<br>- Tích hợp cây phân loại lỗi (Diagnostic Tree) để tự động ánh xạ chỉ số có điểm thấp nhất về nguyên nhân gốc rễ (ví dụ: Context Recall thấp $\rightarrow$ lỗi ở retrieval/chunking; Faithfulness thấp $\rightarrow$ LLM hallucination $\rightarrow$ siết prompt/temperature). |
| **Contextual Enrichment & HyQA** | M5: Enrichment | `contextual_prepend()`, `generate_hypothesis_questions()`, `_enrich_single_call()` | - Đoạn văn khi cắt nhỏ thường mất ngữ cảnh tài liệu gốc. Kỹ thuật Contextual Prepend bổ sung 1 câu tóm tắt vị trí và chủ đề vào đầu chunk trước khi embedding, giúp giảm tỷ lệ trượt tài liệu.<br>- Kỹ thuật HyQA sinh câu hỏi giả định mà đoạn văn có thể giải đáp, thu hẹp khoảng cách từ vựng giữa câu hỏi người dùng và văn bản quy chế.<br>- Áp dụng chế độ gom nhóm `_enrich_single_call()` (1 API call/chunk) thực hiện đồng thời summarize, hyqa, contextual prepend và trích xuất metadata để tối ưu chi phí và độ trễ. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

### 1. Lỗi xung đột Tokenizer và Crash thư viện FlagEmbedding
- **Exact error message:**  
  `ImportError / AttributeError: 'XLMRobertaTokenizerFast' object has no attribute 'sp_model'` hoặc lỗi crash segmentation khi nạp `FlagReranker` trên phiên bản transformers mới.
- **Nguyên nhân gốc rễ & Cách debug:**  
  Gói `FlagEmbedding` phiên bản cũ có xung đột tương thích với các phiên bản `transformers >= 4.40` và tokenizer của HuggingFace trên Windows.
- **Cách giải quyết:**  
  Tuân theo khuyến nghị chuẩn, thay thế hoàn toàn `FlagReranker` bằng lớp `CrossEncoder` từ thư viện `sentence_transformers`:
  ```python
  from sentence_transformers import CrossEncoder
  self._model = CrossEncoder("BAAI/bge-reranker-v2-m3")
  ```
  Cách này chạy ổn định, native và tối ưu hóa tốt trên môi trường PyTorch/HuggingFace hiện đại.

### 2. Lỗi cú pháp phương thức tìm kiếm trên Qdrant Client 1.9+
- **Exact error message:**  
  `AttributeError: 'QdrantClient' object has no attribute 'search'` hoặc cảnh báo deprecation khi dùng `client.search()`.
- **Nguyên nhân gốc rễ & Cách debug:**  
  Trong thư viện `qdrant-client` các phiên bản gần đây (v1.9 trở lên), API truy vấn vector đã được chuẩn hóa sang phương thức `query_points()`.
- **Cách giải quyết:**  
  Cập nhật mã gọi truy vấn trong `DenseSearch.search()`:
  ```python
  response = self.client.query_points(collection, query=query_vector, limit=top_k)
  ```
  Trích xuất dữ liệu an toàn từ `response.points` với `pt.payload` và `pt.score`.

### 3. Tối ưu hóa thời gian chạy Test Suite tránh lỗi Timeout 120s trong `check_lab.py`
- **Exact error message:**  
  `Command '... pytest tests/ ...' timed out after 120 seconds`
- **Nguyên nhân gốc rễ & Cách debug:**  
  Trong `tests/test_m3.py`, mỗi test case đều khởi tạo lại một đối tượng `CrossEncoderReranker()` mới, dẫn đến việc mô hình `BAAI/bge-reranker-v2-m3` dung lượng 2.2GB bị nạp đi nạp lại 5 lần vào RAM trên CPU, mất hơn 60 giây. Tương tự, `bge-m3` cũng bị nạp nhiều lần.
- **Cách giải quyết:**  
  Áp dụng mô hình Singleton / Module-level caching (`_shared_cross_encoder` và `_shared_bge_m3`). Mô hình chỉ được nạp đúng một lần đầu tiên và tái sử dụng cho tất cả các bài test và truy vấn tiếp theo. Nhờ đó, tổng thời gian chạy toàn bộ 37 unit tests giảm từ hơn 180s xuống còn dưới 30s, vượt qua 100% kiểm tra tự động.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Hệ thống Trợ lý Pháp lý & Quy chế Doanh nghiệp Nội bộ (Legal & Compliance Enterprise AI Assistant)

#### 1. Hiện trạng
- **Pipeline hiện tại:**  
  Sử dụng Naive RAG cơ bản: Cắt đoạn theo độ dài ký tự cố định (500 ký tự với overlap 50 ký tự), chỉ dùng Dense Search đơn thuần với OpenAI `text-embedding-3-small`, không có reranking và chưa có enrichment.
- **Vấn đề / Bottlenecks đang gặp:**  
  1. *Retrieval Failure:* Khi nhân viên hỏi các điều khoản có số hiệu cụ thể (ví dụ: "Theo Điều 15 Nghị định 13/2023...") hoặc các con số chính xác (số ngày phép, mức chi phí phụ cấp), Dense search thường trả về các điều khoản chung chung không chứa số liệu.
  2. *Context Fragmentation:* Việc cắt ngang 500 ký tự làm vỡ các điều khoản luật nhiều mục con, dẫn đến câu trả lời thiếu điều kiện loại trừ hoặc mất định nghĩa ban đầu.
  3. *Version Conflict:* Khi có tài liệu quy chế sửa đổi năm 2024 thay thế bản 2023, hệ thống thường trích xuất lẫn lộn cả văn bản cũ đã hết hiệu lực.

#### 2. Kế hoạch cải tiến kỹ thuật (Áp dụng các Module từ Lab 18)
1. **Chunking Strategy:**  
   - Chuyển đổi sang **Hierarchical Chunking (Parent-Child)** kết hợp **Structure-Aware Chunking**.
   - Phân tích cấu trúc văn bản quy phạm pháp luật theo Chương - Điều - Khoản - Điểm. Mỗi "Điều" đóng vai trò là một Parent Chunk (chứa toàn bộ nội dung quy định đầy đủ), và mỗi "Khoản/Điểm" là Child Chunk phục vụ vector indexing.
2. **Search Retrieval:**  
   - Triển khai **Hybrid Search (BM25 + Dense + RRF)**.
   - Sử dụng BM25 với bộ tiền xử lý tiếng Việt `underthesea` để bắt trúng các thực thể số hiệu văn bản, từ viết tắt và ngày tháng; kết hợp với mô hình đa ngôn ngữ `BAAI/bge-m3` trên Qdrant để hiểu ngữ nghĩa câu hỏi tự nhiên.
3. **Reranking:**  
   - Tích hợp tầng **Cross-Encoder Reranker** (`BAAI/bge-reranker-v2-m3`).
   - Lọc từ top-25 ứng viên của Hybrid Search xuống top-3 đoạn trích chính xác nhất trước khi gửi vào LLM synthesis.
4. **Data Enrichment:**  
   - Áp dụng kỹ thuật **Contextual Prepend** và trích xuất siêu dữ liệu (Metadata Extraction: ngày ban hành, phiên bản, trạng thái hiệu lực: active/deprecated).
   - Thiết lập bộ lọc Metadata Filtering trước khi tìm kiếm để loại bỏ hoàn toàn các quy chế đã hết hiệu lực khi câu hỏi hỏi về quy định hiện hành.
5. **Continuous Evaluation:**  
   - Thiết lập pipeline đánh giá tự động bằng **RAGAS** chạy định kỳ mỗi tuần hoặc khi có văn bản mới được nạp vào hệ thống.
   - Giám sát 4 chỉ số để tự động phát hiện suy giảm chất lượng dữ liệu: Faithfulness $\ge 0.85$, Context Precision $\ge 0.80$, Context Recall $\ge 0.80$, Answer Relevancy $\ge 0.85$.

#### 3. Timeline triển khai (4 tuần)
- **Tuần 1:** Xây dựng parser văn bản pháp luật, triển khai Structure-Aware & Hierarchical Chunking; tạo metadata về ngày hiệu lực và phiên bản tài liệu.
- **Tuần 2:** Thiết lập cơ sở dữ liệu vector Qdrant, nạp vector BGE-M3 và dựng chỉ mục BM25 tiếng Việt; tích hợp thuật toán RRF.
- **Tuần 3:** Tích hợp tầng Cross-Encoder Reranker, tinh chỉnh system prompt cho LLM để trích dẫn chính xác số điều luật và không sinh ảo giác.
- **Tuần 4:** Xây dựng bộ test-set gồm 50 câu hỏi nghiệp vụ thực tế; chạy đánh giá định lượng bằng RAGAS, đối chiếu với Naive RAG cũ và đóng gói đưa vào môi trường staging.
