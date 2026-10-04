# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Lê Việt Hoàng  
**Mã số học viên:** 2A202602596  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.6200 | 0.8850 | +0.2650 |
| Answer Relevancy | 0.7100 | 0.8400 | +0.1300 |
| Context Precision | 0.5400 | 0.8100 | +0.2700 |
| Context Recall | 0.6000 | 0.7900 | +0.1900 |

> **Nhận xét tổng quan:**  
> Hệ thống Production RAG đạt mức cải thiện vượt bậc trên toàn bộ 4 chỉ số đo lường so với Naive Baseline:
> - **Faithfulness (+0.2650):** Tăng từ 0.62 lên 0.8850 nhờ tầng Cross-Encoder Reranking chọn lọc chính xác top-3 đoạn trích đắt giá nhất, giảm thiểu tối đa hiện tượng LLM hallucination.
> - **Context Precision (+0.2700):** Tăng mạnh từ 0.54 lên 0.81 nhờ sự kết hợp giữa BM25 tiếng Việt và Dense vector qua thuật toán RRF, đẩy các tài liệu liên quan trực tiếp lên đầu bảng xếp hạng.
> - **Context Recall (+0.1900) & Answer Relevancy (+0.1300):** Cải thiện rõ rệt nhờ cấu trúc Hierarchical Chunking (Parent-Child) bảo toàn đầy đủ bối cảnh quy định cho mô hình ngôn ngữ.

---

## Bottom-5 Failures (Phân tích theo Diagnostic Tree)

### #1. Q17 — Tính phí phạt tạm ứng quá hạn
- **Question:** "Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?"
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** Thời hạn hoàn ứng là 15 ngày kể từ ngày kết thúc công việc. Quá hạn 5 ngày bị tính phí phạt chậm thanh toán 2%/tháng, tương ứng khoảng 50.000 VNĐ.
- **Worst metric:** `context_recall` (0.58)
- **Error Tree:** Output tính ra kết quả tương đối chính xác $\rightarrow$ Context trích xuất chưa đầy đủ công thức tính pro-rata chi tiết nằm ở phụ lục cuối văn bản $\rightarrow$ Query OK nhưng từ khóa "pro-rata" không xuất hiện trong câu hỏi người dùng $\rightarrow$ Missing relevant chunks.
- **Root cause:** Kỹ thuật cắt đoạn đã tách phần nguyên tắc tạm ứng ở thân bài khỏi bảng công thức tính phạt chi tiết ở phụ lục.
- **Suggested fix:** Cải thiện khâu M1 Chunking bằng Structure-Aware Chunking để gom phụ lục biểu phí kèm theo điều khoản quy định, hoặc sử dụng HyQA trong M5 để sinh câu hỏi giả định chứa từ khóa "phạt chậm hoàn ứng / pro-rata".

---

### #2. Q14 — Bồi hoàn chi phí tài trợ đào tạo
- **Question:** "Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?"
- **Expected:** Nhân viên phải cam kết làm việc ít nhất 1 năm sau khi hoàn thành khóa học. Nghỉ sau 8 tháng là trước hạn cam kết, phải hoàn trả 100% chi phí tức 25.000.000 VNĐ.
- **Got:** Do cam kết làm việc sau khóa học là tối thiểu 1 năm (12 tháng), nhân viên nghỉ việc sau 8 tháng phải bồi hoàn 100% chi phí tài trợ là 25.000.000 VNĐ.
- **Worst metric:** `context_recall` (0.60)
- **Error Tree:** Output đúng kết luận $\rightarrow$ Context trích xuất thiếu điều khoản phân cấp bồi hoàn theo mốc thời gian (dưới 1 năm vs 1-2 năm) $\rightarrow$ Retriever chỉ kéo được đoạn cam kết thời gian chung chung $\rightarrow$ Missing relevant chunks.
- **Root cause:** BM25 khớp mạnh với từ "tài trợ khóa học" nên kéo về các đoạn giới thiệu chương trình đào tạo thay vì quy chế cam kết tài chính.
- **Suggested fix:** Tăng `BM25_TOP_K` lên 30 và áp dụng kỹ thuật Contextual Prepend ở M5 nhằm gán rõ bối cảnh "Quy chế bồi hoàn chi phí đào tạo" vào đầu đoạn văn trước khi tạo embedding.

---

### #3. Q12 — Quyền lợi nhân viên Senior đa vế (Multi-hop Query)
- **Question:** "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Nhân viên Senior 9 năm thâm niên được hưởng 18 ngày phép năm (15 ngày cơ bản + 3 ngày thâm niên). Dải lương cấp bậc Senior từ 20 đến 35 triệu VNĐ/tháng.
- **Worst metric:** `context_recall` (0.62)
- **Error Tree:** Output tổng hợp đủ 2 vế $\rightarrow$ Context trích xuất chỉ có 1 đoạn về thang lương đứng ở rank 3, suýt bị cắt khỏi Top-3 $\rightarrow$ Single Query không bao quát tốt 2 chủ đề nằm ở 2 văn bản khác nhau (chính sách phép vs chính sách lương) $\rightarrow$ Missing relevant chunks.
- **Root cause:** Thử thách Multi-hop Reasoning: Một câu hỏi đơn lẻ đòi hỏi thông tin từ 2 miền nghiệp vụ khác biệt (Nghỉ phép và Lương thưởng).
- **Suggested fix:** Bổ sung bước Query Decomposition (Phân rã câu hỏi) trước tầng M2: tách thành 2 truy vấn con ("Quy chế ngày phép thâm niên Senior" và "Thang bảng lương Senior P3-P4") rồi gộp kết quả tìm kiếm lại.

---

### #4. Q7 — Xung đột phiên bản chu kỳ đổi mật khẩu
- **Question:** "Bao lâu phải đổi mật khẩu một lần?"
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải được thay đổi mỗi 120 ngày. Chính sách cũ yêu cầu 90 ngày nhưng đã bị thay thế.
- **Got:** Chính sách mật khẩu v2.0 quy định thời gian thay đổi mật khẩu là 120 ngày một lần, thay cho quy định cũ là 90 ngày.
- **Worst metric:** `context_precision` (0.65)
- **Error Tree:** Output đúng quy chế mới $\rightarrow$ Context trích xuất chứa cả đoạn văn của chính sách cũ v1.0 (90 ngày) ở vị trí cao $\rightarrow$ Câu hỏi không chỉ định năm/phiên bản $\rightarrow$ Too many irrelevant/outdated chunks.
- **Root cause:** Văn bản quy chế cũ và mới có từ vựng và cấu trúc tương đồng tới 95%, khiến cả BM25 và Dense Search đều chấm điểm cao cho cả hai phiên bản.
- **Suggested fix:** Triển khai Metadata Pre-filtering (lọc trường `status: "active"` hoặc `version: "2.0"` được trích xuất tự động từ M5) để loại bỏ văn bản hết hiệu lực trước khi xếp hạng.

---

### #5. Q4 — Xung đột phiên bản ngày phép năm (2023 vs 2024)
- **Question:** "Nhân viên được nghỉ bao nhiêu ngày phép năm?"
- **Expected:** Theo chính sách hiện hành (v2024), nhân viên được nghỉ 15 ngày phép năm có lương. Chính sách cũ (v2023) là 12 ngày nhưng đã bị thay thế.
- **Got:** Theo quy chế v2024 hiện hành, nhân viên được nghỉ 15 ngày phép năm có hưởng lương (trước đây theo quy chế 2023 là 12 ngày).
- **Worst metric:** `context_precision` (0.68)
- **Error Tree:** Output phân biệt được văn bản cũ và mới $\rightarrow$ Context trích xuất xếp đoạn văn bản 2023 (12 ngày) ở rank 2 $\rightarrow$ Too many irrelevant chunks.
- **Root cause:** Mô hình Bi-Encoder không có ý niệm về thời gian hiệu lực nếu không được cung cấp siêu dữ liệu ngày ban hành.
- **Suggested fix:** Thêm trọng số thời gian (Temporal Decay / Recency Re-weighting) trong thuật toán RRF hoặc nạp metadata `effective_date` vào Cross-Encoder để ưu tiên văn bản mới nhất.

---

## Case Study (Phân tích chuyên sâu cho Presentation)

**Question chọn phân tích:**  
> *"Nhân viên được nghỉ bao nhiêu ngày phép năm?"* (Test Question #4)

### Error Tree Walkthrough:
1. **Output đúng?** $\rightarrow$ **ĐÚNG**. Mô hình trả lời chính xác 15 ngày theo quy chế v2024 và lưu ý quy chế cũ 2023 (12 ngày) đã hết hiệu lực.
2. **Context đúng?** $\rightarrow$ **MỘT PHẦN (Precision thấp)**. Trong 3 đoạn trích gửi vào LLM, có 1 đoạn trích từ quy chế 2023 ghi "12 ngày phép năm" đứng ở rank 2, gây nguy cơ cao khiến LLM bị phân vân hoặc trả lời nhầm nếu nhiệt độ (temperature) lớn hơn 0.
3. **Query rewrite / Intent OK?** $\rightarrow$ **OK**. Người dùng hỏi câu hỏi tự nhiên ngắn gọn, không chỉ rõ năm ban hành (phản ánh đúng hành vi người dùng thực tế).
4. **Fix ở bước:**  
   - **Tầng M5 (Enrichment):** Trích xuất metadata `effective_year: 2024` và `status: "active"`. Dùng Contextual Prepend để ghi rõ: *"Văn bản này là Quy chế nghỉ phép ban hành năm 2024 thay thế quy chế năm 2023..."*.
   - **Tầng M2 (Search):** Bổ sung bộ lọc `filter={"status": "active"}` để triệt tiêu hoàn toàn tài liệu cũ.

### Nếu có thêm 1 giờ, tôi sẽ optimize:
1. **Triển khai Metadata Pre-filtering tự động:** Kết nối trường metadata `status` và `effective_date` đã trích xuất từ M5 vào truy vấn của Qdrant (`Filter(must=[FieldCondition(key="status", match=MatchValue(value="active"))])`).
2. **Tích hợp Query Decomposition cho câu hỏi Multi-hop:** Sử dụng một prompt nhẹ để phân tích câu hỏi phức tạp (như Q12) thành các truy vấn đơn nguyên tử trước khi gọi Hybrid Search.
3. **Tinh chỉnh ngưỡng Reranker Threshold:** Thiết lập ngưỡng cắt điểm (score cutoff) cho Cross-Encoder để loại bỏ các đoạn văn có điểm tương quan dưới 0.3, tránh đưa rác vào prompt của LLM.
