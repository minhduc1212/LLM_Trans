# 📚 Novel Translator - LLM-Orchestrated Long-Form Translation Pipeline

Hệ thống dịch thuật tiểu thuyết/truyện dài chất lượng cao sử dụng mô hình ngôn ngữ lớn (LLM-orchestration). Dự án được thiết kế chuyên biệt để giải quyết các thách thức lớn khi dịch truyện dài: sự không nhất quán về đại từ nhân xưng/nhân vật, trôi lệch ngữ cảnh, vượt giới hạn API (rate limit/timeout) và kiểm duyệt nội dung an toàn của nhà cung cấp.

---

## 📁 Cấu trúc dự án

Thư mục gốc chứa các phiên bản phát triển của dự án:

*   **[`v2/`](./v2/) (Hoạt động chính / Khuyên dùng)**: Phiên bản hoàn thiện, tối ưu hóa toàn diện và sẵn sàng cho môi trường production. Tích hợp đầy đủ các cơ chế tự chữa lành, dịch tuần tự mang bối cảnh (Rolling History), trích xuất glossary động, aio async client và ghi file non-blocking.
*   **[`legacy/`](./legacy/) (Lưu trữ)**: Phiên bản mã nguồn cũ đại diện cho các thiết kế ban đầu của đường ống dịch thuật.
*   **[`test/`](./test/) (Thử nghiệm)**: Chứa các script nguyên mẫu (`v2.py`, `v3.py`), sơ đồ kiến trúc SVG và tài liệu phân tích hướng đi ban đầu (`v3.md`).

---

## ⚙️ Các tính năng cốt lõi (Tích hợp trong `v2`)

1.  **Tính nhất quán tuyệt đối (Glossary & Character Tracking)**:
    *   Tự động trích xuất nhân vật và từ khóa đặc trưng trước khi dịch.
    *   Tự động lọc động (relevance filtering) các từ khóa tương ứng với nội dung từng chunk để đưa vào System Prompt, giúp mô hình luôn dịch đúng vai trò và xưng hô.
2.  **Mạch văn mượt mà (Rolling Context History)**:
    *   Sử dụng cơ chế hội thoại đa lượt để chuyển bối cảnh (nguồn và bản dịch) từ các chunk trước cho chunk sau.
    *   Cơ chế Token Budget kiểm soát chặt chẽ dung lượng lịch sử tránh tràn ngữ cảnh.
3.  **Tự chữa lành & Xử lý lỗi (Self-healing & Error Recovery)**:
    *   Tự động tách đôi chunk đệ quy khi gặp lỗi cắt cụt token (`MAX_TOKENS`).
    *   Exponential Backoff kết hợp Jitter xử lý thông minh các mã lỗi API retryable (429, 408, 5xx).
4.  **Chống lười dịch và lặp từ (Similarity Checks)**:
    *   Tính toán độ tương đồng Vector (TF-IDF Cosine Similarity) giữa bản dịch và văn bản gốc hoặc các chunk trước để phát hiện dịch lười/lặp và tự động dịch lại với nhiệt độ (temperature) cao hơn.
5.  **Bỏ chặn kiểm duyệt (NSFW & Violence Bypass)**:
    *   Thiết lập ngưỡng chặn an toàn `BLOCK_NONE` cho toàn bộ các danh mục của Gemini API giúp giữ nguyên văn phong gốc của truyện hành động/18+ mà không bị lỗi hệ thống.
6.  **Tốc độ & An toànConcurrency**:
    *   Sử dụng client bất đồng bộ native (`client.aio`) kết hợp offload tác vụ ghi đĩa sang Thread Pool riêng để không làm nghẽn Event Loop.
    *   Global Semaphore khống chế số lượng request song song thực tế.

---

## 🚀 Bắt đầu nhanh

Để thiết lập cấu hình và bắt đầu chạy thử bản dịch tốt nhất:

1.  Di chuyển vào thư mục hoạt động:
    ```bash
    cd v2
    ```
2.  Đọc hướng dẫn cài đặt và lệnh chạy nhanh tại [v2/README.md](./v2/README.md).
3.  Tìm hiểu chi tiết luồng xử lý và thiết kế các hàm tại [v2/DOC.md](./v2/DOC.md).
