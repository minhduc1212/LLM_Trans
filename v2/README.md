# 📚 Novel Translator Pipeline (v2)

Dịch tiểu thuyết đa ngôn ngữ, đa thể loại sử dụng Google GenAI API (Gemma 4 31B IT, v.v.). Phiên bản **v2** đã được cải tiến toàn diện về hiệu năng, độ an toàn luồng, tính nhất quán ngữ cảnh và cơ chế chống lỗi API/kiểm duyệt.

---

## 🛠️ Tính năng nổi bật trong v2

| Feature | Mô tả | Trạng thái |
|---|---|---|
| **Semantic Chunking** | Tự động chia nhỏ văn bản theo ranh giới câu/đoạn (mặc định 2500 tokens). | Kích hoạt |
| **Concurrency Safety** | Global Semaphore duy nhất khống chế số lượng request song song, triệt tiêu lỗi nghẽn đa luồng. | Kích hoạt |
| **Safety Bypass** | Tự động hạ thấp kiểm duyệt an toàn (`BLOCK_NONE`) giúp dịch chân thực nội dung NSFW/bạo lực hư cấu. | Kích hoạt |
| **Thread-safe Glossary** | Trình quản lý glossary an toàn luồng (RMW Lock) kèm bộ lọc thuật ngữ liên quan từng chunk. | Kích hoạt |
| **Rolling History** | Tích lũy bối cảnh N chunk dịch trước dưới dạng chat đa lượt với cơ chế Token Budget tự cắt gọn. | Kích hoạt |
| **Recursive Splitting** | Tự động chia đôi và dịch đệ quy khi gặp lỗi cắt cụt token (`MAX_TOKENS`). | Kích hoạt |
| **Lazy/Duplicate Detection** | Cosine Similarity (TF-IDF) tự phát hiện lỗi lười dịch hoặc lặp từ và tự động retry với temp cao hơn. | Kích hoạt |
| **Non-blocking Async** | Sử dụng async client (`client.aio`) kết hợp offload I/O ghi đĩa sang Thread Pool để giải phóng event loop. | Kích hoạt |

---

## 📁 Cấu trúc thư mục

```
novel_translator/
├── config/
│   ├── settings.yaml        ← Cấu hình Model, tính năng v2, thông số retry
│   └── genres.yaml          ← Phong cách dịch theo thể loại + chỉ dẫn ngôn ngữ
├── prompts/
│   └── system_prompts.yaml  ← System prompt tối ưu hóa + template trích xuất
├── data/
│   └── {project}_glossary.json  ← Tệp thuật ngữ + nhân vật tự động đồng bộ
├── checkpoints/             ← Lưu trữ trạng thái dịch phục vụ --resume
├── output/                  ← Bản dịch đầu ra + tệp báo cáo glossary
├── src/
│   ├── chunker.py           ← Chia chunk ngữ nghĩa và ước lượng token
│   ├── checkpoint_manager.py
│   ├── glossary_manager.py  ← Trình quản lý glossary thread-safe
│   └── pipeline.py          ← Điều phối viên bất đồng bộ chính (Async Engine)
├── main.py                  ← CLI entry point
├── README.md                ← Hướng dẫn sử dụng nhanh
└── DOC.md                   ← [MỚI] Tài liệu chi tiết kiến trúc & mã nguồn
```

---

## 🚀 Cài đặt & Chuẩn bị

1. **Cài đặt thư viện**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Cấu hình API Key**:
   Tạo tệp `.env` tại thư mục gốc với nội dung:
   ```env
   GEMINI_API_KEY="your_api_key_here"
   ```

---

## 🎯 Hướng dẫn sử dụng

### Dịch truyện mới (mặc định dịch tuần tự mang bối cảnh mượt mà)
```bash
python main.py test_xianxia.txt -p test_xianxia -g xianxia
```

### Dịch nhanh không cần mang bối cảnh (song song hoàn toàn các chunk)
Sửa cấu hình `rolling_history: false` trong `config/settings.yaml` rồi chạy:
```bash
python main.py chapters/ -p my_project -g fantasy --workers 3
```

### Tiếp tục dịch khi gặp sự cố mạng/hết pin giữa chừng
```bash
python main.py test_xianxia.txt -p test_xianxia -g xianxia --resume
```

### Kiểm tra trạng thái tiến trình dịch của dự án
```bash
python main.py --status -p test_xianxia
```

---

## ⚙️ Cấu hình nâng cao trong `config/settings.yaml`

```yaml
features:
  auto_glossary: true                  # Tự động trích xuất nhân vật/thuật ngữ ban đầu
  auto_summary: true                   # Kích hoạt tóm tắt ngữ cảnh
  clean_thinking_tags: true            # Xóa các thẻ nháp <think> của mô hình
  relevance_filtering: true            # Bật lọc glossary khớp với nội dung chunk
  inject_glossary_in_system_prompt: true # Tiêm glossary vào System Prompt thay vì User Message
  rolling_history: true                # Mang bối cảnh dịch trước cho đoạn sau
  use_async_client: true               # Sử dụng client aio bất đồng bộ native
  detect_duplicate_translation: true   # Chống lỗi lười dịch hoặc lặp từ của mô hình

translation:
  chunk_size: 2500                     # Kích thước chunk tối ưu
  history_chapters: 2                  # Số lượng chunk dịch trước làm bối cảnh
  history_token_budget: 4000           # Ngân sách token tối đa cho lịch sử
  duplicate_threshold: 0.85            # Độ nhạy phát hiện trùng lặp
  lazy_threshold: 0.75                 # Độ nhạy phát hiện model copy nguyên văn
  similarity_lookback: 2               # Số chunk liền trước để so khớp trùng lặp
```

*Để biết thêm chi tiết về thiết kế luồng xử lý và cách hoạt động của từng hàm, vui lòng đọc tệp [DOC.md](file:///D:/LT/LLM%20Trans/v2/DOC.md).*
