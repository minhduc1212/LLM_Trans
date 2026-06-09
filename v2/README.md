# 📚 Novel Translator Pipeline

Dịch tiểu thuyết đa ngôn ngữ, đa thể loại sử dụng Google GenAI API (Gemma 4 31B IT, v.v.)

---

## ✨ Tính năng

| Feature | Mô tả |
|---|---|
| **Chunk 2500 tokens** | Giữ mạch lạc, ít chia nhỏ hơn |
| **Glossary tự động** | Trích xuất & đồng bộ thuật ngữ qua các chunk |
| **Character tracking** | Nhất quán xưng hô, tên nhân vật |
| **Context summary** | Tóm tắt ngữ cảnh truyền sang chunk tiếp theo |
| **Checkpointing** | Tiếp tục từ điểm dừng khi gặp sự cố |
| **Parallel files** | Nhiều file cùng lúc |
| **Multi-genre** | 10 thể loại, thuật ngữ riêng cho từng loại |
| **Multi-language** | VI, ZH, EN, JA, KO, ID |
| **Cấu hình tách biệt** | config/, prompts/, data/ dễ tùy chỉnh |

---

## 📁 Cấu trúc

```
novel_translator/
├── config/
│   ├── settings.yaml        ← Model, chunk size, features
│   └── genres.yaml          ← Genre styles + language configs
├── prompts/
│   └── system_prompts.yaml  ← System prompt + user templates
├── data/
│   └── {project}_glossary.json  ← Thuật ngữ + nhân vật (tự tạo)
├── checkpoints/
│   └── {project}/           ← Tiến trình dịch (tự tạo, tự xóa)
├── output/                  ← Bản dịch đầu ra
├── src/
│   ├── chunker.py           ← Chia chunk thông minh
│   ├── checkpoint_manager.py
│   ├── glossary_manager.py
│   └── pipeline.py          ← Engine chính
└── main.py                  ← CLI entry point
```

---

## 🚀 Cài đặt

```bash
pip install -r requirements.txt
# Cấu hình API Key
# Windows (PowerShell):
$env:GEMINI_API_KEY="your_api_key_here"
# Linux/macOS:
export GEMINI_API_KEY="your_api_key_here"
```

---

## 🎯 Sử dụng

### Dịch 1 file
```bash
python main.py novel.txt -p my_project -g scifi
```

### Dịch thư mục (song song)
```bash
python main.py chapters/ -p my_project -g fantasy --workers 3
```

### Tiếp tục sau khi bị ngắt
```bash
python main.py novel.txt -p my_project -g scifi --resume
```

### Kiểm tra tiến trình
```bash
python main.py --status -p my_project
```

### Dịch sang tiếng Trung
```bash
python main.py novel.txt -p my_project -g literary -l zh
```

### Dùng model khác qua GenAI API (Gemini 2.5 Flash, v.v.)
```bash
python main.py novel.txt -p my_project -m gemini-2.5-flash -g thriller
```

### Chạy nhanh hơn (tắt glossary & summary)
```bash
python main.py novel.txt -p my_project --no-glossary --no-summary
```

---

## ⚙️ Tùy chỉnh

### Thay đổi model / chunk size
Sửa `config/settings.yaml`:
```yaml
model:
  name: "gemma4:27b"
  temperature: 0.2
translation:
  chunk_size: 3000
```

### Thêm thuật ngữ thủ công vào Glossary
Sửa `data/{project}_glossary.json`:
```json
{
  "terms": {
    "torpor": "ngủ đông",
    "somaforming": "biến đổi cơ thể (somaforming)"
  },
  "characters": {
    "Ariadne": "nhân vật chính, xưng 'tôi', nữ phi hành gia"
  }
}
```

### Tùy chỉnh system prompt
Sửa `prompts/system_prompts.yaml` → `system_base`.

### Thêm thể loại mới
Sửa `config/genres.yaml` → `genres:` thêm entry mới.

---

## 🎭 Thể loại hỗ trợ

| ID | Tên | Đặc điểm |
|---|---|---|
| `scifi` | Khoa học viễn tưởng | Thuật ngữ kỹ thuật, lạnh, chính xác |
| `fantasy` | Fantasy | Huyền bí, cổ điển, phép thuật |
| `romance` | Lãng mạn | Mượt mà, cảm xúc, tinh tế |
| `thriller` | Thriller | Nhanh, căng thẳng, súc tích |
| `historical` | Lịch sử | Trang trọng, cổ điển |
| `horror` | Kinh dị | U ám, rùng rợn |
| `literary` | Văn học | Tinh tế, giàu hình ảnh |
| `xianxia` | Tiên hiệp | Tu luyện, cảnh giới, linh khí |
| `wuxia` | Võ hiệp | Giang hồ, võ công, hào sảng |
| `isekai` | Isekai | Game system, kỹ năng, cấp độ |

---

## 📊 Output

- **Bản dịch**: `output/{filename}_vi.txt`
- **Glossary report**: `output/{project}_glossary_report.md`
- **Checkpoint** (tự xóa khi xong): `checkpoints/{project}/`
- **Glossary data**: `data/{project}_glossary.json`
