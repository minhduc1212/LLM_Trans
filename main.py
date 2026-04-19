"""
╔══════════════════════════════════════════════════════════════════╗
║         NOVEL TRANSLATION PIPELINE - gemma3:e4b / gemma4:e4b    ║
║   Dịch tiểu thuyết Hàn/Trung/Nhật/Anh → Tiếng Việt              ║
║   Tối ưu: chunking, retry, detect ngôn ngữ, clean output        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import ollama
import re
import sys
import time
import argparse
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL        = "gemma4:e4b"          
CHUNK_SIZE   = 1200                 # Số ký tự tối đa mỗi chunk (tránh vượt context)
MAX_WORKERS  = 2                    # Số luồng song song (tăng nếu RAM > 16GB)
TEMPERATURE  = 0.25                 # Thấp hơn = sát nghĩa hơn, ít "sáng tạo" hơn
NUM_CTX      = 8192                 # Context window (giữ thấp để chạy nhanh)
MAX_RETRIES  = 3                    # Số lần thử lại nếu lỗi
RETRY_DELAY  = 2                    # Giây chờ giữa mỗi lần retry

# ─────────────────────────────────────────────
# SYSTEM PROMPT — Đa ngôn ngữ, giải nghĩa thuật ngữ
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """Bạn là dịch giả tiểu thuyết chuyên nghiệp hàng đầu, thành thạo Hàn, Trung, Nhật, Anh → Việt.

## NHIỆM VỤ
Dịch đoạn văn được cung cấp sang tiếng Việt, đảm bảo:
- Văn phong tự nhiên, trôi chảy, giữ đúng phong cách gốc (lãng mạn, hành động, sci-fi, cổ trang… tùy ngữ cảnh)
- Sát nghĩa từng câu, không bỏ sót thông tin
- Chuyển tải được cảm xúc và nhịp điệu của nguyên tác

## QUY TẮC DỊCH THUẬT NGỮ
| Loại từ | Xử lý |
|---|---|
| Tên riêng (người, địa danh, tàu vũ trụ) | Giữ nguyên nếu là tên nước ngoài (Merian, Aecor, Ariadne…) |
| Tên riêng tiếng Hàn/Trung/Nhật | Phiên âm Việt hoặc giữ nguyên + chú thích nếu cần |
| Thuật ngữ kỹ thuật / chuyên ngành | Dịch sang tiếng Việt tương đương, KHÔNG giữ nguyên tiếng Anh |
| Từ cảm thán, tiếng lóng | Dịch tự nhiên theo văn phong, không dịch cứng nhắc |

## CẤM TUYỆT ĐỐI
1. KHÔNG để lại từ tiếng Anh / Hàn / Trung / Nhật trong bản dịch (trừ tên riêng đã quy định)
2. KHÔNG bọc thuật ngữ trong dấu sao: cấm viết *torpor*, *protocol*, *qi*, *mana*, *chaebol*
3. KHÔNG thêm chú thích, giải thích, tiêu đề, hay bất kỳ nội dung ngoài bản dịch
4. KHÔNG bắt đầu bằng "Dưới đây là bản dịch" hay bất kỳ câu giới thiệu nào
5. KHÔNG lặp lại nguyên văn gốc trong output

## VÍ DỤ DỊCH THUẬT NGỮ
- torpor → trạng thái ngủ đông / giấc đông miên
- protocol → quy trình / giao thức
- nutrient drip → dịch truyền dinh dưỡng
- navigation computer → máy tính dẫn đường
- 气 (khí) → nội lực / khí tức
- 修炼 → tu luyện
- 재벌 (chaebol) → tập đoàn tài phiệt
- 幼馴染 (osananajimi) → bạn thời thơ ấu

## OUTPUT
Chỉ trả về bản dịch thuần tiếng Việt. Không có gì khác."""


# ─────────────────────────────────────────────
# LANGUAGE DETECTOR (nhẹ, không cần thư viện)
# ─────────────────────────────────────────────
def detect_language(text: str) -> str:
    """Nhận diện ngôn ngữ dựa trên phân tích Unicode range."""
    sample = text[:500]
    scores = {
        "Korean":   len(re.findall(r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]', sample)),
        "Chinese":  len(re.findall(r'[\u4E00-\u9FFF\u3400-\u4DBF]', sample)),
        "Japanese": len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\uFF65-\uFF9F]', sample)),
        "English":  len(re.findall(r'[a-zA-Z]', sample)),
    }
    detected = max(scores, key=scores.get)
    # Nếu có cả Kanji lẫn Kana, ưu tiên Japanese
    if scores["Japanese"] > 0 and scores["Chinese"] > 0:
        detected = "Japanese"
    return detected if scores[detected] > 0 else "Unknown"


# ─────────────────────────────────────────────
# SMART CHUNKER — tách tại dấu câu hoàn chỉnh
# ─────────────────────────────────────────────
def smart_chunk(text: str, max_chars: int = CHUNK_SIZE) -> list[str]:
    """
    Tách văn bản thành các chunk, ưu tiên cắt tại:
    1. Dòng trống (xuống đoạn)
    2. Dấu chấm / chấm than / chấm hỏi
    3. Dấu phẩy (phương án dự phòng)
    Không cắt giữa chừng một câu.
    """
    if len(text) <= max_chars:
        return [text.strip()]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_chars, text_len)

        if end == text_len:
            chunks.append(text[start:].strip())
            break

        # Ưu tiên cắt tại dòng trống
        cut = text.rfind('\n\n', start, end)
        if cut == -1 or cut <= start:
            # Cắt tại dấu kết thúc câu (. ! ? 。！？)
            for punct in ['. ', '! ', '? ', '。', '！', '？', '\n']:
                cut = text.rfind(punct, start, end)
                if cut > start:
                    cut += len(punct)
                    break
        if cut <= start:
            # Phương án cuối: cắt tại dấu phẩy
            cut = text.rfind(', ', start, end)
            if cut <= start:
                cut = end  # Bắt buộc cắt nếu không tìm được

        chunks.append(text[start:cut].strip())
        start = cut

    return [c for c in chunks if c]


# ─────────────────────────────────────────────
# CLEAN OUTPUT — xóa thinking tags và rác
# ─────────────────────────────────────────────
def clean_output(raw: str) -> str:
    """Loại bỏ thinking tags, markdown thừa, câu mở đầu của model."""
    # Xóa thinking/reasoning blocks
    text = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<\|.*?\|>', '', text, flags=re.DOTALL)  # Special tokens

    # Xóa các thẻ HTML/XML còn sót
    text = re.sub(r'<[a-zA-Z_][^>]*>.*?</[a-zA-Z_][^>]*>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[a-zA-Z_][^>]*/>', '', text)

    # Xóa câu giới thiệu thường gặp của LLM
    text = re.sub(
        r'^(Dưới đây là bản dịch.*?\n|Bản dịch.*?\n|Here is.*?\n|Translation.*?\n)',
        '', text, flags=re.IGNORECASE
    )

    # Xóa markdown header thừa
    text = re.sub(r'^#+\s.*\n', '', text, flags=re.MULTILINE)

    # Chuẩn hoá khoảng trắng
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ─────────────────────────────────────────────
# TRANSLATE SINGLE CHUNK — với retry logic
# ─────────────────────────────────────────────
def translate_chunk(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    lang: str,
    verbose: bool = True,
) -> str:
    """Dịch một chunk với cơ chế retry tự động."""
    user_msg = f"[Ngôn ngữ gốc: {lang}]\n\n{chunk}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if verbose:
                status = f"  ⟳ Chunk {chunk_index}/{total_chunks}"
                if attempt > 1:
                    status += f" (retry {attempt})"
                print(status, end='\r', flush=True)

            response = ollama.chat(
                model=MODEL,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user',   'content': user_msg},
                ],
                options={
                    'temperature': TEMPERATURE,
                    'num_ctx':     NUM_CTX,
                    'top_p':       0.90,
                    'top_k':       40,
                    'repeat_penalty': 1.1,
                }
            )
            raw = response['message']['content']
            result = clean_output(raw)

            if verbose:
                print(f"  ✓ Chunk {chunk_index}/{total_chunks} — {len(chunk)}c → {len(result)}c")
            return result

        except ollama.ResponseError as e:
            print(f"\n  ✗ Lỗi API chunk {chunk_index}: {e}")
        except Exception as e:
            print(f"\n  ✗ Lỗi không xác định chunk {chunk_index}: {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)

    return f"[LỖI DỊCH CHUNK {chunk_index}]"


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def translate_novel(
    text: str,
    source_lang: Optional[str] = None,
    parallel: bool = False,
    verbose: bool = True,
) -> str:
    """
    Pipeline dịch tiểu thuyết hoàn chỉnh.

    Args:
        text:        Văn bản gốc cần dịch
        source_lang: Ngôn ngữ nguồn (tự động phát hiện nếu None)
        parallel:    Dịch song song (nhanh hơn nhưng cần RAM nhiều hơn)
        verbose:     In tiến trình ra màn hình

    Returns:
        Bản dịch tiếng Việt hoàn chỉnh
    """
    t_start = time.time()

    # 1. Phát hiện ngôn ngữ
    lang = source_lang or detect_language(text)
    if verbose:
        print(f"🔍 Ngôn ngữ phát hiện: {lang}")

    # 2. Tách chunk
    chunks = smart_chunk(text, CHUNK_SIZE)
    total  = len(chunks)
    if verbose:
        print(f"📦 Tách thành {total} chunk(s) | Model: {MODEL}\n")

    # 3. Dịch
    if total == 1 or not parallel:
        # Tuần tự — ổn định hơn
        translated_chunks = [
            translate_chunk(c, i + 1, total, lang, verbose)
            for i, c in enumerate(chunks)
        ]
    else:
        # Song song — nhanh hơn khi có nhiều chunk
        translated_chunks = [None] * total
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {
                executor.submit(translate_chunk, c, i + 1, total, lang, verbose): i
                for i, c in enumerate(chunks)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                translated_chunks[idx] = future.result()

    # 4. Ghép kết quả
    final = '\n\n'.join(p for p in translated_chunks if p)

    t_elapsed = time.time() - t_start
    if verbose:
        print(f"\n{'─'*50}")
        print(f"✅ Hoàn thành! {len(text)} ký tự gốc → {len(final)} ký tự dịch")
        print(f"⏱  Thời gian: {t_elapsed:.1f}s | Trung bình: {t_elapsed/total:.1f}s/chunk")

    return final


# ─────────────────────────────────────────────
# CLI — Sử dụng từ terminal
# ─────────────────────────────────────────────

def run_cli():
    global MODEL
    parser = argparse.ArgumentParser(
        description="Dịch tiểu thuyết Hàn/Trung/Nhật/Anh → Tiếng Việt bằng Ollama"
    )
    parser.add_argument("input",  nargs="?", help="File văn bản gốc (.txt)")
    parser.add_argument("-o", "--output",   help="File lưu bản dịch (.txt)")
    parser.add_argument("-l", "--lang",     help="Chỉ định ngôn ngữ: Korean/Chinese/Japanese/English")
    parser.add_argument("-p", "--parallel", action="store_true", help="Dịch song song")
    parser.add_argument("-m", "--model",    help=f"Tên model Ollama (mặc định: {MODEL})")
    args = parser.parse_args()

    
    if args.model:
        MODEL = args.model

    # Đọc input
    if args.input:
        path = Path(args.input)
        if not path.exists():
            print(f"❌ Không tìm thấy file: {path}")
            sys.exit(1)
        text = path.read_text(encoding='utf-8')
        print(f"📄 Đọc file: {path} ({len(text)} ký tự)")
    else:
        # Dùng đoạn text mẫu mặc định
        text = SAMPLE_TEXT

    print("🚀 Bắt đầu dịch...\n" + "─"*50)
    result = translate_novel(text, source_lang=args.lang, parallel=args.parallel)

    # Ghi output
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(result, encoding='utf-8')
        print(f"\n💾 Đã lưu: {out_path}")
    else:
        print("\n" + "═"*50)
        print("📖 KẾT QUẢ DỊCH:")
        print("═"*50)
        print(result)


# ─────────────────────────────────────────────
# SAMPLE TEXT — Demo mặc định khi chạy trực tiếp
# ─────────────────────────────────────────────
SAMPLE_TEXT = """\
We don’t change much – nothing that would make us unrecognisable, nothing that would push us beyond the realm of our humanity, nothing that changes how I think or act or perceive. Only a small number of genetic supplementations are actually possible, and none of them are permanent. You see, an adult human body is comprised of trillions of cells, and if you don’t constantly maintain the careful changes you’ve made to them, they either revert back to their original template as they naturally replace themselves, or mutate malignantly. Hence, the enzyme patch: a synthetic skin-like delivery system that gives our bodies that little bit extra we need to survive on different worlds. If I were to stop wearing patches, my body would eventually flush the supplementations out, and I’d be the same as I was before I became an astronaut (plus the years and the memories).

Somaforming is an elegant solution, but not an immediate process. If enzyme patches are still used medically, you know this already – if you’re diabetic, for example, and can’t produce insulin on your own. But if you’ve never worn a patch (or if they’re old news by now), you might imagine something more dramatic than is accurate. I once spoke to a kid at an outreach event who was very disappointed to learn that applying a patch does not result in instant transformation (complete with an animation sequence and a theme song, I’d imagine). We astronauts are not superheroes, nor shape-shifters. We’re as human as you. While our bodies are wondrously malleable things, they still need time to adjust. Life-saving organ transplants or helpful medicines can often be met with some level of physiological resistance; the same is true of somaforming. It is more preferable, by far, to be unconscious while your body sorts itself out.
"""


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cli()
    else:
        # Chạy trực tiếp = dùng SAMPLE_TEXT
        print("🚀 Novel Translation Pipeline — gemma4:e4b\n" + "─"*50)
        result = translate_novel(SAMPLE_TEXT, parallel=False)
        print("\n" + "═"*50)
        print("📖 KẾT QUẢ:")
        print("═"*50)
        print(result)