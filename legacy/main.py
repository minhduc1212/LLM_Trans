"""
╔══════════════════════════════════════════════════════════════════╗
║         NOVEL TRANSLATION PIPELINE - gemma3:e4b / gemma4:e4b     ║
║   Dịch tiểu thuyết Hàn/Trung/Nhật/Anh → Tiếng Việt               ║
║   Tối ưu: chunking, retry, detect ngôn ngữ, clean output         ║
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
        description="Dịch tiểu thuyết Hàn/Trung/Nhật/Anh → Tiếng Việt bằng Gemma4:e4b"
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
第2068章 联手营救
梁言催动法力，在阵中不断斩杀白骨骷髅。
他的神识比其他人要庞大许多，附近虚空稍有异动，剑光便提前赶到，一剑就把骷髅扼杀在摇篮之中。

可即便如此，他还是感觉有些力不从心。

因为阵中的白骨骷髅源源不断，无论他们如何斩杀，始终会有新的骷髅诞生，仿佛永远也不会枯竭。

而梁言这边的修士不能失手，一旦失手就有性命之忧，交手仅片刻，已经有七名通玄真君惨死在白骨骷髅的手中！

“这魔像非同一般，应该是葬天帝所留！”

梁言抬头看向那擎天高的雕像，眼中露出了凝重之色。

他能感应到，这座雕像还没有用全力，因为它的千条手臂都用来托住头顶的黑海，而那片黑海也是镇压四圣的关键！

“四圣都陷入了昏迷，想要营救他们就必须破坏牢笼，可这雕像的神通诡异莫测，众人合力都难以抵挡，要如何接近四圣呢”

梁言心念电转，一边斩杀附近的骷髅，一边思忖救人之法。

就在此时，那雕像再次动了，一只手掌从天而降，狠狠拍在众人的防御结界上。

砰！
巨响声中，结界晃动不止，强悍的力量穿透了“三才九绝阵”的防御，使得阵中的十几个修士站立不稳，一屁股跌倒在地上。

而就在他们倒地的一瞬间，背后虚空裂开，一只只白骨骷髅从中钻出，一爪就洞穿了他们的丹田，将元神和真灵拽出，转眼就撕得粉碎！
“啊！”

阵中响起了凄厉的惨叫声。

然而雕像的攻击并没有结束，那条手臂猛地抬起，随后又快速落下，再次击打防御结界。

砰！砰！砰.
一连串的巨响传来，由二十位化劫老祖和七百多位通玄真君联手凝聚的结界，在雕像的猛攻之下居然扭曲变形，连光芒都暗淡了许多。

“不妙啊！”

梁言眉头紧锁，脸色阴沉。

他知道再这样下去，结界迟早会被攻破，众人无法抵挡眼前的诡异雕像，别说营救四圣了，到时候恐怕全军覆没！
正是心急如焚之际，耳畔忽然听见一个声音，悠悠道：“此乃‘托天魔像’，葬天帝亲手炼制，以你们的手段无法战胜”

听到这个声音，梁言心中一动，立刻环顾四周，却没有找到半个人影。

“不用找我了，我是不会现身的。”老者的声音似远还近，不可捉摸。

“前辈既然开口，就一定不会袖手旁观，还请前辈出手相助。”梁言在心中默念道。

“呵呵，我说了不会现身，更不会出手，一切都要靠你自己。”

“前辈的意思是”梁言捕捉到了这句话的关键点，沉声问道：“你觉得我能对付这魔像？”

“当然。”老者的声音悠悠道：“你身上有一样东西，可以让托天魔像失去灵性，成为废铜烂铁。”

“什么东西？”

“洛水。”

“洛水？”梁言眼神一亮。

“你仔细看看，托天魔像的枢纽就在胸口，只要你能将洛水灌入其中，魔像不攻自破！”

得到老者的提示后，梁言立刻抬头看向那座高大的魔像，神识全部放出，果然在魔像的胸口看见了一个“魔”字！

相比与巨大的托天魔像，这个“魔”字显得十分渺小，几乎微不可查。但如果近距离观看的话，这个“魔”字也有百丈方圆，字迹工整，像是被人用刀斧在山壁上仔细凿刻而成！

“托天魔像如此高大，还有各种诡异神通，我连腾空千丈都难，如何去到他的胸口？”梁言皱眉道。

“这就是你要考虑的事情了。”

老者呵呵笑道：“葬天帝用托天魔像将轮回池的池水捞出，并以此镇压四圣，所以魔像大部分力量都用来托举池水，你不是没有机会，就看自己能不能把握住了。”

说完这句话，老者的气息渐渐消失，无论梁言如何询问，对方都不再回应了。

“这人神神秘秘，究竟是什么来历？”

梁言心中疑惑，但他知道现在不是纠结这个的时候。

就这片刻的功夫，托天魔像的攻击越来越猛烈，头顶“佛光”飘荡，魔像大手一下下捶打结界，使得众人的压力越来越大。

而在阵中，白骨骷髅时不时现身，从内部刺杀修士，更是让众人防不胜防！

“内忧外患.只怕再过片刻就会阵脚大乱，这些好不容易逃出生天的修士都要葬身在魔像手中！”

梁言内心焦急，暗中思忖对策。

忽然，他像是想到了什么，口中喃喃自语道：“轮回池轮回池！”

话音刚落，梁言眼中闪过一道精光！
他猛地抬头，目光死死盯着托天魔像头顶举着的那片黑海，暗中把法力注入“天机珠”内，开始凝神感应。

忽然，“天机珠”在体内跳动了一下！
“果然有所感应！”

梁言心中大喜，急忙把赵寻真放了出来，命令她守在附近，不让那些白骨骷髅近身。

赵寻真欣然领命，把界伞祭在头顶，幽幽鬼气散发出来，罩住了梁言和自己。

“天机珠能够感应轮回池水，如果能引动黑海，托天魔像势必要分散一部分力量去镇压，届时它首尾不能兼顾，就是我反击的最好时机！”

想到这里，梁言心无旁骛，把体内的所有法力都运转起来，全力催动天机珠！

冥冥之中，似乎有一条桥梁把天机珠和轮回池的池水连接起来，双方产生了感应，以至于梁言都生出一种幻觉，仿佛自己身处黑海内部，能够感应周围的水流！

梁言慢慢举起了右手，目光死死盯着悬浮在“托天魔像”头顶的黑海。

忽然，他并指如剑，遥遥指向了黑海某处！

轰隆！
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