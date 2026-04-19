import ollama
import re

# 1. System Prompt tinh giản gọn gàng
system_prompt = """Bạn là một dịch giả tiểu thuyết Sci-fi xuất sắc.
Nhiệm vụ: Dịch văn bản sang tiếng Việt với văn phong của chính tác giả hoặc sát với văn phong của tác giả nhất.

CÁC LỆNH CẤM BẮT BUỘC (QUAN TRỌNG):
1. TUYỆT ĐỐI KHÔNG để lại bất kỳ từ tiếng Anh nào trong bản dịch (Ngoại trừ tên riêng viết hoa như Merian, Aecor, Ariadne).
2. KHÔNG ĐƯỢC PHÉP giữ nguyên thuật ngữ và bọc trong dấu sao (Ví dụ: cấm viết *torpor*, *protocol*). Phải dịch 100% sang tiếng Việt.
3. Nếu gặp thuật ngữ khó, hãy tự linh hoạt dịch sang cụm từ tiếng Việt tương đương nhất, không cần giữ nguyên từ gốc, dựa trên ngữ cảnh, bối cảnh tiểu thuyết. Ví dụ: "torpor" có thể dịch là "ngủ đông", "protocol" có thể dịch là "quy trình".

"""


raw_text = """We don’t change much – nothing that would make us unrecognisable, nothing that would push us beyond the realm of our humanity, nothing that changes how I think or act or perceive. Only a small number of genetic supplementations are actually possible, and none of them are permanent. You see, an adult human body is comprised of trillions of cells, and if you don’t constantly maintain the careful changes you’ve made to them, they either revert back to their original template as they naturally replace themselves, or mutate malignantly. Hence, the enzyme patch: a synthetic skin-like delivery system that gives our bodies that little bit extra we need to survive on different worlds. If I were to stop wearing patches, my body would eventually flush the supplementations out, and I’d be the same as I was before I became an astronaut (plus the years and the memories).

Somaforming is an elegant solution, but not an immediate process. If enzyme patches are still used medically, you know this already – if you’re diabetic, for example, and can’t produce insulin on your own. But if you’ve never worn a patch (or if they’re old news by now), you might imagine something more dramatic than is accurate. I once spoke to a kid at an outreach event who was very disappointed to learn that applying a patch does not result in instant transformation (complete with an animation sequence and a theme song, I’d imagine). We astronauts are not superheroes, nor shape-shifters. We’re as human as you. While our bodies are wondrously malleable things, they still need time to adjust. Life-saving organ transplants or helpful medicines can often be met with some level of physiological resistance; the same is true of somaforming. It is more preferable, by far, to be unconscious while your body sorts itself out.
"""

print("🚀 Đang chạy dịch bằng Gemma 4 E4B...\n" + "-"*40)

# 2. Gọi API với cấu hình tối ưu
response = ollama.chat(
    model='gemma4:e4b', 
    messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': raw_text}
    ],
    options={
        'temperature': 0.1,
        'num_ctx': 8192,       # Khóa Context Window ở mức 8K để chạy siêu nhẹ
        'top_p': 0.9,
    }
)

# 3. Lọc kết quả (Loại bỏ block Thinking nếu có)
raw_output = response['message']['content']

# Dùng Regex để xóa mọi nội dung nằm trong các thẻ suy nghĩ (như <think>...</think> hoặc <|channel>thought)
clean_translation = re.sub(r'<[^>]*>.*?</[^>]*>', '', raw_output, flags=re.DOTALL)

# Dọn dẹp các ký tự thừa
final_translation = clean_translation.strip()

print(final_translation)