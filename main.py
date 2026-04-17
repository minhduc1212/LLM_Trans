import ollama
import re

# 1. System Prompt tinh giản gọn gàng
system_prompt = """Bạn là một dịch giả tiểu thuyết Sci-fi xuất sắc.
Nhiệm vụ: Dịch văn bản sang tiếng Việt với văn phong lãng mạn, tự nhiên.

CÁC LỆNH CẤM BẮT BUỘC (QUAN TRỌNG):
1. TUYỆT ĐỐI KHÔNG để lại bất kỳ từ tiếng Anh nào trong bản dịch (Ngoại trừ tên riêng viết hoa như Merian, Aecor, Ariadne).
2. KHÔNG ĐƯỢC PHÉP giữ nguyên thuật ngữ và bọc trong dấu sao (Ví dụ: cấm viết *torpor*, *protocol*). Phải dịch 100% sang tiếng Việt.
3. Nếu gặp thuật ngữ khó, hãy tự linh hoạt dịch sang cụm từ tiếng Việt tương đương nhất, không cần giữ nguyên từ gốc, dựa trên ngữ cảnh, bối cảnh tiểu thuyết. Ví dụ: "torpor" có thể dịch là "ngủ đông", "protocol" có thể dịch là "quy trình".

"""


raw_text = """Waking from torpor is not my favourite experience. On the scale of discomforts, I'd put it on par with a moderate hangover, or the kind of cold where your sinuses creak if you press on your face. The actual sensation feels like neither of those things. Physically, I feel a little stiff, a little weak, but otherwise fine. Waking is more of a mental discomfort, a period in which your consciousness has to reassert itself after years of dormancy. Keep in mind that medically-induced torpor is not the same as sleep. Sleep conveys the passage of time, even if you don't dream. Not so with torpor. First you're awake, then you're not, then you're back . . . but something's missing. Something's missing, and you'll never be able to put your finger on what.

As soon as the Merian established orbit around its first target, a signal was sent from the navigation computer to our crew's torpor chambers. An automated system added a chemical solution to our nutrient drips, and that solution made its way to our respective brains, where it began the business of waking us up. I am told this process takes about an hour, but from my perspective, it happened in an instant. Light. Shapes. Confusion. I had to walk myself through the basics, as if I were reviewing every fact I'd learned during infancy. I have hands. I have a mouth. Those things I see are colours. I'm Ariadne. I exist. Then came memories, and context, and finally, a smile.

We're at Aecor.

I began to unpack the proverbial cotton from my mind, and walked myself through protocol. First, I pulled on the tabs that freed my wrists from their soft fabric restraints, then undid the ties around my waist and ankles as well. This may sound macabre, being tied up inside what amounts to a high-tech shipping crate, but the restraints are for a good cause, and removing them by yourself is a breeze. They're snugly attached to the sides of the torpor chamber, keeping me suspended in the middle of the container while I'm unconscious so that I don't float into the sides. This is far preferable to waking up with bruises all over.
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
        'temperature': 0.3,
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