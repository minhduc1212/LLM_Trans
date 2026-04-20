import ollama
import re

# 1. System Prompt tinh giản gọn gàng
system_prompt = """Bạn là dịch giả văn học chuyên nghiệp. Tiêu chí hàng đầu: Trung thành tuyệt đối với ý nghĩa nguyên tác nhưng câu văn phải tuân thủ đúng cấu trúc ngữ pháp tiếng Việt.

  LUẬT BẮT BUỘC:
  1. BẢO TOÀN NỘI DUNG: Tuyệt đối KHÔNG thêm thắt tình tiết, KHÔNG bỏ sót ý, KHÔNG bóp méo văn phong của tác giả gốc.
  2. NGỮ PHÁP THUẦN VIỆT: Được phép chuyển đổi câu bị động thành chủ động, thêm đại từ hoặc đảo trật tự từ để câu không bị sượng (ví dụ: xử lý câu tỉnh lược), nhưng thông điệp phải y hệt bản gốc.
  3. ĐỊNH DẠNG: Giữ đúng số lượng đoạn văn. Tuyệt đối KHÔNG thêm ký tự (*, #, —, _, v.v.) ngoài bản gốc.
  5. CHỈ in ra bản dịch thuần túy. Không giải thích, không bình luận.
"""


raw_text = """I began to unpack the proverbial cotton from my mind, and walked myself through protocol. First, I pulled on the tabs that freed my wrists from their soft fabric restraints, then undid the ties around my waist and ankles as well. This may sound macabre, being tied up inside what amounts to a high-tech shipping crate, but the restraints are for a good cause, and removing them by yourself is a breeze. They’re snugly attached to the sides of the torpor chamber, keeping me suspended in the middle of the container while I’m unconscious so that I don’t float into the sides. This is far preferable to waking up with bruises all over.

Once my limbs were free, I hit the button that opened the chamber door. The light in my room was low, but I winced all the same as my eyes remembered how to adjust themselves. Torpor chambers regularly wash their occupants, but a daily spray of cleaning solution isn’t the same as a proper bath. My eyes, nose, and mouth were all crusty around the edges. Twenty-eight years without a real scrub will do that to you.

My hair, shaved before launch, had grown well past my shoulders. My nails had reached a hideous length as well, about what you’d expect after two years of no clipping. That’s about how much I aged in twenty-eight years of transit – two years. Torpor slows you down, and interstellar travel at half the speed of light further stalls the clock, but neither presses pause entirely. Cells divide and the heart keeps beating. We buy ourselves time while in torpor, not immortality.

I opened the hygiene kit, which some clever interior engineer had bolted to the wall within arm’s reach of my chamber. Nail clippers were the first item I retrieved, followed by a tiny collection bag. I pruned myself, returning my digits to usefulness. Curved keratin shards floated unattractively before me; I hid them away in the little bag as quickly as I could. My unruly hair would have to wait, but I took an elastic band from the kit and tied back my mermaid-like floating locks. The ground teams really do think of everything.

One by one, I removed the electrode patches that covered me from face to feet. Their steady pulses had kept my muscles from atrophying, and for that, I was grateful. Next, I removed the nutrient drip from my arm, bandaged myself, and collected the few drops of blood that had floated free. I then took a breath, readied some therapeutic profanities, and removed the catheter from the place where catheters go.

Ah, the glamour of space travel.

I could hear the faint rustle of my crewmates going through the same checklist of waking. The walls aboard the Merian are thin, but there are walls, and that point’s key. I’ve seen stills from classic movies in which space-travelling crews are put to sleep, but their chambers or pods or what have you are always lined up side by side, these grim rows of morgue-like containment. Let me be clear on this point: when you’ve woken up from nearly three decades of induced unconsciousness, and every orifice has gunk around it, and your nails look like talons, and your skin smells like a cross between a freshly-washed hospital bathroom and an abandoned pen at a zoo, and you’ve just pulled a tube wet with urine out of yourself . . . you need a minute alone. And that’s only taking basic hygiene and vanity into consideration. There’s an even more important psychological matter at hand during this time.
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