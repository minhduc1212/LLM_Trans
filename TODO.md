tóm tắt nội dung đoạn truyện -> cho vào đoạn sau

chia đoạn văn bản thành các phần nhỏ hơn (chunk) -> 1000 token mỗi chunk

dịch trong thẻ <text> </text>

biên tập lại đoạn dịch nếu cần thiết để đảm bảo tính mạch lạc và tự nhiên của ngôn ngữ đích.

qwen2.5:14b
gemma3:12b
gemma   
Sakura LLM
ALMA
Helsinki-NLP
qwen3.5:14b -> thinking lâu, và quá nhiều -> X

test text:
    Waking from torpor is not my favourite experience. On the scale of discomforts, I'd put it \
on par with a moderate hangover, or the kind of cold where your sinuses creak if you press \
on your face. The actual sensation feels like neither of those things. Physically, I feel a \
little stiff, a little weak, but otherwise fine. Waking is more of a mental discomfort, a \
period in which your consciousness has to reassert itself after years of dormancy. Keep in \
mind that medically-induced torpor is not the same as sleep. Sleep conveys the passage of \
time, even if you don't dream. Not so with torpor. First you're awake, then you're not, \
then you're back . . . but something's missing. Something's missing, and you'll never be \
able to put your finger on what.

As soon as the Merian established orbit around its first target, a signal was sent from \
the navigation computer to our crew's torpor chambers. An automated system added a chemical \
solution to our nutrient drips, and that solution made its way to our respective brains, \
where it began the business of waking us up. I am told this process takes about an hour, \
but from my perspective, it happened in an instant. Light. Shapes. Confusion. I had to walk \
myself through the basics, as if I were reviewing every fact I'd learned during infancy. \
I have hands. I have a mouth. Those things I see are colours. I'm Ariadne. I exist. \
Then came memories, and context, and finally, a smile.

We're at Aecor.

I began to unpack the proverbial cotton from my mind, and walked myself through protocol. \
First, I pulled on the tabs that freed my wrists from their soft fabric restraints, then \
undid the ties around my waist and ankles as well. This may sound macabre, being tied up \
inside what amounts to a high-tech shipping crate, but the restraints are for a good cause, \
and removing them by yourself is a breeze. They're snugly attached to the sides of the torpor \
chamber, keeping me suspended in the middle of the container while I'm unconscious so that \
I don't float into the sides. This is far preferable to waking up with bruises all over.


tăng chunk lên 2500 token để giảm số lần phải chia nhỏ văn bản, giúp duy trì tính mạch lạc và tự nhiên của ngôn ngữ đích hơn.

Giữ đúng cấu trúc đoạn văn bản gốc, 

Không thêm *hoặc* hoặc các ký tự đặc biệt khác vào đoạn dịch, trừ khi chúng xuất hiện trong văn bản gốc.

prompt in small still best to use: literal, use suitable word, "Nếu gặp thuật ngữ khó, hãy tự linh hoạt dịch sang cụm từ tiếng Việt tương đương nhất, không cần giữ nguyên từ gốc, dựa trên ngữ cảnh, bối cảnh tiểu thuyết. Ví dụ: "torpor" có thể dịch là "ngủ đông", "protocol" có thể dịch là "quy trình"." -> hoặc giữ nguyên từ gốc nếu không có từ tương đương, nhưng phải đảm bảo tính mạch lạc và tự nhiên của ngôn ngữ đích.  + chú thích giải thích từ gốc đó. Ví dụ Somaforming (biến đổi cơ thể). => muốn để dịch từ khó đúng nghĩa -> có ngữ cảnh, bối cảnh tóm tắt nội dung đoạn truyện -> cho vào prompt đoạn sau 

-> lưu glossarry để giải thích các thuật ngữ khó, giúp người đọc hiểu rõ hơn về bối cảnh và nội dung của truyện. + đồng bộ cho lần dịch sau, nếu gặp lại thuật ngữ đó thì sẽ dịch theo cách đã giải thích trong glossary.

lưu cách xưng hô, cách gọi nhân vật, để đảm bảo sự nhất quán trong suốt quá trình dịch -> nếu ngữ cảnh thay đổi -> thay đổi theo

tóm tắt nội dung đoạn truyện -> cho vào prompt đoạn sau

chia thành các phần nhỏ hơn (chunk) để dịch, mỗi phần khoảng 1000 token, đảm bảo tính mạch lạc và tự nhiên của ngôn ngữ đích.

checkpointing: lưu lại tiến trình dịch sau mỗi chunk, để có thể tiếp tục từ điểm dừng nếu có sự cố (như mất điện hoặc Ctrl+C). File checkpoint sẽ tự động xóa khi hoàn thành.

Dịch song song

dịch theo phong cách của các thể loại(vì các thể loại sẽ có thuật ngữ riêng cho nó)

chia các config, prompt, glossary, checkpointing thành các file riêng biệt để dễ quản lý và tái sử dụng cho các dự án dịch khác nhau.

testing v2: xem lại phần chia chunk vì câu nếu không có dấu . thì sẽ bị miss ví dụ "第2068章 联手营救"
            -> xem cách tăng tốc độ dịch
            tự điều chỉnh văn phong theo ý thích
            điều chỉnh prompt = gemini