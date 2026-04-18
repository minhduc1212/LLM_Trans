# command
# Dịch file, lưu ra file
python novel_translator.py input.txt -o output.txt

# Chỉ định ngôn ngữ + song song
python novel_translator.py manhwa.txt -l Korean -p

# Đổi model
python novel_translator.py light_novel.txt -m gemma4:e4b

# Chạy demo mặc định
python novel_translator.py