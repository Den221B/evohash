import re
lines = open('diplom.tex', encoding='utf-8').read().split('\n')
# search only body (after СОДЕРЖАНИЕ / Введение), skip front matter (отзыв/задание/аннотация) start
start = next(i for i, l in enumerate(lines) if 'bfseries Введение' in l)
terms = ['ASR', 'LPIPS', r'\bLLM', r'\bDCT', 'black-box', 'benchmark', 'backend',
         r'\bseed', r'\bFPR', 'benign recall', 'consistency gate', 'input purification',
         'оракул', 'score', 'black-box']
for t in terms:
    pat = re.compile(t)
    for i in range(start, len(lines)):
        if pat.search(lines[i]):
            ctx = lines[i].strip()
            print(f'--- {t!r}  (body line {i+1}) ---')
            print('   ', ctx[:200])
            break
