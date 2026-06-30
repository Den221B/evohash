import re
txt = open('diplom.tex', encoding='utf-8').read()
body = txt.split(r'\begin{thebibliography}')[0]
order = []
for m in re.finditer(r'\\cite\{([^}]+)\}', body):
    for k in m.group(1).split(','):
        k = k.strip()
        if k and k not in order:
            order.append(k)
print('citation order ({}):'.format(len(order)))
for i, k in enumerate(order, 1):
    print(i, k)
