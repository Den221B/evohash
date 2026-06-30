# -*- coding: utf-8 -*-
HASHES = ['phash', 'pdq', 'photodna', 'neuralhash']
HN = {'phash': 'pHash', 'pdq': 'PDQ', 'photodna': 'PhotoDNA', 'neuralhash': 'NeuralHash'}
ORDER = {
    'phash': ['atkscopes', 'nes', 'prokos', 'simba', 'zo_signsgd', 'pHash_GigaEvolve', 'pHash_OpenEvolve'],
    'pdq': ['atkscopes', 'nes', 'prokos', 'simba', 'zo_signsgd', 'PDQ_GigaEvolve', 'PDQ_OpenEvolve'],
    'photodna': ['atkscopes', 'nes', 'prokos', 'simba', 'zo_signsgd', 'PhotoDNA_GigaEvolve', 'PhotoDNA_OpenEvolve'],
    'neuralhash': ['atkscopes', 'NeuralHash_GigaEvolve', 'NeuralHash_OpenEvolve'],
}

def pretty(a):
    m = {'nes': 'NES', 'simba': 'SimBA', 'zo_signsgd': 'ZO-SignSGD', 'prokos': 'Prokos', 'atkscopes': 'ATKScopes'}
    if a in m:
        return m[a]
    if a.endswith('GigaEvolve'):
        return 'GigaEvolve'
    if a.endswith('OpenEvolve'):
        return 'OpenEvolve'
    return a

def typ(a):
    return 'эволюционная атака' if 'Evolve' in a else 'оптимизированная конфигурация'

intro = (
    r"В приложении приведены примеры работы оптимизированных и эволюционных атак для всех "
    r"четырёх хеш-функций. Для всех конфигураций используются две фиксированные пары изображений "
    r"(одни и те же во всём приложении), выбранные как успешные сразу во всех~24 конфигурациях "
    r"<<хеш--атака>> и имеющие наименьшее среднее искажение~$L_2$. Каждая строка содержит четыре "
    r"изображения: исходное~$X$, целевое~$X_t$, атакованное~$X'$ и карту различия~$\Delta$. "
    r"Карта различия вычисляется как поэлементный модуль разности $\Delta=|X'-X|$, усиленный "
    r"в~10 раз и ограниченный диапазоном $[0;255]$: без усиления возмущения в собственной полосе "
    r"признаков хеша визуально неразличимы, а единый коэффициент~10 для всех конфигураций "
    r"обеспечивает сопоставимость карт различия между атаками и хеш-функциями. Под каждой строкой "
    r"указаны $L_2$-искажение и LPIPS для соответствующей пары. "
    r"Примеры сгруппированы по хеш-функциям: pHash --- рисунки~В.1--В.7, PDQ --- В.8--В.14, "
    r"PhotoDNA --- В.15--В.21, NeuralHash --- В.22--В.24. Все показанные атаки достигают коллизии "
    r"($d \leq \tau$; пороги~$\tau$ приведены в таблице~\ref{tab:thresholds})."
)

L = [
    r"\newpage",
    r"\setcounter{figure}{0}",
    r"\renewcommand{\thefigure}{В.\arabic{figure}}",
    r"\section*{\centering\normalsize\bfseries ПРИЛОЖЕНИЕ В\\Примеры атакованных изображений}",
    r"\addcontentsline{toc}{section}{ПРИЛОЖЕНИЕ В}",
    "",
    intro,
    "",
]
for h in HASHES:
    for a in ORDER[h]:
        L += [
            r"\begin{figure}[H]",
            r"\centering",
            r"\includegraphics[width=0.92\textwidth]{figures/diploma/gallery/%s_%s.pdf}" % (h, a),
            r"\caption{%s, %s (%s)}" % (HN[h], pretty(a), typ(a)),
            r"\label{fig:gal_%s_%s}" % (h, a),
            r"\end{figure}",
            "",
        ]

with open("appendix_gallery.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("wrote appendix_gallery.tex with 24 figures")
