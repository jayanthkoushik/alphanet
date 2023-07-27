\hypertarget{tbl:dataset_stats}{}
\begin{table}
  \centering
  \begin{tabular}{@{}lrrrclrrrcrrr@{}}
    \toprule

    \multirow{2}*{Dataset} &
    \multicolumn{3}{c}{Samples} &&
    \multicolumn{4}{c}{Classes} &&
    \multicolumn{3}{c}{Train samples} \\

    \cmidrule{2-4} \cmidrule{6-9} \cmidrule{11-13}

    & train & val & test && total & many & med. & few && many & med. & few \\

    \midrule

    ImageNet‑LT & 115,846 & 20,000 & 50,000 && 1,000 & 385 & 479 & 136 && 88,693 & 25,510 & 1,643 \\
    Places‑LT & 62,500 & 7,300 & 36,500 && 365 & 131 & 163 & 71 && 52,762 & 8,934 & 804 \\
    CIFAR‑100‑LT & 10,847 & - & 10,000 && 100 & 35 & 35 & 30 && 8,824 & 1,718 & 305 \\
    iNaturalist & 437,513 & - & 24,426 && 8,142 & 842 & 4,076 & 3,224 && 258,340 & 133,061 & 46,112 \\

    \bottomrule
  \end{tabular}
  \caption{Statistics of long-tailed datasets.\label{tbl:dataset_stats}}
\end{table}
