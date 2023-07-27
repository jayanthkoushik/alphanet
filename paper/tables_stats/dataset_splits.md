\hypertarget{tbl:dataset_splits}{}
\begin{table}
  \centering
  \begin{tabular}{@{}lrrrcrrr@{}}
    \toprule

    \multirow{2}*{Dataset} &
    \multicolumn{3}{c}{Min. class samples} &&
    \multicolumn{3}{c}{Max. class samples} \\

    \cmidrule{2-4} \cmidrule{6-8}

    & many & med. & few && many & med. & few \\

    \midrule

    ImageNet‑LT & 101 & 20 & 5 && 1,280 & 100 & 19 \\
    Places‑LT & 103 & 20 & 5 && 4,980 & 100 & 19 \\
    CIFAR‑100‑LT & 102 & 20 & 5 && 500 & 98 & 19 \\
    iNaturalist & 101 & 20 & 2 && 1,000 & 100 & 19 \\

    \bottomrule
  \end{tabular}
  \caption{Minimum and maximum per-class training samples for long-tailed datasets.\label{tbl:dataset_splits}}
\end{table}
