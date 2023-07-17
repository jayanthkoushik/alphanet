#!/usr/bin/env sh

for f in $*; do
    sed -E 's/(ImageNet|Places|CIFAR-100)-LT/\1‑LT/g; s/CIFAR-100/CIFAR‑100/g; s/(ResNeX?t)-([0-9]+)/\1‑\2/g; s/-(cRT|LWS|RIDE|LTR)/‑\1/g; s/([tT]op)-/\1‑/g; s/\$\\tau\$-normalized/\$\\tau\$‑normalized/g;' "${f}" > "${f}.tmp"
    if ! diff -q "${f}" "${f}.tmp" > /dev/null; then
        cat $f.tmp > $f
        echo "+ updated '${f}'"
    fi
    rm $f.tmp
done
