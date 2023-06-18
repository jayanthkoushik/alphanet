#!/usr/bin/env sh

set -e

function check_does_not_exist() {
    if [ ! -e "$1" ] || [ ! -s "$1" ]; then
        return 0
    else
        echo "WARNING: skipping '${1}': file already exists"
        return 1
    fi
}

for rset in "main" "randomnns"; do
    for dataset in imagenetlt_resnext50_crt imagenetlt_resnext50_lws imagenetlt_resnext50_ride placeslt_resnet152_crt placeslt_resnet152_lws cifar100_resnet32_ltr cifar100_resnet32_ride; do
        sfile="results/${rset}/${dataset}/summary.txt"
        if check_does_not_exist "${sfile}"; then
            (set -x; python run_printres.py --base-res-dir "results/${rset}/${dataset}" --rel-exp-paths "rho_0.1" "rho_0.25" "rho_0.5" "rho_0.75" "rho_1" "rho_1.25" "rho_1.5" "rho_1.75" "rho_2" --exp-names "0.1" "0.2" "0.5" "0.75" "1" "1.25" "1.5" "1.75" "2" --res-files-pattern "rep_*/result.pth" --exp-prefix "AlphaNet ($\\rho=" --exp-suffix "$)" > "${sfile}")
        fi

        sfile="results/${rset}/${dataset}/summary_top5.txt"
        if check_does_not_exist "${sfile}"; then
            (set -x; python run_printres.py --base-res-dir "results/${rset}/${dataset}" --rel-exp-paths "rho_0.1" "rho_0.25" "rho_0.5" "rho_0.75" "rho_1" "rho_1.25" "rho_1.5" "rho_1.75" "rho_2" --exp-names "0.1" "0.2" "0.5" "0.75" "1" "1.25" "1.5" "1.75" "2" --res-files-pattern "rep_*/result.pth" --exp-prefix "AlphaNet ($\\rho=" --exp-suffix "$)" --acc-k 5 > "${sfile}")
        fi

        if [ "${rset}" = "main" ]; then
            sfile="results/${rset}/${dataset}/summary_nn_adjusted.txt"
            if check_does_not_exist "${sfile}"; then
                (set -x; python run_printres.py --base-res-dir "results/${rset}/${dataset}" --rel-exp-paths "rho_0.1" "rho_0.25" "rho_0.5" "rho_0.75" "rho_1" "rho_1.25" "rho_1.5" "rho_1.75" "rho_2" --exp-names "0.1" "0.2" "0.5" "0.75" "1" "1.25" "1.5" "1.75" "2" --res-files-pattern "rep_*/result.pth" --exp-prefix "AlphaNet ($\\rho=" --exp-suffix "$)" --show-adjusted-acc > "${sfile}")
            fi
        fi
    done

    for dataset in imagenetlt_resnext50_crt imagenetlt_resnext50_lws imagenetlt_resnext50_ride; do
        for level in 2 3 4 5 10; do
            sfile="results/${rset}/${dataset}/summary_nn_semantic_${level}_adjusted.txt"
            if check_does_not_exist "${sfile}"; then
                (set -x; python run_printres.py --base-res-dir "results/${rset}/${dataset}" --rel-exp-paths "rho_0.1" "rho_0.25" "rho_0.5" "rho_0.75" "rho_1" "rho_1.25" "rho_1.5" "rho_1.75" "rho_2" --exp-names "0.1" "0.2" "0.5" "0.75" "1" "1.25" "1.5" "1.75" "2" --res-files-pattern "rep_*/result.pth" --exp-prefix "AlphaNet ($\\rho=" --exp-suffix "$)" --show-adjusted-acc --adjusted-acc-semantic --adjusted-acc-semantic-nns-level ${level} --imagenet-data-root "data/ImageNetLT" > "${sfile}")
            fi
        done
    done
done

for dist in "euclidean"; do
    for rho in "0.25" "0.5" "1" "2"; do
        sfile="results/nnsweep_${dist}/imagenetlt_resnext50_crt/rho_${rho}/summary.txt"
        if check_does_not_exist "${sfile}"; then
            (set -x; python run_printres.py --base-res-dir "results/nnsweep_${dist}/imagenetlt_resnext50_crt/rho_${rho}" --rel-exp-paths "k_1" "k_2" "k_3" "k_4" "k_5" "k_6" "k_7" "k_8" "k_9" "k_10" --exp-names "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" --res-files-pattern "rep_*/result.pth" --exp-prefix "AlphaNet (\$k=" --exp-suffix "$)" > "${sfile}")
        fi

        sfile="results/nnsweep_${dist}/imagenetlt_resnext50_crt/rho_${rho}/summary_top5.txt"
        if check_does_not_exist "${sfile}"; then
            (set -x; python run_printres.py --base-res-dir "results/nnsweep_${dist}/imagenetlt_resnext50_crt/rho_${rho}" --rel-exp-paths "k_1" "k_2" "k_3" "k_4" "k_5" "k_6" "k_7" "k_8" "k_9" "k_10" --exp-names "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" --res-files-pattern "rep_*/result.pth" --exp-prefix "AlphaNet (\$k=" --exp-suffix "$)" --acc-k 5 > "${sfile}")
        fi

        sfile="results/nnsweep_${dist}/imagenetlt_resnext50_crt/rho_${rho}/summary_nn_adjusted.txt"
        if check_does_not_exist "${sfile}"; then
            (set -x; python run_printres.py --base-res-dir "results/nnsweep_${dist}/imagenetlt_resnext50_crt/rho_${rho}" --rel-exp-paths "k_1" "k_2" "k_3" "k_4" "k_5" "k_6" "k_7" "k_8" "k_9" "k_10" --exp-names "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" --res-files-pattern "rep_*/result.pth" --exp-prefix "AlphaNet (\$k=" --exp-suffix "$)" --show-adjusted-acc > "${sfile}")
        fi

        for level in 2 3 4 5 10; do
            sfile="results/nnsweep_${dist}/imagenetlt_resnext50_crt/rho_${rho}/summary_nn_semantic_${level}_adjusted.txt"
            if check_does_not_exist "${sfile}"; then
                (set -x; python run_printres.py --base-res-dir "results/nnsweep_${dist}/imagenetlt_resnext50_crt/rho_${rho}" --rel-exp-paths "k_1" "k_2" "k_3" "k_4" "k_5" "k_6" "k_7" "k_8" "k_9" "k_10" --exp-names "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" --res-files-pattern "rep_*/result.pth" --exp-prefix "AlphaNet (\$k=" --exp-suffix "$)" --show-adjusted-acc --adjusted-acc-semantic --adjusted-acc-semantic-nns-level ${level} --imagenet-data-root "data/ImageNetLT" > "${sfile}")
            fi
        done
    done
done
