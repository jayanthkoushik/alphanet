#!/usr/bin/env sh

set -e

function check_does_not_exist() {
    if [ ! -e "$1" ] || [ ! -s "$1" ]; then
        mkdir -p "$(dirname "$1")"
        return 0
    else
        echo "WARNING: skipping '${1}': file already exists"
        return 1
    fi
}

function get_dataset_desc() {
    case $1 in
        imagenetlt_resnext50_crt)
            echo "\\\\\\\\acs{cRT} baseline on ImageNet-LT"
            ;;
        imagenetlt_resnext50_lws)
            echo "\\\\\\\\acs{LWS} baseline on ImageNet-LT"
            ;;
        imagenetlt_resnext50_ride)
            echo "\\\\\\\\acs{RIDE} baseline on ImageNet-LT"
            ;;
        placeslt_resnet152_crt)
            echo "\\\\\\\\acs{cRT} baseline on Places-LT"
            ;;
        placeslt_resnet152_lws)
            echo "\\\\\\\\acs{LWS} baseline on Places-LT"
            ;;
        cifar100_resnet32_ltr)
            echo "\\\\\\\\acs{LTR} baseline on CIFAR-100-LT"
            ;;
        cifar100_resnet32_ride)
            echo "\\\\\\\\acs{RIDE} baseline on CIFAR-100-LT"
            ;;
        inatlt_resnet152_crt)
            echo "\\\\\\\\acs{cRT} baseline on iNaturalist-LT"
            ;;
        *)
            echo "Unknown dataset: $1" >&2
            exit 1
            ;;
    esac
}

for rset in "main" "randomnns"; do
    if [ "${rset}" = "main" ]; then
        alphanet_desc="AlphaNet using 5 nearest neighbors based on Euclidean distance"
    else
        alphanet_desc="AlphaNet using 5 randomly chosen neighbors"
    fi

    for dataset in imagenetlt_resnext50_crt imagenetlt_resnext50_lws imagenetlt_resnext50_ride placeslt_resnet152_crt placeslt_resnet152_lws cifar100_resnet32_ltr cifar100_resnet32_ride; do
        dataset_desc=$(get_dataset_desc "${dataset}")

        sfile="paper/tables/appendix/${rset}/${dataset}/acc_top1.md"
        if check_does_not_exist "${sfile}"; then
            (set -x; python run_printres.py --base-res-dir "results/${rset}/${dataset}" --rel-exp-paths "rho_0.1" "rho_0.25" "rho_0.5" "rho_0.75" "rho_1" "rho_1.25" "rho_1.5" "rho_1.75" "rho_2" --exp-names "0.1" "0.2" "0.5" "0.75" "1" "1.25" "1.5" "1.75" "2" --res-files-pattern "rep_*/result.pth" --exp-prefix "AlphaNet (_ρ_\\ =\\ " --exp-suffix ")" --no-print-csv > "${sfile}")
            (set -x; echo "\n: Top-1 accuracy for ${alphanet_desc}, with ${dataset_desc}. {#tbl:${rset}_${dataset}_top1}" >> "${sfile}")
        fi

        sfile="paper/tables/appendix/${rset}/${dataset}/acc_top5.md"
        if check_does_not_exist "${sfile}"; then
            (set -x; python run_printres.py --base-res-dir "results/${rset}/${dataset}" --rel-exp-paths "rho_0.1" "rho_0.25" "rho_0.5" "rho_0.75" "rho_1" "rho_1.25" "rho_1.5" "rho_1.75" "rho_2" --exp-names "0.1" "0.2" "0.5" "0.75" "1" "1.25" "1.5" "1.75" "2" --res-files-pattern "rep_*/result.pth" --exp-prefix "AlphaNet (_ρ_\\ =\\ " --exp-suffix ")" --no-print-csv --acc-k 5 > "${sfile}")
            (set -x; echo "\n: Top-5 accuracy for ${alphanet_desc}, with ${dataset_desc}. {#tbl:${rset}_${dataset}_top5}" >> "${sfile}")
        fi

        if [ "${rset}" = "main" ]; then
            sfile="paper/tables/appendix/${rset}/${dataset}/acc_nn_adjusted_top1.md"
            if check_does_not_exist "${sfile}"; then
                (set -x; python run_printres.py --base-res-dir "results/${rset}/${dataset}" --rel-exp-paths "rho_0.1" "rho_0.25" "rho_0.5" "rho_0.75" "rho_1" "rho_1.25" "rho_1.5" "rho_1.75" "rho_2" --exp-names "0.1" "0.2" "0.5" "0.75" "1" "1.25" "1.5" "1.75" "2" --res-files-pattern "rep_*/result.pth" --exp-prefix "AlphaNet (_ρ_\\ =\\ " --exp-suffix ")" --no-print-csv --show-adjusted-acc > "${sfile}")
                (set -x; echo "\n: Accuracy computed by considering nearest neighbor predictions as correct, for ${alphanet_desc}, with ${dataset_desc}. {#tbl:${rset}_${dataset}_adjusted_top1}" >> "${sfile}")
            fi
        fi
    done

    dataset="inatlt_resnet152_crt"
    dataset_desc=$(get_dataset_desc "${dataset}")

    sfile="paper/tables/appendix/${rset}/${dataset}/acc_top1.md"
    if check_does_not_exist "${sfile}"; then
        (set -x; python run_printres.py --base-res-dir "results/${rset}/${dataset}" --rel-exp-paths "rho_0.01" "rho_0.02" "rho_0.03" "rho_0.04" "rho_0.05" --exp-names "0.01" "0.02" "0.03" "0.04" "0.05" --res-files-pattern "rep_*/result.pth" --exp-prefix "AlphaNet (_ρ_\\ =\\ " --exp-suffix ")" --no-print-csv > "${sfile}")
        (set -x; echo "\n: Top-1 accuracy for ${alphanet_desc}, with ${dataset_desc}. {#tbl:${rset}_${dataset}_top1}" >> "${sfile}")
    fi

    sfile="paper/tables/appendix/${rset}/${dataset}/acc_top5.md"
    if check_does_not_exist "${sfile}"; then
        (set -x; python run_printres.py --base-res-dir "results/${rset}/${dataset}" --rel-exp-paths "rho_0.01" "rho_0.02" "rho_0.03" "rho_0.04" "rho_0.05" --exp-names "0.01" "0.02" "0.03" "0.04" "0.05" --res-files-pattern "rep_*/result.pth" --exp-prefix "AlphaNet (_ρ_\\ =\\ " --exp-suffix ")" --no-print-csv --acc-k 5 > "${sfile}")
        (set -x; echo "\n: Top-5 accuracy for ${alphanet_desc}, with ${dataset_desc}. {#tbl:${rset}_${dataset}_top5}" >> "${sfile}")
    fi

    if [ "${rset}" = "main" ]; then
        sfile="paper/tables/appendix/${rset}/${dataset}/acc_nn_adjusted_top1.md"
        if check_does_not_exist "${sfile}"; then
            (set -x; python run_printres.py --base-res-dir "results/${rset}/${dataset}" --rel-exp-paths "rho_0.01" "rho_0.02" "rho_0.03" "rho_0.04" "rho_0.05" --exp-names "0.01" "0.02" "0.03" "0.04" "0.05" --res-files-pattern "rep_*/result.pth" --exp-prefix "AlphaNet (_ρ_\\ =\\ " --exp-suffix ")" --no-print-csv --show-adjusted-acc > "${sfile}")
            (set -x; echo "\n: Accuracy computed by considering nearest neighbor predictions as correct, for ${alphanet_desc}, with ${dataset_desc}. {#tbl:${rset}_${dataset}_adjusted_top1}" >> "${sfile}")
        fi
    fi

    for dataset in imagenetlt_resnext50_crt imagenetlt_resnext50_lws imagenetlt_resnext50_ride; do
        dataset_desc=$(get_dataset_desc "${dataset}")
        for level in 2 3 4 5 10; do
            sfile="paper/tables/appendix/${rset}/${dataset}/acc_nn_semantic_${level}_adjusted_top1.md"
            if check_does_not_exist "${sfile}"; then
                (set -x; python run_printres.py --base-res-dir "results/${rset}/${dataset}" --rel-exp-paths "rho_0.1" "rho_0.25" "rho_0.5" "rho_0.75" "rho_1" "rho_1.25" "rho_1.5" "rho_1.75" "rho_2" --exp-names "0.1" "0.2" "0.5" "0.75" "1" "1.25" "1.5" "1.75" "2" --res-files-pattern "rep_*/result.pth" --exp-prefix "AlphaNet (_ρ_\\ =\\ " --exp-suffix ")" --no-print-csv --show-adjusted-acc --adjusted-acc-semantic --adjusted-acc-semantic-nns-level ${level} --imagenet-data-root "data/ImageNetLT" > "${sfile}")
                (set -x; echo "\n: Accuracy computed by considering predictions within ${level} WordNet nodes as correct, for ${alphanet_desc}, with ${dataset_desc}. {#tbl:${rset}_${dataset}_semantic${level}_adjusted_top1}" >> "${sfile}")
            fi
        done
    done
done

for dist in "euclidean" "cosine"; do
    if [ ${dist} = "euclidean" ]; then
        dist_desc="Euclidean distance"
    else
        dist_desc="cosine distance"
    fi

    for rho in "0.25" "0.5" "1" "2"; do
        sfile="paper/tables/appendix/nnsweep_${dist}/imagenetlt_resnext50_crt/rho_${rho}/acc_top1.md"
        if check_does_not_exist "${sfile}"; then
            (set -x; python run_printres.py --base-res-dir "results/nnsweep_${dist}/imagenetlt_resnext50_crt/rho_${rho}" --rel-exp-paths "k_1" "k_2" "k_3" "k_4" "k_5" "k_6" "k_7" "k_8" "k_9" "k_10" --exp-names "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" --res-files-pattern "rep_*/result.pth" --exp-prefix "AlphaNet (_k_\\ =\\ " --exp-suffix ")" --no-print-csv > "${sfile}")
            (set -x; echo "\n: Top-1 accuracy for AlphaNet using varying number of nearest neighbors (_k_) based on ${dist_desc}, with \\\\acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_${dist}_imagenetlt_resnext50_crt_rho_${rho}_top1}" >> "${sfile}")
        fi

        sfile="paper/tables/appendix/nnsweep_${dist}/imagenetlt_resnext50_crt/rho_${rho}/acc_top5.md"
        if check_does_not_exist "${sfile}"; then
            (set -x; python run_printres.py --base-res-dir "results/nnsweep_${dist}/imagenetlt_resnext50_crt/rho_${rho}" --rel-exp-paths "k_1" "k_2" "k_3" "k_4" "k_5" "k_6" "k_7" "k_8" "k_9" "k_10" --exp-names "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" --res-files-pattern "rep_*/result.pth" --exp-prefix "AlphaNet (_k_\\ =\\ " --exp-suffix ")" --no-print-csv --acc-k 5 > "${sfile}")
            (set -x; echo "\n: Top-5 accuracy for AlphaNet using varying number of nearest neighbors (_k_) based on ${dist_desc}, with \\\\acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_${dist}_imagenetlt_resnext50_crt_rho_${rho}_top5}" >> "${sfile}")
        fi

        for level in 2 3 4 5 10; do
            sfile="paper/tables/appendix/nnsweep_${dist}/imagenetlt_resnext50_crt/rho_${rho}/acc_nn_semantic_${level}_adjusted_top1.md"
            if check_does_not_exist "${sfile}"; then
                (set -x; python run_printres.py --base-res-dir "results/nnsweep_${dist}/imagenetlt_resnext50_crt/rho_${rho}" --rel-exp-paths "k_1" "k_2" "k_3" "k_4" "k_5" "k_6" "k_7" "k_8" "k_9" "k_10" --exp-names "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" --res-files-pattern "rep_*/result.pth" --exp-prefix "AlphaNet (_k_\\ =\\ " --exp-suffix ")" --no-print-csv --show-adjusted-acc --adjusted-acc-semantic --adjusted-acc-semantic-nns-level ${level} --imagenet-data-root "data/ImageNetLT" > "${sfile}")
                (set -x; echo "\n: Accuracy computed by considering predictions within ${level} WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on ${dist_desc}, with \\\\acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_${dist}_imagenetlt_resnext50_crt_rho_${rho}_semantic${level}_adjusted_top1}" >> "${sfile}")
            fi
        done
    done
done
