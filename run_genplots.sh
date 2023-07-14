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

BASE_SAVE_DIR="paper/figures"
CONTEXT_DEFAULT="paper"
THEME_DEFAULT="light"
REP_DEFAULT="*"
N_BOOT_DEFAULT=10000

################################################

while getopts "hc:t:r:n:" opt; do
    case ${opt} in
        c) CONTEXT="${OPTARG}";;
        t) THEME="${OPTARG}";;
        r) REP="${OPTARG}";;
        n) N_BOOT="${OPTARG}";;
        h) echo "usage: run_genplots.sh [-h] [-c context] [-t theme] [-r res_rep] [-n n_boot]"
           echo "options:"
           echo "  -h: display this help message and exit"
           echo "  -c: context (['${CONTEXT_DEFAULT}'] or 'notebook')"
           echo "  -t: theme (['${THEME_DEFAULT}'] or 'dark')"
           echo "  -r: result repetition to use (default: '${REP_DEFAULT}', all repetitions)"
           echo "  -n: number of bootstraps (default: ${N_BOOT_DEFAULT})"
           exit;;
        \?) echo "usage: run_genplots.sh [-h] [-c context] [-t theme] [-r res_rep] [-n n_boot]"
            exit 1;;
    esac
done

context=${CONTEXT:-"${CONTEXT_DEFAULT}"}
theme=${THEME:-"${THEME_DEFAULT}"}
rep=${REP:-"${REP_DEFAULT}"}
n_boot=${N_BOOT:-${N_BOOT_DEFAULT}}

################################################

if [ "${context}" = "paper" ]; then
    dfont="serif"
    mfont="stix"
    if [ "${theme}" = "light" ]; then
        ext=".pgf"
        xargs=("--plot:bg" "#ffffff" "--plot:fg-primary" "#000000" "--plot:fg-secondary" "#bbbbbb")
    else
        ext="_dark.pgf"
        xargs=("--plot:bg" "#000000" "--plot:fg-primary" "#ffffff" "--plot:fg-secondary" "#444444")
    fi
    save_dir="${BASE_SAVE_DIR}"
else
    dfont="sans-serif"
    mfont="stixsans"
    xargs=("--plot:font:sans-serif" "system-ui" " -apple-system" "Segoe UI" "Roboto" "Helvetica Neue" "Noto Sans" "Liberation Sans" "Arial" "sans-serif")
    if [ "${theme}" = "light" ]; then
        ext=".svg"
    else
        ext="_dark.svg"
    fi
    save_dir="${BASE_SAVE_DIR}/_www"
fi

mainargs=("--plot:theme" "${theme}" "--plot:context" "${context}" "--plot:font:default" "${dfont}" "--plot:font:math" "${mfont}")

################################################

if [ "${context}" = "paper" ]; then width="2"; else width="2.5"; fi

sfile="${save_dir}/doggies${ext}"
if check_does_not_exist "${sfile}"; then
    (set -x; python run_makeplot.py PlotClassExamples --srcs "paper/figures/_src/lhasa.png" "paper/figures/_src/terrier.png" --labels "Lhasa" "Tibetan Terrier" --plot:dpi 300 --plot:width "${width}" --plot:aspect 1.5 --plot:file "${sfile}" "${mainargs[@]}" "${xargs[@]}";)
fi

sfile="${save_dir}/pred_counts_imagenetlt_crt_baseline${ext}"
if check_does_not_exist "${sfile}"; then
    (set -x; python run_makeplot.py PlotPredCounts --base-res-dir "data/ImageNetLT/baselines" --exp-sub-dirs "" --res-files-pattern "resnext50_crt.pkl" --n-boot ${n_boot} --plot:width "${width}" --plot:aspect 1.25 --plot:file "${sfile}" "${mainargs[@]}" "${xargs[@]}";)
fi

sfile="${save_dir}/cls_acc_vs_nndist_imagenetlt_crt_baseline${ext}"
if check_does_not_exist "${sfile}"; then
    (set -x; python run_makeplot.py PlotSplitClsAccDeltaVsNNDist --base-res-dir "data/ImageNetLT/baselines" --exp-sub-dirs "" --res-files-pattern "resnext50_crt.pkl" --n-boot ${n_boot} --splits "few" --acc "baseline" --r-font-size "medium" --plot-r-loc 0.9 0.95 --plot:aspect 1 --plot:width "${width}" --rasterize-scatter --plot-params:xlim 8 20 --plot-params:xticks 10 12 14 16 18 --plot-params:xticklabels "\$10\$" "\$12\$" "\$14\$" "\$16\$" "\$18\$" --plot-params:ylim -0.1 0.9 --plot-params:yticks 0 0.2 0.4 0.6 0.8 --plot-params:yticklabels "\$0\$" "\$0.2\$" "\$0.4\$" "\$0.6\$" "\$0.8\$" --plot-params:ylabel "Accuracy" --plot:file "${sfile}" "${mainargs[@]}" "${xargs[@]}";)
fi

sfile="${save_dir}/cls_delta_vs_nndist_imagenetlt_crt_rho_05${ext}"
if check_does_not_exist "${sfile}"; then
    (set -x; python run_makeplot.py PlotSplitClsAccDeltaVsNNDist --base-res-dir "results/main/imagenetlt_resnext50_crt" --exp-sub-dirs "rho_0.5" --res-files-pattern "rep_${rep}/result.pth" --n-boot ${n_boot} --splits "few" --acc "delta" --r-font-size "medium" --plot-r-loc 0.9 0.95 --plot:aspect 1 --plot:width "${width}" --rasterize-scatter --plot-params:xlim 8 20 --plot-params:xticks 10 12 14 16 18 --plot-params:xticklabels "\$10\$" "\$12\$" "\$14\$" "\$16\$" "\$18\$" --plot-params:ylim -0.2 0.6 --plot-params:yticks -0.1 0 0.1 0.2 0.3 0.4 0.5 --plot-params:yticklabels "\$-0.1\$" "\$0\$" "\$0.1\$" "\$0.2\$" "\$0.3\$" "\$0.4\$" "\$0.5\$" --plot:file "${sfile}" "${mainargs[@]}" "${xargs[@]}";)
fi

sfile="${save_dir}/euclidean_random_split_deltas_vs_rho_imagenetlt_crt${ext}"
if check_does_not_exist "${sfile}"; then
    (set -x; python run_makeplot.py PlotSplitAccVsExp --base-res-dir "results" --exp-sub-dir "main/imagenetlt_resnext50_crt/rho_0.1" "main/imagenetlt_resnext50_crt/rho_0.25" "main/imagenetlt_resnext50_crt/rho_0.5" "main/imagenetlt_resnext50_crt/rho_0.75" "main/imagenetlt_resnext50_crt/rho_1" "main/imagenetlt_resnext50_crt/rho_1.25" "main/imagenetlt_resnext50_crt/rho_1.5" "main/imagenetlt_resnext50_crt/rho_1.75" "main/imagenetlt_resnext50_crt/rho_2" "randomnns/imagenetlt_resnext50_crt/rho_0.1" "randomnns/imagenetlt_resnext50_crt/rho_0.25" "randomnns/imagenetlt_resnext50_crt/rho_0.5" "randomnns/imagenetlt_resnext50_crt/rho_0.75" "randomnns/imagenetlt_resnext50_crt/rho_1" "randomnns/imagenetlt_resnext50_crt/rho_1.25" "randomnns/imagenetlt_resnext50_crt/rho_1.5" "randomnns/imagenetlt_resnext50_crt/rho_1.75" "randomnns/imagenetlt_resnext50_crt/rho_2" --exp-names "\$0.10\$" "\$0.25\$" "\$0.50\$" "\$0.75\$" "\$1.00\$" "\$1.25\$" "\$1.50\$" "\$1.75\$" "\$2.00\$" "\$0.10\$" "\$0.25\$" "\$0.50\$" "\$0.75\$" "\$1.00\$" "\$1.25\$" "\$1.50\$" "\$1.75\$" "\$2.00\$" --xlabel "$\\rho$" --res-files-pattern "rep_${rep}/result.pth" --n-boot ${n_boot} --col "metric" --y "acc_delta" --plot:width "full" --plot:aspect 3 --legend-loc "upper right" --legend-bbox-to-anchor 0.99 0.85 --plot:file "${sfile}" "${mainargs[@]}" "${xargs[@]}";)
fi

sfile="${save_dir}/rhos_few_pred_changes_nn_base_imagenetlt_crt${ext}"
if check_does_not_exist "${sfile}"; then
    (set -x; python run_makeplot.py PlotPredChanges --base-res-dir "results/main/imagenetlt_resnext50_crt" --exp-sub-dirs "rho_0.5" "rho_1" "rho_1.5" --exp-names "$\\rho=0.5$" "$\\rho=1$" "$\\rho=1.5$" --res-files-pattern "rep_${rep}/result.pth" --n-boot ${n_boot} --plot:width "full" --plot:aspect "3:2" --label-size "x-small" --split "few" --nn-split "base" --plot:file "${sfile}" "${mainargs[@]}" "${xargs[@]}";)
fi

for split in few base all; do
    sfile="${save_dir}/${split}_pred_changes_nn_semantic4_imagenetlt_crt_rho_05${ext}"
    if check_does_not_exist "${sfile}"; then
        (set -x; python run_makeplot.py PlotPredChanges --base-res-dir "results/main/imagenetlt_resnext50_crt" --exp-sub-dirs "rho_0.5" --res-files-pattern "rep_${rep}/result.pth" --n-boot ${n_boot} --plot:width "${width}" --plot:aspect "1:2" --label-size "x-small" --split "${split}" --nn-split "semantic" --semantic-nns-level 4 --imagenet-data-root "data/ImageNetLT" --plot:file "${sfile}" "${mainargs[@]}" "${xargs[@]}";)
    fi
done

################################################

for rho in "0.25" "0.5" "1" "2"; do
    case $rho in
        0.25)
            rhostr="025"
            ;;
        0.5)
            rhostr="05"
            ;;
        *)
            rhostr="${rho}"
            ;;
    esac

    sfile="${save_dir}/appendix/euclidean_cosine_split_accs_vs_k_imagenetlt_crt_rho_${rhostr}${ext}"
    if check_does_not_exist "${sfile}"; then
        (set -x; python run_makeplot.py PlotSplitAccVsExp --base-res-dir "results" --exp-sub-dirs "nnsweep_euclidean/imagenetlt_resnext50_crt/rho_${rho}/k_1" "nnsweep_euclidean/imagenetlt_resnext50_crt/rho_${rho}/k_2" "nnsweep_euclidean/imagenetlt_resnext50_crt/rho_${rho}/k_3" "nnsweep_euclidean/imagenetlt_resnext50_crt/rho_${rho}/k_4" "nnsweep_euclidean/imagenetlt_resnext50_crt/rho_${rho}/k_5" "nnsweep_euclidean/imagenetlt_resnext50_crt/rho_${rho}/k_6" "nnsweep_euclidean/imagenetlt_resnext50_crt/rho_${rho}/k_7" "nnsweep_euclidean/imagenetlt_resnext50_crt/rho_${rho}/k_8" "nnsweep_euclidean/imagenetlt_resnext50_crt/rho_${rho}/k_9" "nnsweep_euclidean/imagenetlt_resnext50_crt/rho_${rho}/k_10" "nnsweep_cosine/imagenetlt_resnext50_crt/rho_${rho}/k_1" "nnsweep_cosine/imagenetlt_resnext50_crt/rho_${rho}/k_2" "nnsweep_cosine/imagenetlt_resnext50_crt/rho_${rho}/k_3" "nnsweep_cosine/imagenetlt_resnext50_crt/rho_${rho}/k_4" "nnsweep_cosine/imagenetlt_resnext50_crt/rho_${rho}/k_5" "nnsweep_cosine/imagenetlt_resnext50_crt/rho_${rho}/k_6" "nnsweep_cosine/imagenetlt_resnext50_crt/rho_${rho}/k_7" "nnsweep_cosine/imagenetlt_resnext50_crt/rho_${rho}/k_8" "nnsweep_cosine/imagenetlt_resnext50_crt/rho_${rho}/k_9" "nnsweep_cosine/imagenetlt_resnext50_crt/rho_${rho}/k_10" --exp-names "\$1\$" "\$2\$" "\$3\$" "\$4\$" "\$5\$" "\$6\$" "\$7\$" "\$8\$" "\$9\$" "\$10\$" "\$1\$" "\$2\$" "\$3\$" "\$4\$" "\$5\$" "\$6\$" "\$7\$" "\$8\$" "\$9\$" "\$10\$" --res-files-pattern "rep_${rep}/result.pth" --n-boot ${n_boot} --col "metric" --y "acc" --xlabel "\$k$" --plot:width "full" --plot:aspect 3 --legend-loc "upper right" --legend-bbox-to-anchor 0.99 0.85 --plot:file "${sfile}" "${mainargs[@]}" "${xargs[@]}";)
    fi
done

################################################

if [ "${context}" = "paper" ]; then width="5"; else width="6.25"; fi

for dataset in imagenetlt_resnext50_crt imagenetlt_resnext50_lws imagenetlt_resnext50_ride placeslt_resnet152_crt placeslt_resnet152_lws cifarlt_resnet32_ride cifarlt_resnet34_ltr; do
    if [ "${dataset}" = "cifarlt_resnet34_ltr" ]; then
        expdirs=("rho_1" "rho_2" "rho_3")
        expnames=("$\\rho=1$" "$\\rho=2$" "$\\rho=3$")
    else
        expdirs=("rho_0.5" "rho_1" "rho_1.5")
        expnames=("$\\rho=0.5$" "$\\rho=1$" "$\\rho=1.5$")
    fi
    dname=$(echo ${dataset} | cut -d_ -f1,3)

    sfile="${save_dir}/appendix/rhos_cls_deltas_${dname}${ext}"
    if check_does_not_exist "${sfile}"; then
        (set -x; python run_makeplot.py PlotClsAccDeltaBySplit --base-res-dir "results/main/${dataset}" --exp-sub-dirs "${expdirs[@]}" --exp-names "${expnames[@]}" --res-files-pattern "rep_${rep}/result.pth" --n-boot ${n_boot} --plot:width "${width}" --plot:aspect 1 --plot:file "${sfile}" "${mainargs[@]}" "${xargs[@]}";)
    fi

    case ${dataset} in
        imagenetlt_resnext50_ride|placeslt_resnet152_lws|cifarlt_resnet34_ltr)
            xxargs=("--add-axes-guide" "--legend-loc" "center" "--legend-bbox-to-anchor" "0.91" "0.5" "--legend-ncols" "1")
            ;;
        *)
            xxargs=("--no-add-axlabels")
            ;;
    esac

    sfile="${save_dir}/appendix/rhos_cls_delta_vs_nndist_${dname}${ext}"
    if check_does_not_exist "${sfile}"; then
        (set -x; python -O run_makeplot.py PlotSplitClsAccDeltaVsNNDist --base-res-dir "results/main/${dataset}" --exp-sub-dirs "${expdirs[@]}" --exp-names "${expnames[@]}" --res-files-pattern "rep_${rep}/result.pth" --n-boot ${n_boot} --splits "few" "base" --acc "delta" --plot:width "full" --plot:aspect 5 --rasterize-scatter --plot-params:xticks --plot-params:yticks --plot-r-loc 0.99 0.99 --no-despine --plot-params:ylabel "Accuracy change" --plot:file "${sfile}" --add-dummy-before --add-dummy-after "${xxargs[@]}" "${mainargs[@]}" "${xargs[@]}";)
    fi
done
