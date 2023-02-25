#!/usr/bin/env sh

set -e

N_BOOT=10000
SAVE_DIR="figures"
REP="*"

mkdir -p "${SAVE_DIR}"

for i in 0 1 2; do
    if [ ${i} -eq 0 ]; then
        context="paper"
        dfont="Latin Modern Math"
        mfont="cm"
        theme="light"
        ext=".pdf"
    else
        context="notebook"
        dfont="Latin Modern Sans"
        mfont="stixsans"
        if [ ${i} -eq 1 ]; then
            theme="light"
            ext=".svg"
        else
            theme="dark"
            ext="_dark.svg"
        fi
    fi

    (set -x; python run_makeplot.py PlotSplitAccVsExp --base-res-dir "results" --exp-sub-dir "main/imagenetlt_resnext50_crt/rho_0.1" "main/imagenetlt_resnext50_crt/rho_0.25" "main/imagenetlt_resnext50_crt/rho_0.5" "main/imagenetlt_resnext50_crt/rho_0.75" "main/imagenetlt_resnext50_crt/rho_1" "main/imagenetlt_resnext50_crt/rho_1.25" "main/imagenetlt_resnext50_crt/rho_1.5" "main/imagenetlt_resnext50_crt/rho_1.75" "main/imagenetlt_resnext50_crt/rho_2" "randomnns/imagenetlt_resnext50_crt/rho_0.1" "randomnns/imagenetlt_resnext50_crt/rho_0.25" "randomnns/imagenetlt_resnext50_crt/rho_0.5" "randomnns/imagenetlt_resnext50_crt/rho_0.75" "randomnns/imagenetlt_resnext50_crt/rho_1" "randomnns/imagenetlt_resnext50_crt/rho_1.25" "randomnns/imagenetlt_resnext50_crt/rho_1.5" "randomnns/imagenetlt_resnext50_crt/rho_1.75" "randomnns/imagenetlt_resnext50_crt/rho_2" --exp-names "0.10" "0.25" "0.50" "0.75" "1.00" "1.25" "1.50" "1.75" "2.00" "0.10" "0.25" "0.50" "0.75" "1.00" "1.25" "1.50" "1.75" "2.00" --xlabel "$\\rho$" --res-files-pattern "rep_${REP}/result.pth" --n-boot ${N_BOOT} --col "metric" --y "acc_delta" --plot:width "full" --plot:aspect 3 --plot:theme "${theme}" --plot:context "${context}" --plot:font:default "${dfont}" --plot:font:math "${mfont}" --legend-loc "upper right" --legend-bbox-to-anchor 0.99 0.85 --plot:file "${SAVE_DIR}/imagenetlt_crt_euclidean_random_split_deltas_vs_rho${ext}";)

    (set -x; python run_makeplot.py PlotClsAccDeltaBySplit --base-res-dir "results/main/imagenetlt_resnext50_crt" --exp-sub-dirs "rho_0.5" --res-files-pattern "rep_${REP}/result.pth" --n-boot ${N_BOOT} --plot:width "full" --plot:aspect 3 --plot:theme "${theme}" --plot:context "${context}" --plot:font:default "${dfont}" --plot:font:math "${mfont}" --plot:file "${SAVE_DIR}/imagenetlt_crt_rho_0.5_cls_deltas${ext}";)

    if [ ${i} -eq 0 ]; then
        width="3"
    else
        width="4.5"
    fi

    (set -x; python run_makeplot.py PlotSplitClsAccDeltaVsNNDist --base-res-dir "data/ImageNetLT/baselines" --exp-sub-dirs "" --res-files-pattern "resnext50_crt.pkl" --n-boot ${N_BOOT} --splits "few" --acc "baseline" --r-font-size "small" --plot:aspect 1 --plot:width "${width}" --plot:theme "${theme}" --plot:context "${context}" --plot:font:default "${dfont}" --plot:font:math "${mfont}" --rasterize-scatter --plot-params:xlim 8 20 --plot-params:xticks 10 12 14 16 18 --plot-params:ylim -0.1 0.9 --plot-params:yticks 0 0.2 0.4 0.6 0.8 --plot-params:ylabel "Accuracy" --plot:file "${SAVE_DIR}/imagenetlt_crt_baseline_cls_acc_vs_nndist${ext}")

    (set -x; python run_makeplot.py PlotSplitClsAccDeltaVsNNDist --base-res-dir "results/main/imagenetlt_resnext50_crt" --exp-sub-dirs "rho_0.5" --res-files-pattern "rep_${REP}/result.pth" --n-boot ${N_BOOT} --splits "few" --acc "delta" --r-font-size "small" --plot:aspect 1 --plot:width "${width}" --plot:theme "${theme}" --plot:context "${context}" --plot:font:default "${dfont}" --plot:font:math "${mfont}" --rasterize-scatter --plot-params:xlim 8 20 --plot-params:xticks 10 12 14 16 18 --plot-params:ylim -0.2 0.6 --plot-params:yticks -0.1 0 0.1 0.2 0.3 0.4 0.5 --plot:file "${SAVE_DIR}/imagenetlt_crt_rho_0.5_cls_delta_vs_nndist${ext}";)

    (set -x; python run_makeplot.py PlotSplitClsAccDeltaVsNNDist --base-res-dir "results/randomnns/imagenetlt_resnext50_crt" --exp-sub-dirs "rho_0.5" --res-files-pattern "rep_${REP}/result.pth" --n-boot ${N_BOOT} --splits "few" --acc "delta" --r-font-size "small" --plot:aspect 1 --plot:width "${width}" --plot:theme "${theme}" --plot:context "${context}" --plot:font:default "${dfont}" --plot:font:math "${mfont}" --rasterize-scatter --plot-params:xlim 16 28 --plot-params:xticks 18 20 22 24 26 --plot-params:ylim -0.2 0.6 --plot-params:yticks -0.1 0 0.1 0.2 0.3 0.4 0.5 --plot:file "${SAVE_DIR}/imagenetlt_crt_randomnns_rho_0.5_cls_delta_vs_nndist${ext}";)

    (set -x; python run_makeplot.py PlotPredChanges --base-res-dir "results/nnsweep_euclidean/imagenetlt_resnext50_crt" --exp-sub-dirs "rho_0.25" "rho_1" "rho_2" --exp-names "$\\rho=0.25$" "$\\rho=1$" "$\\rho=2$" --res-files-pattern "k_10/rep_${REP}/result.pth" --n-boot ${N_BOOT} --plot:width "full" --plot:aspect 3:2 --plot:theme "${theme}" --plot:context "${context}" --plot:font:default "${dfont}" --plot:font:math "${mfont}" --label-size "xx-small" --plot:file "${SAVE_DIR}/imagenetlt_crt_k_10_rhos_pred_changes${ext}";)

    for rho in "0.25" "0.5" "1" "2"; do
        (set -x; python run_makeplot.py PlotSplitAccVsExp --base-res-dir "results" --exp-sub-dirs "nnsweep_euclidean/imagenetlt_resnext50_crt/rho_${rho}/k_2" "nnsweep_euclidean/imagenetlt_resnext50_crt/rho_${rho}/k_3" "nnsweep_euclidean/imagenetlt_resnext50_crt/rho_${rho}/k_4" "nnsweep_euclidean/imagenetlt_resnext50_crt/rho_${rho}/k_5" "nnsweep_euclidean/imagenetlt_resnext50_crt/rho_${rho}/k_6" "nnsweep_euclidean/imagenetlt_resnext50_crt/rho_${rho}/k_7" "nnsweep_euclidean/imagenetlt_resnext50_crt/rho_${rho}/k_8" "nnsweep_euclidean/imagenetlt_resnext50_crt/rho_${rho}/k_9" "nnsweep_euclidean/imagenetlt_resnext50_crt/rho_${rho}/k_10" "nnsweep_cosine/imagenetlt_resnext50_crt/rho_${rho}/k_2" "nnsweep_cosine/imagenetlt_resnext50_crt/rho_${rho}/k_3" "nnsweep_cosine/imagenetlt_resnext50_crt/rho_${rho}/k_4" "nnsweep_cosine/imagenetlt_resnext50_crt/rho_${rho}/k_5" "nnsweep_cosine/imagenetlt_resnext50_crt/rho_${rho}/k_6" "nnsweep_cosine/imagenetlt_resnext50_crt/rho_${rho}/k_7" "nnsweep_cosine/imagenetlt_resnext50_crt/rho_${rho}/k_8" "nnsweep_cosine/imagenetlt_resnext50_crt/rho_${rho}/k_9" "nnsweep_cosine/imagenetlt_resnext50_crt/rho_${rho}/k_10" --exp-names "2" "3" "4" "5" "6" "7" "8" "9" "10" "2" "3" "4" "5" "6" "7" "8" "9" "10" --res-files-pattern "rep_${REP}/result.pth" --n-boot ${N_BOOT} --col "metric" --y "acc" --xlabel "\$k$" --plot:width "full" --plot:aspect 3 --plot:theme "${theme}" --plot:context "${context}" --plot:font:default "${dfont}" --plot:font:math "${mfont}" --legend-loc "upper right" --legend-bbox-to-anchor 0.99 0.85 --plot:file "${SAVE_DIR}/imagenetlt_crt_rho_${rho}_euclidean_cosine_split_accs_vs_k${ext}";)

        for dist in euclidean cosine; do
            (set -x; python run_makeplot.py PlotAlphaDist --base-res-dir "results/nnsweep_${dist}/imagenetlt_resnext50_crt/rho_${rho}" --exp-sub-dirs "k_2" "k_3" "k_4" "k_5" "k_6" "k_7" "k_8" "k_9" "k_10" --exp-names "\$k=2$" "\$k=3$" "\$k=4$" "\$k=5$" "\$k=6$" "\$k=7$" "\$k=8$" "\$k=9$" "\$k=10$" --res-files-pattern "rep_${REP}/result.pth" --n-boot ${N_BOOT} --col-wrap 5 --legend-loc "center" --legend-bbox-to-anchor 0.9 0.25 --legend-ncols 2 --plot:width "full" --plot:aspect 2.5 --plot:theme "${theme}" --plot:context "${context}" --plot:font:default "${dfont}" --plot:font:math "${mfont}" --plot:file "${SAVE_DIR}/imagenetlt_crt_rho_${rho}_ks_alpha_dists${ext}";)
        done
    done

    for dataset in imagenetlt_resnext50_crt imagenetlt_resnext50_lws imagenetlt_resnext50_ride placeslt_resnet152_crt placeslt_resnet152_lws cifar100_resnet32_ltr cifar100_resnet32_ride; do
        (set -x; python run_makeplot.py PlotAlphaDist --base-res-dir "results/main/${dataset}" --exp-sub-dirs "rho_0.1" "rho_0.25" "rho_0.5" "rho_0.75" "rho_1" "rho_1.25" "rho_1.5" "rho_1.75" "rho_2" --exp-names "$\\rho=0.1$" "$\\rho=0.25$" "$\\rho=0.5$" "$\\rho=0.75$" "$\\rho=1$" "$\\rho=1.25$" "$\\rho=1.5$" "$\\rho=1.75$" "$\\rho=2$" --res-files-pattern "rep_${REP}/result.pth" --n-boot ${N_BOOT} --col-wrap 5 --legend-loc "center" --legend-bbox-to-anchor 0.9 0.25 --legend-ncols 2 --plot:width "full" --plot:aspect 2.5 --plot:theme "${theme}" --plot:context "${context}" --plot:font:default "${dfont}" --plot:font:math "${mfont}" --plot:file "${SAVE_DIR}/${dataset}_rhos_alpha_dists${ext}";)

        (set -x; python run_makeplot.py PlotSplitClsAccDeltaVsNNDist --base-res-dir "results/main/${dataset}" --exp-sub-dirs "rho_0.1" "rho_0.25" "rho_0.5" "rho_0.75" "rho_1" "rho_1.25" "rho_1.5" "rho_1.75" "rho_2" --exp-names "$\\rho=0.1$" "$\\rho=0.25$" "$\\rho=0.5$" "$\\rho=0.75$" "$\\rho=1$" "$\\rho=1.25$" "$\\rho=1.5$" "$\\rho=1.75$" "$\\rho=2$" --res-files-pattern "rep_${REP}/result.pth" --n-boot ${N_BOOT} --splits "few" "base" --col-wrap 5 --acc "delta" --plot:width "full" --plot:aspect "5:2" --plot:theme "${theme}" --plot:context "${context}" --plot:font:default "${dfont}" --plot:font:math "${mfont}" --rasterize-scatter --plot-params:xticks --plot-params:yticks --plot-r-loc 0.99 0.99 --legend-loc "upper right" --legend-bbox-to-anchor 1 0.5 --legend-ncols 1 --add-axes-guide --no-despine --plot-params:ylabel "Accuracy change" --plot:file "${SAVE_DIR}/${dataset}_rhos_cls_delta_vs_nndist${ext}")
    done
done
