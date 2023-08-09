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

BASE_SAVE_DIR="paper/tables"
CONTEXT_DEFAULT="paper"

################################################

while getopts "hc:" opt; do
    case ${opt} in
        c) CONTEXT="${OPTARG}";;
        h) echo "usage: run_gentables.sh [-h] [-c context]"
           echo "options:"
           echo "  -h: display this help message and exit"
           echo "  -c: context (['${CONTEXT_DEFAULT}'] or 'notebook')"
           exit;;
        \?) echo "usage: run_genplots.sh [-h] [-c context]"
            exit 1;;
    esac
done

context=${CONTEXT:-"${CONTEXT_DEFAULT}"}
if [ "${context}" = "paper" ]; then
    save_dir="${BASE_SAVE_DIR}"
    xxargs=()
    alpha="\$\\\\alpha\$"
    balph="\$\\\\balph\$"
    rho_prefix="\$\\\\rho"
    rho_suffix="\$"
    kprefix="\$k"
    ksuffix="\$"
elif [ "${context}" = "notebook" ]; then
    save_dir="${BASE_SAVE_DIR}/_www"
    xxargs=("--for-web")
    alpha="_α_"
    balph="**_α_**"
    rho_prefix="_ρ_"
    rho_suffix=""
    kprefix="_k_"
    ksuffix=""
else
    echo "unknown context: ${context}"
    exit 1
fi

################################################
# SKIP

for acck in 1 5; do
    sfile="${save_dir}/datasets_split_top${acck}_accs_vs_model_tworho.md"

    # NOTE: SKIPPING
    # if check_does_not_exist "${sfile}"; then
    if false; then
        (set -x; echo "Method                        Few            Med.            Many         Overall" >> "${sfile}")
        (set -x; echo "-----------------  --------------  --------------  --------------  --------------" >> "${sfile}")

        for datasrc in "imagenetlt" "placeslt" "cifarlt"; do
            case $datasrc in
                imagenetlt)
                    datasrcname="ImageNet‑LT"
                    datasrcdir="ImageNetLT"
                    models=("resnext50_crt" "resnext50_lws" "resnext50_ride")
                    ;;
                placeslt)
                    datasrcname="Places‑LT"
                    datasrcdir="PlacesLT"
                    models=("resnet152_crt" "resnet152_lws")
                    ;;
                cifarlt)
                    datasrcname="CIFAR‑100‑LT"
                    datasrcdir="CIFARLT"
                    models=("resnet32_ride" "resnet34_ltr")
                    ;;
                *)
                    echo "how did you get here?" >&2
                    exit 1
                    ;;
            esac

            (set -x; echo "**${datasrcname}**" >> "${sfile}")

            for model in ${models[@]}; do
                case $model in
                    resnext50_crt|resnet152_crt)
                        modelname="cRT"
                        ;;
                    resnext50_lws|resnet152_lws)
                        modelname="LWS"
                        ;;
                    resnext50_ride|resnet32_ride)
                        modelname="RIDE"
                        ;;
                    resnet34_ltr)
                        modelname="LTR"
                        ;;
                    *)
                        echo "how did you get here?" >&2
                        exit 1
                        ;;
                esac

                (set -x; python run_printres.py --base-res-dir "data/${datasrcdir}/baselines" --rel-exp-paths "" --exp-names "${modelname}" --res-files-pattern "${model}.pkl" --no-print-csv --no-show-baselines --num-col-width 14 --name-col-width 17 --no-show-hdr "${xxargs[@]}" >> "${sfile}")
                (set -x; python run_printres.py --base-res-dir "results/tworho/${datasrc}_${model}/rho1_0.5_rho2_0.1" --rel-exp-paths "" --exp-names `echo "${alpha}‑${modelname}"` --res-files-pattern "rep_*/result.pth" --no-print-csv --no-show-baselines --num-col-width 14 --name-col-width 17 --no-show-hdr "${xxargs[@]}" >> "${sfile}")
                (set -x; echo "<!-- -->" >> "${sfile}")
            done

            (set -x; echo "<!-- -->" >> "${sfile}")
        done

        (set -x; head --lines=-2 "${sfile}" > "${sfile}.tmp")
        (set -x; mv "${sfile}.tmp" "${sfile}")
        (set -x; echo "\nTODO: Caption" >> "${sfile}")
    fi
done

################################################

sfile="${save_dir}/datasets_split_accs_vs_rho_ltr.md"

if check_does_not_exist "${sfile}"; then
    (set -x; echo "Method                        Few            Med.            Many         Overall" >> "${sfile}")
    (set -x; echo "-----------------  --------------  --------------  --------------  --------------" >> "${sfile}")

    datasrc="cifarlt"
    datasrcdir="CIFARLT"
    datasrcname="CIFAR‑100‑LT"

    (set -x; echo "**${datasrcname}**" >> "${sfile}")

    model="resnet34_ltr"
    modelname="LTR"
    rhos=(rho_0.5 rho_1 rho_1.5)
    rhostrs=(0.5 1 1.5)

    (set -x; python run_printres.py --base-res-dir "data/${datasrcdir}/baselines" --rel-exp-paths "" --exp-names "${modelname}" --res-files-pattern "${model}.pkl" --no-print-csv --no-show-baselines --num-col-width 14 --name-col-width 17 --no-show-hdr "${xxargs[@]}" >> "${sfile}")
    (set -x; echo "${alpha}‑${modelname}" >> "${sfile}")
    (set -x; python run_printres.py --base-res-dir "results/main/${datasrc}_${model}" --rel-exp-paths "${rhos[@]}" --exp-names "${rhostrs[@]}" --res-files-pattern "rep_*/result.pth" --exp-prefix `echo "${rho_prefix}="` --exp-suffix "${rho_suffix}" --no-print-csv --no-show-baselines --no-show-hdr --num-col-width 14 --name-col-width 17 "${xxargs[@]}" >> "${sfile}")
    (set -x; echo "\n: Mean split accuracy in percents (standard deviation in superscript) on ${datasrcname} using the ${modelname} model. {#tbl:datasets_split_accs_vs_rho_ltr}" >> "${sfile}")
fi

################################################

sfile="${save_dir}/datasets_split_accs_vs_rho_ride.md"

if check_does_not_exist "${sfile}"; then
    (set -x; echo "Method                        Few            Med.            Many         Overall" >> "${sfile}")
    (set -x; echo "-----------------  --------------  --------------  --------------  --------------" >> "${sfile}")

    modelname="RIDE"
    rhos=(rho_0.5 rho_1 rho_1.5)
    rhostrs=(0.5 1 1.5)

    for datasrc in imagenetlt cifarlt; do
        if [ "${datasrc}" = "imagenetlt" ]; then
            model="resnext50_ride"
            datasrcdir="ImageNetLT"
            datasrcname="ImageNet‑LT"
        else
            model="resnet32_ride"
            datasrcdir="CIFARLT"
            datasrcname="CIFAR‑100‑LT"
        fi

        (set -x; echo "**${datasrcname}**" >> "${sfile}")
        (set -x; python run_printres.py --base-res-dir "data/${datasrcdir}/baselines" --rel-exp-paths "" --exp-names "${modelname}" --res-files-pattern "${model}.pkl" --no-print-csv --no-show-baselines --num-col-width 14 --name-col-width 17 --no-show-hdr "${xxargs[@]}" >> "${sfile}")
        (set -x; echo "${alpha}‑${modelname}" >> "${sfile}")
        (set -x; python run_printres.py --base-res-dir "results/main/${datasrc}_${model}" --rel-exp-paths "${rhos[@]}" --exp-names "${rhostrs[@]}" --res-files-pattern "rep_*/result.pth" --exp-prefix `echo "${rho_prefix}="` --exp-suffix "${rho_suffix}" --no-print-csv --no-show-baselines --no-show-hdr --num-col-width 14 --name-col-width 17 "${xxargs[@]}" >> "${sfile}")

        if [ "${datasrc}" = "imagenetlt" ]; then
            (set -x; echo "<!--  -->" >> "${sfile}")
        fi
    done

    (set -x; echo "\n: Mean split accuracy in percents (standard deviation in superscript) on ImageNet‑LT and CIFAR‑100‑LT using the ensemble RIDE model. ${alpha}‑RIDE applies AlphaNet on average features from the ensemble. {#tbl:datasets_split_accs_vs_rho_ride}" >> "${sfile}")
fi

################################################

sfile="${save_dir}/datasets_baselines_split_accs_vs_rho.md"

if check_does_not_exist "${sfile}"; then
    (set -x; echo "Method                        Few            Med.            Many         Overall" >> "${sfile}")
    (set -x; echo "-----------------  --------------  --------------  --------------  --------------" >> "${sfile}")

    rhos=(rho_0.5 rho_1 rho_1.5)
    rhostrs=(0.5 1 1.5)

    for datasrc in imagenetlt placeslt; do
        if [ "${datasrc}" = "imagenetlt" ]; then
            datasrcdir="ImageNetLT"
            datasrcname="ImageNet‑LT"
            models=(resnext50_crt resnext50_lws)
        else
            datasrcdir="PlacesLT"
            datasrcname="Places‑LT"
            models=(resnet152_crt resnet152_lws)
        fi

        (set -x; echo "**${datasrcname}**" >> "${sfile}")

        if [ "${datasrc}" = "imagenetlt" ]; then
            if [ "${context}" = "paper" ]; then
                (set -x; echo "NCM                        \$28.1\$          \$45.3\$          \$56.6\$          \$47.3\$" >> "${sfile}")
                (set -x; echo "\$\\\\tau\$‑normalized          \$30.7\$          \$46.9\$          \$59.1\$          \$49.4\$" >> "${sfile}")
            else
                (set -x; echo "NCM                          28.1            45.3            56.6            47.3" >> "${sfile}")
                (set -x; echo "_τ_‑normalized               30.7            46.9            59.1            49.4" >> "${sfile}")
            fi
        else
            if [ "${context}" = "paper" ]; then
                (set -x; echo "NCM                        \$27.3\$          \$37.1\$          \$40.4\$          \$36.4\$" >> "${sfile}")
                (set -x; echo "\$\\\\tau\$‑normalized          \$31.8\$          \$40.7\$          \$37.8\$          \$37.9\$" >> "${sfile}")
            else
                (set -x; echo "NCM                          27.3            37.1            40.4            36.4" >> "${sfile}")
                (set -x; echo "_τ_‑normalized               30.7            46.9            59.1            49.4" >> "${sfile}")
            fi
        fi

        for model in ${models[@]}; do
            case $model in
                resnext50_crt|resnet152_crt)
                    modelname="cRT"
                    ;;
                resnext50_lws|resnet152_lws)
                    modelname="LWS"
                    ;;
                *)
                    echo "bad model: ${model}" >&2
                    exit 1
                    ;;
            esac

            (set -x; echo "<!--  -->" >> "${sfile}")
            (set -x; python run_printres.py --base-res-dir "data/${datasrcdir}/baselines" --rel-exp-paths "" --exp-names "${modelname}" --res-files-pattern "${model}.pkl" --no-print-csv --no-show-baselines --num-col-width 14 --name-col-width 17 --no-show-hdr "${xxargs[@]}" >> "${sfile}")
            (set -x; echo "${alpha}‑${modelname}" >> "${sfile}")
            (set -x; python run_printres.py --base-res-dir "results/main/${datasrc}_${model}" --rel-exp-paths "${rhos[@]}" --exp-names "${rhostrs[@]}" --res-files-pattern "rep_*/result.pth" --exp-prefix `echo "${rho_prefix}="` --exp-suffix "${rho_suffix}" --no-print-csv --no-show-baselines --no-show-hdr --num-col-width 14 --name-col-width 17 "${xxargs[@]}" >> "${sfile}")
        done

        if [ "${datasrc}" = "imagenetlt" ]; then
            (set -x; echo "<!--  -->" >> "${sfile}")
            (set -x; echo "<!--  -->" >> "${sfile}")
        fi
    done

    (set -x; echo "\n: Mean split accuracy in percents (standard deviation in superscript) of AlphaNet and various baseline methods on ImageNet‑LT and Places‑LT. ${alpha}‑cRT and ${alpha}‑LWS are AlphaNet models applied over cRT and LWS features respectively. {#tbl:datasets_baselines_split_accs_vs_rhos}" >> "${sfile}")
fi

################################################

for acck in 1 5; do
    for datasrc in "imagenetlt" "cifarlt" "placeslt" "inat"; do
        sfile="${save_dir}/appendix/models_split_top${acck}_accs_vs_rho_${datasrc}.md"
        if check_does_not_exist "${sfile}"; then
            if [ "${datasrc}" = "imagenetlt" ]; then
                datasrcdir="ImageNetLT"
                datasrcname="ImageNet‑LT"
                models=("resnext50_crt" "resnext50_lws" "resnext50_ride")
            elif [ "${datasrc}" = "cifarlt" ]; then
                datasrcdir="CIFARLT"
                datasrcname="CIFAR‑100‑LT"
                models=("resnet32_ride" "resnet34_ltr")
            elif [ "${datasrc}" = "placeslt" ]; then
                datasrcdir="PlacesLT"
                datasrcname="Places‑LT"
                models=("resnet152_crt" "resnet152_lws")
            elif [ "${datasrc}" = "inat" ]; then
                datasrcdir="iNaturalist"
                datasrcname="iNaturalist"
                models=("resnet152_crt")
            else
                echo "bad datasrc: ${datasrc}" >&2
                exit 1
            fi

            if [ "${datasrc}" = "inat" ]; then
                rhos=(rho_0.01 rho_0.02 rho_0.03 rho_0.04 rho_0.05)
                rhostrs=(0.01 0.02 0.03 0.04 0.05)
                modelsdesc="AlphaNet applied to cRT"
            else
                rhos=(rho_0.1 rho_0.2 rho_0.3 rho_0.4 rho_0.5 rho_0.75 rho_1 rho_1.25 rho_1.5 rho_1.75 rho_2 rho_3)
                rhostrs=(0.1 0.2 0.3 0.4 0.5 0.75 1 1.25 1.5 1.75 2 3)
                modelsdesc="AlphaNet applied to different models"
            fi

            xargs=()
            for model in "${models[@]}"; do
                case $model in
                    resnext50_crt|resnet152_crt)
                        modelname="cRT"
                        ;;
                    resnext50_lws|resnet152_lws)
                        modelname="LWS"
                        ;;
                    resnext50_ride|resnet32_ride)
                        modelname="RIDE"
                        ;;
                    resnet34_ltr)
                        modelname="LTR"
                        ;;
                    *)
                        echo "bad model: ${model}" >&2
                        exit 1
                        ;;
                esac

                (set -x; python run_printres.py --base-res-dir "data/${datasrcdir}/baselines" --rel-exp-paths "" --exp-names "${modelname}" --res-files-pattern "${model}.pkl" --no-print-csv --no-show-baselines --acc-k ${acck} --num-col-width 14 --name-col-width 17 --exp-str Model "${xargs[@]}" "${xxargs[@]}" >> "${sfile}")
                (set -x; echo "${balph}**‑${modelname}**" >> "${sfile}")
                (set -x; python run_printres.py --base-res-dir "results/main/${datasrc}_${model}" --rel-exp-paths "${rhos[@]}" --exp-names "${rhostrs[@]}" --res-files-pattern "rep_*/result.pth" --exp-prefix `echo "${rho_prefix}="` --exp-suffix "${rho_suffix}" --no-print-csv --no-show-baselines --no-show-hdr --num-col-width 14 --name-col-width 17 --acc-k ${acck} "${xxargs[@]}" >> "${sfile}")
                (set -x; echo "<!--  -->" >> "${sfile}")

                xargs=("--no-show-hdr")
            done

            (set -x; head --lines=-1 "${sfile}" > "${sfile}.tmp")
            (set -x; mv "${sfile}.tmp" "${sfile}")
            (set -x; echo "\n: Top‑${acck} accuracy on ${datasrcname}, using ${modelsdesc}. {#tbl:models_split_top${acck}_accs_vs_rho_${datasrc}}" >> "${sfile}")
        fi
    done
done

################################################

rhos=(rho_0.1 rho_0.2 rho_0.3 rho_0.4 rho_0.5 rho_0.75 rho_1 rho_1.25 rho_1.5 rho_1.75 rho_2 rho_3)
rhostrs=(0.1 0.2 0.3 0.4 0.5 0.75 1 1.25 1.5 1.75 2 3)
sfile="${save_dir}/appendix/models_split_semantic4_accs_vs_rho_imagenetlt.md"
if check_does_not_exist "${sfile}"; then
    xargs=()
    for model in crt lws ride; do
        case $model in
            crt)
                modelname="cRT"
                ;;
            lws)
                modelname="LWS"
                ;;
            ride)
                modelname="RIDE"
                ;;
        esac

        (set -x; python run_printres.py --base-res-dir "data/ImageNetLT/baselines" --rel-exp-paths "" --exp-names "${modelname}" --res-files-pattern "resnext50_${model}.pkl" --no-print-csv --no-show-baselines --num-col-width 14 --name-col-width 17 --exp-str Model --show-adjusted-acc --adjusted-acc-semantic --adjusted-acc-semantic-nns-level 4 --imagenet-data-root "data/ImageNetLT" "${xargs[@]}" "${xxargs[@]}" >> "${sfile}")

        (set -x; echo "${balph}**‑${modelname}**" >> "${sfile}")
        (set -x; python run_printres.py --base-res-dir "results/main/imagenetlt_resnext50_${model}" --rel-exp-paths "${rhos[@]}" --exp-names "${rhostrs[@]}" --res-files-pattern "rep_*/result.pth" --exp-prefix `echo "${rho_prefix}="` --exp-suffix "${rho_suffix}" --no-print-csv --no-show-baselines --no-show-hdr --num-col-width 14 --name-col-width 17 --show-adjusted-acc --adjusted-acc-semantic --adjusted-acc-semantic-nns-level 4 --imagenet-data-root "data/ImageNetLT" "${xxargs[@]}" >> "${sfile}")
        (set -x; echo "<!--  -->" >> "${sfile}")

        xargs=("--no-show-hdr")
    done

    (set -x; head --lines=-1 "${sfile}" > "${sfile}.tmp")
    (set -x; mv "${sfile}.tmp" "${sfile}")
    (set -x; echo "\n: ImageNet‑LT accuracy computed by considering predictions within 4 WordNet nodes as correct, for AlphaNet applied to different models. {#tbl:models_split_semantic4_accs_vs_rho_imagenetlt}" >> "${sfile}")
fi

################################################

for dist in "euclidean" "cosine"; do
    if [ ${dist} = "euclidean" ]; then
        dist_desc="Euclidean distance"
    else
        dist_desc="cosine distance"
    fi

    for acck in 1 5; do
        sfile="${save_dir}/appendix/rhos_split_top${acck}_accs_vs_k_imagenetlt_crt_${dist}.md"
        if check_does_not_exist "${sfile}"; then
            (set -x; python run_printres.py --base-res-dir "data/ImageNetLT/baselines" --rel-exp-paths "" --exp-names "cRT" --res-files-pattern "resnext50_crt.pkl" --no-print-csv --no-show-baselines --acc-k ${acck} --num-col-width 14 --name-col-width 17 --exp-str Model "${xxargs[@]}" > "${sfile}")
            (set -x; echo "<!--  -->" >> "${sfile}")
            (set -x; echo "${balph}**‑cRT**" >> "${sfile}")
            for rho in "0.25" "0.5" "1" "2"; do
                (set -x; echo "${rho_prefix}=${rho}${rho_suffix}" >> "${sfile}")
                (set -x; python run_printres.py --base-res-dir "results/nnsweep_${dist}/imagenetlt_resnext50_crt/rho_${rho}" --rel-exp-paths "k_1" "k_2" "k_3" "k_4" "k_5" "k_6" "k_7" "k_8" "k_9" "k_10" --exp-names "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" --res-files-pattern "rep_*/result.pth" --exp-prefix "${kprefix}=" --exp-suffix "${ksuffix}" --no-print-csv --no-show-baselines --no-show-hdr --num-col-width 14 --name-col-width 17 --acc-k ${acck} "${xxargs[@]}" >> "${sfile}")
                if [ "${rho}" != "2" ]; then
                    (set -x; echo "<!--  -->" >> "${sfile}")
                fi
            done

            (set -x; echo "\n: Per-split test top‑${acck} accuracies for ${alpha}‑cRT on ImageNet‑LT using ${kprefix}${ksuffix} nearest neighbors based on ${dist_desc}. {#tbl:rhos_split_top1_accs_vs_k_imagenetlt_crt_cosine}" >> "${sfile}")
        fi
    done
done
