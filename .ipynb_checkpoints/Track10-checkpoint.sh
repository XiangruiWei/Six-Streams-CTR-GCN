#!/bin/bash

# ==============================================================================
# track10.sh
# ==============================================================================

# 定义设备编号
DEVICES=(0 1 2 3)  # 可用的GPU设备
NGPUS=${#DEVICES[@]}  # GPU数量

# 定义项目根目录
PROJECT_DIR=$(pwd)

# 定义模型目录路径
PROCESS_DATA_DIR="$PROJECT_DIR/Process_data"
MODEL_INFERENCE_DIR="$PROJECT_DIR/Model_inference"
MIX_GCN_DIR="$MODEL_INFERENCE_DIR/Mix_GCN"
MIX_FORMER_DIR="$MODEL_INFERENCE_DIR/Mix_Former"
TEST_DATASET_DIR="$PROJECT_DIR/Test_dataset"
CHECKPOINTS_DIR="$PROJECT_DIR/checkpoints"

# 创建输出和检查点目录（如果不存在）
mkdir -p "$CHECKPOINTS_DIR"
mkdir -p "$MIX_GCN_DIR/dataset"
mkdir -p "$MIX_FORMER_DIR/dataset"

# 定义输出日志文件
LOG_FILE="$PROJECT_DIR/output_mix.txt"
> "$LOG_FILE"

# 函数: 输出并记录日志
log() {
    echo "$1" | tee -a "$LOG_FILE"
}

# 输出开始信息
log "==============================="
log "训练过程开始于 $(date)"
log "==============================="

# ==============================================================================
# 函数：训练 Mix_GCN 模型
# ==============================================================================
train_mix_gcn() {
    echo "============================="
    echo "开始训练 Mix_GCN 模型"
    echo "============================="

    # 进入 Mix_GCN 目录
    cd "$MIX_GCN_DIR"

    CONFIGS=(
        "ctrgcn_V2_J.yaml"
        "ctrgcn_V2_B.yaml"
        "ctrgcn_V2_JM.yaml"
        "ctrgcn_V2_BM.yaml"
        "ctrgcn_V2_J_3d.yaml"
        "ctrgcn_V2_B_3d.yaml"
        "ctrgcn_V2_JM_3d.yaml"
        "ctrgcn_V2_BM_3d.yaml"
    )

    idx=0
    total=${#CONFIGS[@]}
    declare -A PIDS  # 存储PID与对应的GPU设备

    while [ $idx -lt $total ] || [ ${#PIDS[@]} -gt 0 ]; do
        # 启动新的任务（如果有可用的GPU）
        while [ ${#PIDS[@]} -lt $NGPUS ] && [ $idx -lt $total ]; do
            CONFIG=${CONFIGS[$idx]}
            DEVICE=${DEVICES[$((idx % NGPUS))]}
            echo "训练 $CONFIG 使用设备 $DEVICE..."
            CUDA_VISIBLE_DEVICES=$DEVICE python main.py --config ./config/$CONFIG --device 0 &
            PID=$!
            PIDS[$PID]=$DEVICE
            idx=$((idx+1))
        done

        # 等待任意一个进程结束
        if [ ${#PIDS[@]} -gt 0 ]; then
            wait -n
            # 检查并移除已完成的进程
            for PID in "${!PIDS[@]}"; do
                if ! kill -0 $PID 2>/dev/null; then
                    echo "进程 $PID 已完成"
                    unset PIDS[$PID]
                fi
            done
        fi
    done

    echo "Mix_GCN 模型训练完成。"

    # 返回项目根目录
    cd "$PROJECT_DIR"
}

# ==============================================================================
# 函数：训练 Mix_Former 模型
# ==============================================================================
train_mix_former() {
    echo "============================="
    echo "开始训练 Mix_Former 模型"
    echo "============================="

    # 进入 Mix_Former 目录
    cd "$MIX_FORMER_DIR"

    CONFIGS=(
        "mixformer_V2_J.yaml"
        "mixformer_V2_B.yaml"
        "mixformer_V2_JM.yaml"
        "mixformer_V2_BM.yaml"
        "mixformer_V2_k2.yaml"
        "mixformer_V2_k2M.yaml"
    )

    idx=0
    total=${#CONFIGS[@]}
    declare -A PIDS

    while [ $idx -lt $total ] || [ ${#PIDS[@]} -gt 0 ]; do
        # 启动新的任务（如果有可用的GPU）
        while [ ${#PIDS[@]} -lt $NGPUS ] && [ $idx -lt $total ]; do
            CONFIG=${CONFIGS[$idx]}
            DEVICE=${DEVICES[$((idx % NGPUS))]}
            echo "训练 $CONFIG 使用设备 $DEVICE..."
            CUDA_VISIBLE_DEVICES=$DEVICE python main.py --config ./config/$CONFIG --device 0 &
            PID=$!
            PIDS[$PID]=$DEVICE
            idx=$((idx+1))
        done

        # 等待任意一个进程结束
        if [ ${#PIDS[@]} -gt 0 ]; then
            wait -n
            # 检查并移除已完成的进程
            for PID in "${!PIDS[@]}"; do
                if ! kill -0 $PID 2>/dev/null; then
                    echo "进程 $PID 已完成"
                    unset PIDS[$PID]
                fi
            done
        fi
    done

    echo "Mix_Former 模型训练完成。"

    # 返回项目根目录
    cd "$PROJECT_DIR"
}

# ==============================================================================
# 函数：模型推理 Mix_GCN
# ==============================================================================
inference_mix_gcn() {
    echo "============================="
    echo "开始进行 Mix_GCN 模型推理"
    echo "============================="

    # 进入 Mix_GCN 目录
    cd "$MIX_GCN_DIR"

    # 定义模型列表
    MIX_GCN_MODELS=(
        "ctrgcn_V2_J.yaml:ctrgcn_V2_J.pt"
        "ctrgcn_V2_B.yaml:ctrgcn_V2_B.pt"
        "ctrgcn_V2_JM.yaml:ctrgcn_V2_JM.pt"
        "ctrgcn_V2_BM.yaml:ctrgcn_V2_BM.pt"
        "ctrgcn_V2_J_3d.yaml:ctrgcn_V2_J_3d.pt"
        "ctrgcn_V2_B_3d.yaml:ctrgcn_V2_B_3d.pt"
        "ctrgcn_V2_JM_3d.yaml:ctrgcn_V2_JM_3d.pt"
        "ctrgcn_V2_BM_3d.yaml:ctrgcn_V2_BM_3d.pt"
        "tdgcn_V2_J.yaml:tdgcn_V2_J.pt"
        "tdgcn_V2_B.yaml:tdgcn_V2_B.pt"
        "tdgcn_V2_JM.yaml:tdgcn_V2_JM.pt"
        "tdgcn_V2_BM.yaml:tdgcn_V2_BM.pt"
        "mstgcn_V2_J.yaml:mstgcn_V2_J.pt"
        "mstgcn_V2_B.yaml:mstgcn_V2_B.pt"
        "mstgcn_V2_JM.yaml:mstgcn_V2_JM.pt"
        "mstgcn_V2_BM.yaml:mstgcn_V2_BM.pt"
    )

    idx=0
    total=${#MIX_GCN_MODELS[@]}
    declare -A PIDS

    while [ $idx -lt $total ] || [ ${#PIDS[@]} -gt 0 ]; do
        # 启动新的任务（如果有可用的GPU）
        while [ ${#PIDS[@]} -lt $NGPUS ] && [ $idx -lt $total ]; do
            model=${MIX_GCN_MODELS[$idx]}
            CONFIG_FILE=$(echo $model | cut -d':' -f1)
            WEIGHTS_FILE=$(echo $model | cut -d':' -f2)
            DEVICE=${DEVICES[$((idx % NGPUS))]}
            echo "推理模型 $CONFIG_FILE 使用权重 $WEIGHTS_FILE 使用设备 $DEVICE..."
            CUDA_VISIBLE_DEVICES=$DEVICE python main.py --config ./config/$CONFIG_FILE --phase test --save-score True --weights ./checkpoints/$WEIGHTS_FILE --device 0 &
            PID=$!
            PIDS[$PID]=$DEVICE
            idx=$((idx+1))
        done

        # 等待任意一个进程结束
        if [ ${#PIDS[@]} -gt 0 ]; then
            wait -n
            # 检查并移除已完成的进程
            for PID in "${!PIDS[@]}"; do
                if ! kill -0 $PID 2>/dev/null; then
                    echo "进程 $PID 已完成"
                    unset PIDS[$PID]
                fi
            done
        fi
    done

    echo "Mix_GCN 模型推理完成。"

    # 返回项目根目录
    cd "$PROJECT_DIR"
}

# ==============================================================================
# 函数：模型推理 Mix_Former
# ==============================================================================
inference_mix_former() {
    echo "============================="
    echo "开始进行 Mix_Former 模型推理"
    echo "============================="

    # 进入 Mix_Former 目录
    cd "$MIX_FORMER_DIR"

    # 定义模型列表
    MIX_FORMER_MODELS=(
        "mixformer_V2_J.yaml:mixformer_V2_J.pt"
        "mixformer_V2_B.yaml:mixformer_V2_B.pt"
        "mixformer_V2_JM.yaml:mixformer_V2_JM.pt"
        "mixformer_V2_BM.yaml:mixformer_V2_BM.pt"
        "mixformer_V2_k2.yaml:mixformer_V2_k2.pt"
        "mixformer_V2_k2M.yaml:mixformer_V2_k2M.pt"
    )

    idx=0
    total=${#MIX_FORMER_MODELS[@]}
    declare -A PIDS

    while [ $idx -lt $total ] || [ ${#PIDS[@]} -gt 0 ]; do
        # 启动新的任务（如果有可用的GPU）
        while [ ${#PIDS[@]} -lt $NGPUS ] && [ $idx -lt $total ]; do
            model=${MIX_FORMER_MODELS[$idx]}
            CONFIG_FILE=$(echo $model | cut -d':' -f1)
            WEIGHTS_FILE=$(echo $model | cut -d':' -f2)
            DEVICE=${DEVICES[$((idx % NGPUS))]}
            echo "推理模型 $CONFIG_FILE 使用权重 $WEIGHTS_FILE 使用设备 $DEVICE..."
            CUDA_VISIBLE_DEVICES=$DEVICE python main.py --config ./config/$CONFIG_FILE --phase test --save-score True --weights ./checkpoints/$WEIGHTS_FILE --device 0 &
            PID=$!
            PIDS[$PID]=$DEVICE
            idx=$((idx+1))
        done

        # 等待任意一个进程结束
        if [ ${#PIDS[@]} -gt 0 ]; then
            wait -n
            # 检查并移除已完成的进程
            for PID in "${!PIDS[@]}"; do
                if ! kill -0 $PID 2>/dev/null; then
                    echo "进程 $PID 已完成"
                    unset PIDS[$PID]
                fi
            done
        fi
    done

    echo "Mix_Former 模型推理完成。"

    # 返回项目根目录
    cd "$PROJECT_DIR"
}

# ==============================================================================
# 函数：模型融合（集成）
# ==============================================================================
ensemble_models() {
    echo "============================="
    echo "开始进行模型融合（集成）"
    echo "============================="

    # 定义验证样本文件
    VAL_SAMPLE_V2="./Process_data/CS_test_V2.txt"

    echo "集成 Mix_GCN 模型的结果（CSv2）..."
    python Ensemble_MixGCN.py \
        --ctrgcn_J2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_J/epoch1_test_score.pkl \
        --ctrgcn_B2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_B/epoch1_test_score.pkl \
        --ctrgcn_JM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_JM/epoch1_test_score.pkl \
        --ctrgcn_BM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_BM/epoch1_test_score.pkl \
        --ctrgcn_J3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_J_3d/epoch1_test_score.pkl \
        --ctrgcn_B3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_B_3d/epoch1_test_score.pkl \
        --ctrgcn_JM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_JM_3d/epoch1_test_score.pkl \
        --ctrgcn_BM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_BM_3d/epoch1_test_score.pkl \
        --tdgcn_J2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_J/epoch1_test_score.pkl \
        --tdgcn_B2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_B/epoch1_test_score.pkl \
        --tdgcn_JM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_JM/epoch1_test_score.pkl \
        --tdgcn_BM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_BM/epoch1_test_score.pkl \
        --mstgcn_J2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_J/epoch1_test_score.pkl \
        --mstgcn_B2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_B/epoch1_test_score.pkl \
        --mstgcn_JM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_JM/epoch1_test_score.pkl \
        --mstgcn_BM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_BM/epoch1_test_score.pkl \
        --val_sample "$VAL_SAMPLE_V2" \
        --benchmark V2

    echo "集成 Mix_Former 模型的结果（CSv2）..."
    python Ensemble_MixFormer.py \
        --mixformer_J_Score ./Model_inference/Mix_Former/output/mixformer_V2_J/epoch1_test_score.pkl \
        --mixformer_B_Score ./Model_inference/Mix_Former/output/mixformer_V2_B/epoch1_test_score.pkl \
        --mixformer_JM_Score ./Model_inference/Mix_Former/output/mixformer_V2_JM/epoch1_test_score.pkl \
        --mixformer_BM_Score ./Model_inference/Mix_Former/output/mixformer_V2_BM/epoch1_test_score.pkl \
        --mixformer_k2_Score ./Model_inference/Mix_Former/output/mixformer_V2_k2/epoch1_test_score.pkl \
        --mixformer_k2M_Score ./Model_inference/Mix_Former/output/mixformer_V2_k2M/epoch1_test_score.pkl \
        --benchmark V2

    echo "模型融合（集成）完成。"
}

# ==============================================================================
# 主执行流程
# ==============================================================================
main() {
    echo "============================="
    echo "开始执行训练和融合脚本"
    echo "============================="

    # 训练 Mix_GCN 模型
    train_mix_gcn

    # 训练 Mix_Former 模型
    train_mix_former

    # 模型推理 Mix_GCN
    inference_mix_gcn

    # 模型推理 Mix_Former
    inference_mix_former

    # 模型融合（集成）
    ensemble_models

    echo "============================="
    echo "所有步骤完成"
    echo "============================="
}

# 执行主流程
main

# 输出结束信息
log "==============================="
log "训练过程于 $(date) 完成"
log "==============================="