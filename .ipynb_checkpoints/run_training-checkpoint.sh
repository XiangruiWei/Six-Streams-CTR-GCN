#!/bin/bash

# 脚本名称: run_training.sh
# 功能: 自动化执行训练步骤并保存终端输出到 output.txt

# 定义日志文件
LOG_FILE="output.txt"

# 清空日志文件（如果不希望覆盖，可以注释掉下一行）
> "$LOG_FILE"

# 函数: 输出并记录日志
log() {
    echo "$1" | tee -a "$LOG_FILE"
}

# 输出开始信息
log "==============================="
log "训练过程开始于 $(date)"
log "==============================="

# 6.1. 训练 Joint 流
log "6.1. 开始训练 Joint 流..."
sh train_uav_joint.sh 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "错误: 训练 Joint 流失败。"
    exit 1
fi
log "6.1. 训练 Joint 流完成。"

# 6.2. 训练 Bone 流
log "6.2. 开始训练 Bone 流..."
sh train_uav_bone.sh 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "错误: 训练 Bone 流失败。"
    exit 1
fi
log "6.2. 训练 Bone 流完成。"

# 6.3. 训练 Motion 流
log "6.3. 开始训练 Motion 流..."
sh train_uav_motion.sh 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "错误: 训练 Motion 流失败。"
    exit 1
fi
log "6.3. 训练 Motion 流完成。"

# 6.4. 使用长尾损失进行训练
log "6.4. 开始使用长尾损失进行训练..."
sh train_uav_longtail.sh 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "错误: 使用长尾损失进行训练失败。"
    exit 1
fi
log "6.4. 使用长尾损失进行训练完成。"

# 6.5. 使用长尾损失进行训练
log "6.5. 开始使用时间进行训练..."
sh train_uav_tta.sh 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "错误: 使用长尾损失进行训练失败。"
    exit 1
fi
log "6.5. ...。"

# 7.2. 使用 v2 权重进行集成
log "7.1. 开始使用 v1 权重进行集成..."
sh ensemble_v1.sh 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "错误: 使用 v1 权重进行集成失败。"
    exit 1
fi
log "7.1. 使用 v1 权重进行集成完成。"


# 7.2. 使用 v2 权重进行集成
log "7.2. 开始使用 v2 权重进行集成..."
sh ensemble_v2.sh 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "错误: 使用 v2 权重进行集成失败。"
    exit 1
fi
log "7.2. 使用 v2 权重进行集成完成。"

# 输出结束信息
log "==============================="
log "训练过程于 $(date) 完成"
log "==============================="