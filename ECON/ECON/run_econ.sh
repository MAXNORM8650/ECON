#!/bin/bash

# ECON Multi-Agent LLM Framework Training Script
# Simple and effective training launcher

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 默认参数
CONFIG="src/config/config.yaml"
API_KEY=""
N_AGENTS=3
ENV="huggingface_dataset_env"
SEED=42
COORDINATOR_MODEL=""
EXECUTOR_MODEL=""
EXPERIMENT_NAME=""

# 打印函数
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 显示帮助
show_help() {
    cat << EOF
ECON Framework Training Script

Usage: $0 [OPTIONS]

Options:
    -k, --api-key KEY           Together AI API key (required)
    -c, --config FILE           Configuration file (default: src/config/config.yaml)
    -a, --agents N              Number of agents (default: 3)
    -e, --env ENV               Environment name (default: huggingface_dataset_env)
    -s, --seed N                Random seed (default: 42)
    --coordinator-model MODEL   Coordinator LLM model name
    --executor-model MODEL      Executor LLM model name
    --experiment-name NAME      Experiment name for logging
    --wandb                     Enable wandb logging
    --wandb-project PROJECT     wandb project name
    --wandb-entity ENTITY       wandb entity name
    --wandb-tags TAGS           wandb tags (comma-separated)
    -h, --help                  Show this help

Examples:
    # Basic usage
    $0 --api-key your_api_key_here

    # Custom configuration with 5 agents
    $0 --api-key your_key --agents 5 --config examples/configs/large_scale.yaml

    # With specific models and wandb tracking
    $0 --api-key your_key \\
       --coordinator-model meta-llama/Llama-3.3-70B-Instruct-Turbo \\
       --executor-model meta-llama/Llama-3.3-70B-Instruct-Turbo \\
       --wandb --wandb-project "ECON-Math-Reasoning"

    # Fast training configuration
    $0 --api-key your_key --config examples/configs/fast_training.yaml

EOF
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -k|--api-key)
            API_KEY="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -a|--agents)
            N_AGENTS="$2"
            shift 2
            ;;
        -e|--env)
            ENV="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        --coordinator-model)
            COORDINATOR_MODEL="$2"
            shift 2
            ;;
        --executor-model)
            EXECUTOR_MODEL="$2"
            shift 2
            ;;
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --wandb)
            USE_WANDB="--use_wandb"
            shift
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb-entity)
            WANDB_ENTITY="$2"
            shift 2
            ;;
        --wandb-tags)
            WANDB_TAGS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查必需参数
if [[ -z "$API_KEY" ]]; then
    print_error "API key is required. Use --api-key to specify."
    show_help
    exit 1
fi

# 检查配置文件
if [[ ! -f "$CONFIG" ]]; then
    print_error "Configuration file not found: $CONFIG"
    exit 1
fi

# 设置环境
print_info "Starting ECON Framework Training"
print_info "===================================="
print_info "Config: $CONFIG"
print_info "API Key: ${API_KEY:0:10}..."
print_info "Agents: $N_AGENTS"
print_info "Environment: $ENV"
print_info "Seed: $SEED"

if [[ -n "$COORDINATOR_MODEL" ]]; then
    print_info "Coordinator Model: $COORDINATOR_MODEL"
fi

if [[ -n "$EXECUTOR_MODEL" ]]; then
    print_info "Executor Model: $EXECUTOR_MODEL"
fi

if [[ -n "$EXPERIMENT_NAME" ]]; then
    print_info "Experiment Name: $EXPERIMENT_NAME"
fi

# 设置Python路径和API密钥
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
export TOGETHER_API_KEY="$API_KEY"

# 构建训练命令
CMD="python3 src/train.py"
CMD="$CMD --config $CONFIG"
CMD="$CMD --api_key $API_KEY"
CMD="$CMD --n_agents $N_AGENTS"
CMD="$CMD --env $ENV"
CMD="$CMD --seed $SEED"

if [[ -n "$COORDINATOR_MODEL" ]]; then
    CMD="$CMD --coordinator_model $COORDINATOR_MODEL"
fi

if [[ -n "$EXECUTOR_MODEL" ]]; then
    CMD="$CMD --executor_model $EXECUTOR_MODEL"
fi

if [[ -n "$EXPERIMENT_NAME" ]]; then
    CMD="$CMD --experiment_name $EXPERIMENT_NAME"
fi

if [[ -n "$USE_WANDB" ]]; then
    CMD="$CMD $USE_WANDB"
fi

if [[ -n "$WANDB_PROJECT" ]]; then
    CMD="$CMD --wandb_project $WANDB_PROJECT"
fi

if [[ -n "$WANDB_ENTITY" ]]; then
    CMD="$CMD --wandb_entity $WANDB_ENTITY"
fi

if [[ -n "$WANDB_TAGS" ]]; then
    CMD="$CMD --wandb_tags $WANDB_TAGS"
fi

print_info "Executing: $CMD"
print_info "===================================="

# 执行训练
exec $CMD 