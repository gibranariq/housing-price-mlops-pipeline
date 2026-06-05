#!/bin/bash

# Color codes for clean output display
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check docker prerequisites
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed or not in PATH.${NC}"
        exit 1
    fi
}

start_services() {
    echo -e "${BLUE}=== Starting MLOps Services (FastAPI + Streamlit + MLflow) ===${NC}"
    docker compose --profile mlflow-ui up -d --build
    echo -e "${GREEN}Services started successfully!${NC}"
    echo -e "Access points:"
    echo -e " - ${CYAN}Streamlit UI (Frontend):${NC} http://localhost:8501"
    echo -e " - ${CYAN}MLflow Tracking UI:${NC} http://localhost:5001"
    echo -e " - ${CYAN}FastAPI serving API Docs:${NC} http://localhost:8000/docs"
}

stop_services() {
    echo -e "${YELLOW}=== Stopping MLOps Services ===${NC}"
    docker compose --profile mlflow-ui down
    echo -e "${GREEN}Services stopped successfully.${NC}"
}

train_model() {
    echo -e "${BLUE}=== Running Model Training (Ensemble Lasso, Ridge, CatBoost, XGBoost) ===${NC}"
    docker compose --profile train run trainer
    echo -e "${GREEN}Model training pipeline run completed successfully.${NC}"
}

run_monitoring() {
    echo -e "${BLUE}=== Running Data Drift & Concept Drift Monitoring ===${NC}"
    docker compose --profile monitor run monitor
    echo -e "${GREEN}Monitoring analysis finished.${NC}"
}

show_status() {
    echo -e "${BLUE}=== Checking Active Container Status ===${NC}"
    docker ps --filter name=ames
}

show_logs() {
    echo -e "${BLUE}=== Showing Service Logs (Press Ctrl+C to exit) ===${NC}"
    docker compose logs -f
}

menu() {
    while true; do
        echo -e "\n${CYAN}==================================================${NC}"
        echo -e "${YELLOW}        Ames Housing MLOps Control Center         ${NC}"
        echo -e "${CYAN}==================================================${NC}"
        echo -e "1) Start MLOps Services (Frontend, Backend, MLflow)"
        echo -e "2) Stop MLOps Services"
        echo -e "3) Run Model Training & Registry"
        echo -e "4) Run Data Drift & Monitoring Checks"
        echo -e "5) Show Container Status"
        echo -e "6) View Live Services Logs"
        echo -e "7) Exit"
        echo -e "${CYAN}--------------------------------------------------${NC}"
        read -p "Select option [1-7]: " opt
        case $opt in
            1) start_services ;;
            2) stop_services ;;
            3) train_model ;;
            4) run_monitoring ;;
            5) show_status ;;
            6) show_logs ;;
            7) echo -e "${GREEN}Goodbye!${NC}"; exit 0 ;;
            *) echo -e "${RED}Invalid option. Please choose between 1 and 7.${NC}" ;;
        esac
    done
}

# Main Execution
check_docker

if [ -z "$1" ]; then
    menu
else
    case "$1" in
        start) start_services ;;
        stop) stop_services ;;
        train) train_model ;;
        monitor) run_monitoring ;;
        status) show_status ;;
        logs) show_logs ;;
        *)
            echo -e "${RED}Usage: $0 {start|stop|train|monitor|status|logs}${NC}"
            echo -e "Or run without arguments for interactive menu."
            exit 1
            ;;
    esac
fi
