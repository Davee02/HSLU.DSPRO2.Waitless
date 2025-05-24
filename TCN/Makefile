
.PHONY: help setup install login clean train sweep train-all sweep-all


help:
	@echo "TCN Training Commands:"
	@echo "  make setup          - Setup environment and install dependencies"
	@echo "  make install        - Install Python dependencies only"
	@echo "  make login          - Login to Weights & Biases"
	@echo "  make train RIDE=X   - Train single model for ride X"
	@echo "  make sweep RIDE=X   - Run hyperparameter sweep for ride X"
	@echo "  make train-all      - Train models for all rides"
	@echo "  make sweep-all      - Run sweeps for all rides"
	@echo "  make clean          - Clean up generated files"
	@echo ""
	@echo "Examples:"
	@echo "  make train RIDE=poseidon"
	@echo "  make sweep RIDE=poseidon COUNT=50"


setup:
	@echo "Setting up TCN training environment..."
	chmod +x run_experiments.sh
	./run_experiments.sh setup

install:
	pip install -r requirements.txt

login:
	wandb login

train:
ifndef RIDE
	$(error RIDE is not set. Use: make train RIDE=poseidon)
endif
	python train.py --ride $(RIDE)


sweep:
ifndef RIDE
	$(error RIDE is not set. Use: make sweep RIDE=poseidon)
endif
	python run_sweep.py --ride $(RIDE) --count $(or $(COUNT),50)

train-all:
	./run_experiments.sh train-all

sweep-all:
	./run_experiments.sh sweep-all $(or $(COUNT),30)

clean:
	rm -rf models/*.pt models/*.pkl
	rm -rf logs/*.log
	rm -rf wandb/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

clean-configs:
	@read -p "Are you sure you want to delete all config files? [y/N] " confirm && [ "$confirm" = "y" ]
	rm -rf configs/*.yaml
	@echo "Config files deleted. Run 'make setup' to recreate defaults."

add-ride:
ifndef RIDE
	$(error RIDE is not set. Use: make add-ride RIDE=newride PATH=/path/to/data.parquet)
endif
ifndef PATH
	$(error PATH is not set. Use: make add-ride RIDE=newride PATH=/path/to/data.parquet)
endif
	@echo "Adding new ride: $(RIDE)"
	@echo "  $(RIDE):" >> configs/rides_config.yaml
	@echo "    data_path: \"$(PATH)\"" >> configs/rides_config.yaml
	@echo "    display_name: \"$(RIDE)\"" >> configs/rides_config.yaml
	@echo "    description: \"Auto-generated entry for $(RIDE)\"" >> configs/rides_config.yaml
	@echo "Added $(RIDE) to configs/rides_config.yaml"

list-rides:
	@echo "Available rides:"
	@grep -A 3 "^  [a-zA-Z]" configs/rides_config.yaml | grep -E "^  [a-zA-Z]|display_name" | sed 's/^  //' || echo "No rides configured. Run 'make setup' first."