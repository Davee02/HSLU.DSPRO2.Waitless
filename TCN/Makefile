# TCN Training Makefile
.PHONY: help setup install login clean train sweep train-all sweep-all add-ride list-rides

help:
	@echo "TCN Training Commands:"
	@echo "  make setup          - Initial setup"
	@echo "  make login          - Login to W&B"
	@echo "  make train RIDE=X   - Train single model"
	@echo "  make sweep RIDE=X   - Run hyperparameter sweep"
	@echo "  make train-all      - Train all rides"
	@echo "  make sweep-all      - Run sweeps for all rides"
	@echo "  make add-ride RIDE=X DATAPATH=Y - Add new ride"
	@echo "  make list-rides     - Show configured rides"
	@echo "  make clean          - Clean up files"

setup:
	@echo "🚀 Setting up TCN training environment..."
	python setup.py

install:
	pip install -r requirements.txt

login:
	wandb login

train:
ifndef RIDE
	$(error RIDE is not set. Use: make train RIDE=poseidon)
endif
	@echo "🚂 Training model for ride: $(RIDE)"
	python train.py --ride $(RIDE)

sweep:
ifndef RIDE
	$(error RIDE is not set. Use: make sweep RIDE=poseidon)
endif
	@echo "🔍 Running hyperparameter sweep for ride: $(RIDE)"
	python run_sweep.py --ride $(RIDE) --count $(or $(COUNT),50)

train-all:
	@echo "🚂 Training models for all configured rides..."
	@python -c "import yaml; rides = yaml.safe_load(open('configs/rides_config.yaml'))['rides']; [__import__('subprocess').run(['python', 'train.py', '--ride', ride]) for ride in rides.keys()]"

sweep-all:
	@echo "🔍 Running sweeps for all configured rides..."
	@python -c "import yaml; rides = yaml.safe_load(open('configs/rides_config.yaml'))['rides']; [__import__('subprocess').run(['python', 'run_sweep.py', '--ride', ride, '--count', '$(or $(COUNT),30)']) for ride in rides.keys()]"

add-ride:
ifndef RIDE
	$(error RIDE is not set. Use: make add-ride RIDE=newride DATAPATH=/path/to/data.parquet)
endif
ifndef DATAPATH
	$(error DATAPATH is not set. Use: make add-ride RIDE=newride DATAPATH=/path/to/data.parquet)
endif
	@echo "➕ Adding new ride: $(RIDE)"
	@mkdir -p configs
	@python -c "import yaml, os; \
		config = yaml.safe_load(open('configs/rides_config.yaml')) if os.path.exists('configs/rides_config.yaml') else {}; \
		config.setdefault('rides', {})['$(RIDE)'] = {'data_path': '$(DATAPATH)', 'display_name': '$(RIDE)', 'description': 'Auto-generated entry for $(RIDE)'}; \
		yaml.dump(config, open('configs/rides_config.yaml', 'w'), default_flow_style=False, indent=2)"
	@echo "✅ Added $(RIDE) to configs/rides_config.yaml"

list-rides:
	@echo "📋 Available rides:"
	@python -c "import yaml; rides = yaml.safe_load(open('configs/rides_config.yaml', 'r')).get('rides', {}); [print(f'  - {name}: {info.get(\"display_name\", name)}') for name, info in rides.items()] if rides else print('  No rides configured.')" 2>/dev/null || echo "  No rides configured. Run 'make setup' first."

clean:
	@echo "🧹 Cleaning up..."
	rm -rf models/*.pt models/*.pkl logs/*.log wandb/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete