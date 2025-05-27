# TCN Training Makefile with Autoregressive Support
.PHONY: help setup install login clean train sweep train-all sweep-all add-ride list-rides
.PHONY: train-ar predict train-ar-all

setup:
	@echo "🚀 Setting up TCN training environment..."
	python setup.py

install:
	pip install -r requirements.txt

login:
	wandb login

# Original TCN model training
train:
ifndef RIDE
	$(error RIDE is not set. Use: make train RIDE=poseidon)
endif
	@echo "🚂 Training original TCN model for ride: $(RIDE)"
	python -m TCN.train --ride $(RIDE) --epochs 50

sweep:
ifndef RIDE
	$(error RIDE is not set. Use: make sweep RIDE=poseidon)
endif
	@echo "🔍 Running hyperparameter sweep for ride: $(RIDE)"
	python -m TCN.sweep --ride $(RIDE) --config standard --count $(or $(COUNT),50)

# Multi-day prediction
predict:
ifndef RIDE
	$(error RIDE is not set. Use: make predict RIDE=poseidon DATA=path/to/data.parquet DAYS=3)
endif
ifndef DATA
	$(error DATA is not set. Use: make predict RIDE=poseidon DATA=path/to/data.parquet DAYS=3)
endif
	@echo "🔮 Predicting $(or $(DAYS),1) days for ride: $(RIDE)"
	python autoregressive_inference.py --ride $(RIDE) --data-path $(DATA) --num-days $(or $(DAYS),1) $(if $(PLOT),--plot) $(if $(SAVE),--save-path $(SAVE))

# Example prediction with plots
predict-example:
ifndef RIDE
	$(error RIDE is not set. Use: make predict-example RIDE=poseidon)
endif
	@echo "🔮 Example prediction with plots for ride: $(RIDE)"
	@python -c "import yaml; rides = yaml.safe_load(open('configs/rides_config.yaml'))['rides']; data_path = rides['$(RIDE)']['data_path']; print(f'Using data: {data_path}')"
	@python -c "import yaml; rides = yaml.safe_load(open('configs/rides_config.yaml'))['rides']; data_path = rides['$(RIDE)']['data_path']; __import__('subprocess').run(['python', 'autoregressive_inference.py', '--ride', '$(RIDE)', '--data-path', data_path, '--num-days', '3', '--plot', '--save-path', 'predictions/$(RIDE)_3day_prediction.csv'])"

# Data management
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
		config = yaml.safe_load(open('configs/rides_config.yaml')) if os.path.exists('configs/rides_config.yaml') else {'rides': {}, 'global_settings': {'splits_output_dir': '../data/processed/splits', 'base_data_dir': '../data/processed/ep/rides', 'model_output_dir': './models'}, 'default_params': {'epochs': 100, 'patience': 10, 'seed': 42, 'use_wandb': True}}; \
		config['rides']['$(RIDE)'] = {'data_path': '$(DATAPATH)', 'display_name': '$(RIDE)', 'description': 'Auto-generated entry for $(RIDE)'}; \
		yaml.dump(config, open('configs/rides_config.yaml', 'w'), default_flow_style=False, indent=2)"
	@echo "✅ Added $(RIDE) to configs/rides_config.yaml"

list-rides:
	@echo "📋 Available rides:"
	@python -c "import yaml; rides = yaml.safe_load(open('configs/rides_config.yaml', 'r')).get('rides', {}); [print(f'  - {name}: {info.get(\"display_name\", name)}') for name, info in rides.items()] if rides else print('  No rides configured.')" 2>/dev/null || echo "  No rides configured. Run 'make setup' first."
