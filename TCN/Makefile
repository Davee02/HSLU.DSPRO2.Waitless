# TCN Training Makefile with Autoregressive Support
.PHONY: help setup install login clean train sweep train-all sweep-all add-ride list-rides
.PHONY: train-ar predict train-ar-all

help:
	@echo "TCN Training Commands:"
	@echo "  Basic Setup:"
	@echo "    make setup          - Initial setup"
	@echo "    make login          - Login to W&B"
	@echo "    make install        - Install dependencies"
	@echo ""
	@echo "  Original TCN Model:"
	@echo "    make train RIDE=X   - Train original TCN model"
	@echo "    make sweep RIDE=X   - Run hyperparameter sweep"
	@echo "    make train-all      - Train all rides (original)"
	@echo "    make sweep-all      - Run sweeps for all rides"
	@echo ""
	@echo "  Autoregressive TCN Model:"
	@echo "    make train-ar RIDE=X     - Train autoregressive model"
	@echo "    make train-ar-all        - Train autoregressive for all rides"
	@echo "    make predict RIDE=X DAYS=N DATA=path  - Multi-day prediction"
	@echo ""
	@echo "  Data Management:"
	@echo "    make add-ride RIDE=X DATAPATH=Y - Add new ride"
	@echo "    make list-rides     - Show configured rides"
	@echo "    make clean          - Clean up files"

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
	python train.py --ride $(RIDE)

sweep:
ifndef RIDE
	$(error RIDE is not set. Use: make sweep RIDE=poseidon)
endif
	@echo "🔍 Running hyperparameter sweep for ride: $(RIDE)"
	python run_sweep.py --ride $(RIDE) --count $(or $(COUNT),50)

train-all:
	@echo "🚂 Training original TCN models for all configured rides..."
	@python -c "import yaml; rides = yaml.safe_load(open('configs/rides_config.yaml'))['rides']; [__import__('subprocess').run(['python', 'train.py', '--ride', ride]) for ride in rides.keys()]"

sweep-all:
	@echo "🔍 Running sweeps for all configured rides..."
	@python -c "import yaml; rides = yaml.safe_load(open('configs/rides_config.yaml'))['rides']; [__import__('subprocess').run(['python', 'run_sweep.py', '--ride', ride, '--count', '$(or $(COUNT),30)']) for ride in rides.keys()]"

# Autoregressive TCN model training
train-ar:
ifndef RIDE
	$(error RIDE is not set. Use: make train-ar RIDE=poseidon)
endif
	@echo "🤖 Training autoregressive TCN model for ride: $(RIDE)"
	python train_autoregressive.py --ride $(RIDE) --wandb-project waitless-autoregressive-tcn-hslu-dspro2-fs25

train-ar-all:
	@echo "🤖 Training autoregressive TCN models for all configured rides..."
	@python -c "import yaml; rides = yaml.safe_load(open('configs/rides_config.yaml'))['rides']; [__import__('subprocess').run(['python', 'train_autoregressive.py', '--ride', ride, '--wandb-project', 'waitless-autoregressive-tcn-hslu-dspro2-fs25']) for ride in rides.keys()]"

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

# Training status and monitoring
status:
	@echo "📊 Training Status:"
	@echo "   Original Models:"
	@ls -la models/*.pt models/*.pkl 2>/dev/null | wc -l | xargs -I {} echo "     {} models found"
	@echo "   Autoregressive Models:"
	@ls -la models/autoregressive/*.pt 2>/dev/null | wc -l | xargs -I {} echo "     {} models found"
	@echo "   Predictions:"
	@ls -la predictions/*.csv 2>/dev/null | wc -l | xargs -I {} echo "     {} prediction files found"

# Create prediction directories
create-dirs:
	@mkdir -p models/autoregressive predictions logs

# Cleanup
clean:
	@echo "🧹 Cleaning up..."
	rm -rf models/*.pt models/*.pkl models/autoregressive/*.pt models/autoregressive/*.pkl
	rm -rf logs/*.log wandb/ predictions/*.csv
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

clean-predictions:
	@echo "🧹 Cleaning predictions..."
	rm -rf predictions/*.csv

# Development helpers
test-setup:
	@echo "🧪 Testing setup..."
	python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
	python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
	python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
	python -c "from pytorch_tcn import TCN; print('PyTorch TCN: OK')"
	python -c "import wandb; print(f'W&B: {wandb.__version__}')"

# Example workflows
workflow-single:
ifndef RIDE
	$(error RIDE is not set. Use: make workflow-single RIDE=poseidon)
endif
	@echo "🔄 Running complete single-ride workflow for: $(RIDE)"
	@echo "1. Training original model..."
	@make train RIDE=$(RIDE)
	@echo "2. Training autoregressive model..."
	@make train-ar RIDE=$(RIDE)
	@echo "3. Creating example prediction..."
	@make predict-example RIDE=$(RIDE)
	@echo "✅ Workflow complete for $(RIDE)"

predict-day:
ifndef RIDE
	$(error RIDE is not set. Use: make predict-day RIDE=poseidon DATA=path/to/data.parquet)
endif
ifndef DATA
	$(error DATA is not set. Use: make predict-day RIDE=poseidon DATA=path/to/data.parquet)
endif
	@echo "📈 Creating daily prediction visualization for: $(RIDE)"
	python daily_prediction_viz.py --ride $(RIDE) --data-path $(DATA) $(if $(DATE),--date $(DATE)) $(if $(SAVE),--save-path $(SAVE)) --model-type $(or $(MODEL),auto)

# Predict today for a ride (uses ride's default data path)
predict-today:
ifndef RIDE
	$(error RIDE is not set. Use: make predict-today RIDE=poseidon)
endif
	@echo "📈 Creating today's prediction for: $(RIDE)"
	@python -c "import yaml; rides = yaml.safe_load(open('configs/rides_config.yaml'))['rides']; data_path = rides['$(RIDE)']['data_path']; print(f'Using data: {data_path}')"
	@python -c "import yaml; rides = yaml.safe_load(open('configs/rides_config.yaml'))['rides']; data_path = rides['$(RIDE)']['data_path']; __import__('subprocess').run(['python', 'daily_prediction_viz.py', '--ride', '$(RIDE)', '--data-path', data_path, '--save-path', 'predictions/$(RIDE)_daily_prediction.png'])"

# Predict yesterday (useful for evaluation)
predict-yesterday:
ifndef RIDE
	$(error RIDE is not set. Use: make predict-yesterday RIDE=poseidon)
endif
	@echo "📈 Creating yesterday's prediction for: $(RIDE)"
	@python -c "import yaml, datetime; rides = yaml.safe_load(open('configs/rides_config.yaml'))['rides']; data_path = rides['$(RIDE)']['data_path']; yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d'); __import__('subprocess').run(['python', 'daily_prediction_viz.py', '--ride', '$(RIDE)', '--data-path', data_path, '--date', yesterday, '--save-path', f'predictions/$(RIDE)_yesterday_prediction.png'])"


# Documentation
docs:
	@echo "📚 TCN Training Documentation"
	@echo ""
	@echo "TRAINING WORKFLOW:"
	@echo "1. Setup:           make setup"
	@echo "2. Login to W&B:    make login"
	@echo "3. Add rides:       make add-ride RIDE=name DATAPATH=path"
	@echo "4. Train original:  make train RIDE=name"
	@echo "5. Train AR:        make train-ar RIDE=name"
	@echo "6. Predict:         make predict RIDE=name DATA=data.parquet DAYS=3"
	@echo ""
	@echo "QUICK START:"
	@echo "  make workflow-single RIDE=poseidon"
	@echo ""
	@echo "BATCH OPERATIONS:"
	@echo "  make train-all"
	@echo "  make train-ar-all"