conda run -n cv pip install dvc black flake8 isort rich
conda run -n cv python -m ipykernel install --user --name cv --display-name "CV (Python 3.11)"
conda run -n cv dvc init
New-Item -ItemType Directory -Force -Path .vscode
Set-Content -Path .vscode\settings.json -Value '{
  "python.defaultInterpreterPath": "~\.conda\envs\cv\python.exe",
  "editor.formatOnSave": true,
  "editor.rulers": [88],
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  },
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.pylintEnabled": false,
  "jupyter.notebookFileRoot": "${workspaceFolder}",
  "files.autoSave": "afterDelay",
  "files.autoSaveDelay": 1000,
  "editor.fontSize": 14,
  "editor.lineHeight": 1.6,
  "editor.minimap.enabled": false,
  "terminal.integrated.defaultProfile.linux": "bash",
  "git.autofetch": true,
  "editor.bracketPairColorization.enabled": true,
  "editor.guides.bracketPairs": true
}'
$dirs = @("phase0_math", "phase1_classical", "phase2_deeplearning", "phase3_data", "phase4_advanced", "phase5_production", "notebooks", "experiments", "data", "models", "utils", "configs")
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path $d
}

New-Item -ItemType File -Force -Path "utils\__init__.py"
New-Item -ItemType File -Force -Path "utils\metrics.py"
New-Item -ItemType File -Force -Path "utils\visualize.py"
New-Item -ItemType File -Force -Path "utils\data_utils.py"
New-Item -ItemType File -Force -Path "utils\training.py"
New-Item -ItemType File -Force -Path "configs\train.yaml"

Add-Content -Path .gitignore -Value "
# Large files
data/
models/
*.pth
*.pt
*.onnx
*.h5
*.pkl

# Jupyter
.ipynb_checkpoints/

# Python
__pycache__/
*.pyc
*.pyo

# Secrets
.env
*.env
secrets.yaml
wandb/

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log
runs/
"

git add .
git commit -m "chore: initial project structure and setup"
