# ai-vuln-prioritization
CSC 761 Project

This repository contains the implementation of an AI framework for prioritizing software vulnerabilities using graph neural networks and explainable AI, as described in the final project report.

Environment Specs

Python: 3.12
OS: Tested on Ubuntu 22.04 and Windows 11
Hardware: GPU recommended (NVIDIA with CUDA 12.1) for GNN training; fallback to CPU.

Installation Instructions

1.  Clone the repository:textgit clone https://github.com/awarneritm/ai-vuln-prioritization.git
cd ai-vuln-prioritization

2.  Create a virtual environment (Python 3.12 recommended):textpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3.  Install dependencies:textpip install -r requirements.txt

*How to Run the Project*

Prepare Data:
Download full CVEfixes from https://www.kaggle.com/datasets/girish17019/cvefixes-vulnerable-and-fixed-code.
Run preprocessing: python src/data_prep.py --data_path data/raw/cvefixes --output_path data/processed

Train Model:
python src/train.py --data_dir data/processed --epochs 50 --lr 0.001 --output_model models/gnn_model.pth

Run Inference:
python src/infer.py --model_path models/gnn_model.pth --input_cve example_cve.json --output results/priorities.json

Generate Explanations:
python src/explain.py --model_path models/gnn_model.pth --input results/priorities.json --output results/justifications/

Evaluate:
Results will be saved in results/metrics.csv and graphs/.


For questions, contact aaron.warner@trojans.dsu.edu.

ai-vuln-prioritization/
├── src/
├── data/
├── results/
│   ├── metrics.csv
│   ├── graphs/          (5 PNGs)
│   └── justifications/  (5 .txt files)
├── models/                  ← NOW COMPLETE
│   ├── gnn_exploit_predictor.pth
│   ├── xgboost_context_prioritizer.json
│   ├── codebert_tokenizer/
│   ├── model_config.yaml
│   ├── training_history.pt
│   └── README_models.md
└── requirements.txt
