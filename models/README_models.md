# Models Folder – AI-Enhanced Vulnerability Prioritization
Trained on CVEfixes dataset (2025)

### File Descriptions

- `gnn_exploit_predictor.pth`  
  Heterogeneous GraphSAGE model (2 layers, hidden dim=256)  
  Predicts exploitation probability (0–1) from CVE → Repo → Vendor graph  
  Trained for 47 epochs, best val AUC = 0.920

- `xgboost_context_prioritizer.json`  
  Final prioritization layer  
  Input: [gnn_exploit_prob, asset_criticality, network_exposure]  
  Output: Priority score 0–10

- `codebert_tokenizer/`  
  microsoft/codebert-base tokenizer used for code embeddings  
  Required for inference on new code diffs

- `model_config.yaml`  
  Full reproducible config (layers, learning rates, seeds, etc.)

- `training_history.pt`  
  Torch tensor containing epoch-wise train/val loss + AUC  
  Can be loaded and plotted

All models tested on Python 3.10 + PyTorch 2.0 + PyTorch-Geometric 2.3
