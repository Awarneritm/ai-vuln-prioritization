# ai-vuln-prioritization
CSC 761 Project

This repository contains the implementation of an AI framework for prioritizing software vulnerabilities using graph neural networks and explainable AI, as described in the final project report.

Environment Specs

Python: 3.12
OS: Tested on Ubuntu 22.04 and Windows 11
Hardware: GPU recommended (NVIDIA with CUDA 12.1) for GNN training; fallback to CPU.

Installation Instructions

1.  Clone the repository:textgit clone https://github.com/aaronwarner/ai-vuln-prioritization.git
cd ai-vuln-prioritization

2.  Create a virtual environment (Python 3.12 recommended):textpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3.  Install dependencies:textpip install -r requirements.txt
