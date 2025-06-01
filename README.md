## VQSVD – Variational Quantum Singular Value Decomposition Project
This repository contains the VQSVD (Variable Quantum Singular Value Decomposition) project. Follow the steps below to set up and run the project locally.

VQSVD (Variational Quantum Singular Value Decomposition) is a hybrid quantum-classical algorithm designed to approximate the singular value decomposition (SVD) of a matrix using quantum circuits.

It leverages a parameterized quantum circuit (ansatz) to represent the singular vectors and uses a classical optimizer to minimize a cost function related to the reconstruction of the matrix. VQSVD is inspired by variational algorithms like VQE and is well-suited for near-term quantum devices (NISQ).


## 🚀 Getting Started

These instructions will help you set up the project on your local machine for development and testing purposes.

## 📦 Prerequisites

Make sure you have the following installed on your system:

- [Python 3.8](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)

## ⚙️ Setup Instructions

```bash
# Clone the repository
git clone https://github.com/Universidad-Cenfotec/VQSVD-Project.git

# Navigate into the project directory
cd VQSVD-Project

# Create a virtual environment (Remember that the env should be created with Python3.8)
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

# Upgrade pip 
python -m pip install --upgrade pip

# Install the required packages
pip install paddlepaddle==2.3.0 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
pip install -r requirements.txt
```

## ✅ Ready to Go

Once all dependencies are installed, you are ready to run and develop with the VQSVD project.

## 🤝 Contributions
Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request. Ensure your code adheres to the existing style and includes appropriate tests.


## 📜 License
This project is licensed under the Apache License 2.0. You may freely use, modify, and distribute this software under the terms of the Apache 2.0 License. See the ./LICENSE file for full details.


