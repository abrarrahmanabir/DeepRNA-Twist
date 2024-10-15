## DeepRNA-Twist: RNA Torsion Angle Prediction
We introduce DeepRNA-Twist, a novel deep learning framework designed to predict RNA torsion and pseudo-torsion angles directly from sequence. DeepRNA-Twist utilizes RNA language model embeddings, which provides rich, context-aware feature representations of RNA sequences. Additionally, it introduces 2A3IDC module, combining inception networks with dilated convolutions and multi-head attention mechanism. The dilated convolutions capture long-range dependencies in the sequence without requiring a large number of parameters, while the multi-head attention mechanism enhances the model’s ability to focus on both local and global structural features simultaneously.
### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/abrarrahmanabir/DeepRNA-Twist.git
   cd DeepRNA-Twist


2. **Install dependencies**:

   ```bash
   pip install numpy tensorflow torch scikit-learn pandas transformers keras tqdm


### How to Run
1. **Train the Model**:
To start the training process, execute the following command:

   ```bash
   python main.py





