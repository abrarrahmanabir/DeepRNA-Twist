## DeepRNA-Twist: RNA Torsion Angle Prediction

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/abrarrahmanabir/DeepRNA-Twist.git
   cd DeepRNA-Twist


2. **Install dependencies**:

   ```bash
   pip install numpy tensorflow torch scikit-learn pandas transformers keras tqdm


   Data
Place the following .npy files in the data/ directory:

features.npy: Input features for RNA sequences.
sine_cosine_values.npy: Ground truth torsion angles (as sine and cosine values).
attention_masks.npy: Attention masks indicating valid positions in the sequence.
pos_ids.npy: Positional IDs for the sequences.
How to Run
Train the Model: To start the training process, execute the following command:

bash
Copy code
python main.py
Inference: To make predictions with the trained model, modify the main.py script for inference mode and run:

bash
Copy code
python main.py --mode inference --test_data /path/to/test_data.npy
Evaluation
After training, the script will compute the Periodic Mean Absolute Error (MAE) for overall torsion angles and individual angles. The results will be printed in the terminal.
