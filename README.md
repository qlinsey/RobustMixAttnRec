# RobustMixAttnRec

Sequential Recommendation via Adaptive Robust Attention with Multi-dimensional Embeddings

Sequential recommendation models have achieved state-of-the-art performance using self-attention mechanism. It has since been found that moving beyond only using item ID and positional embeddings leads to a significant accuracy boost when predicting the next item. In recent literature, it was reported that a multi-dimensional kernel embedding with temporal contextual kernels to capture users’ diverse behavioural patterns results in a substantial performance improvement. In this study, we further improve the sequential recommender model's robustness and generalization by introducing a mix-attention mechanism with  a layer-wise noise injection (LNI) regularization. We refer to our proposed model as adaptive robust sequential recommendation framework (ADRRec), and demonstrate through extensive experiments that our model outperforms existing self-attention architectures.

# Dataset:
  beauty
  game
  ml-1m
  ml-20m
  
# Set Up:
pip install -r requirements.txt

# Run:
python run.py --templates train_mixtime
python run.py --templates train_bert
OR python run.py --templates train_meantime --dataset_code game --hidden_units 128

# Code Base: 
https://github.com/SungMinCho/MEANTIME/ 
