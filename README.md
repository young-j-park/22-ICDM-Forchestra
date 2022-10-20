# Forchestra

## TS2Vec
We used the TS2Vec code from: https://github.com/yuezhihan/ts2vec. This repo doesn't have a copyright for it.

## How to Run (Examples)
### How to train a model without initializing
python main.py --model_save_path ./params/forchestra.pt --output_fname_prefix ./results/prediction

### How to train a model with initializing
python main.py --repr_model_path ./params/ts2vec.pt --base_model_path ./params/base.pt --model_save_path ./params/forchestra.pt --output_fname_prefix ./results/prediction

### How to get prediction using a trained model
python main.py --model_load_path ./params/forchestra.pt --output_fname_prefix ./results/prediction --skip_train

## Techinical Notes
- We used different scaling method for representation and prediction module. Selecting a proper scaling method for each module is important.
- CNN-based model works better for the representation module (classification), while LSTM-based model works better for the prediction module (regression).
