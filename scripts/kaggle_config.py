import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Setup Kaggle API với username + key
os.environ['KAGGLE_USERNAME'] = '26'
os.environ['KAGGLE_KEY'] = 'KGAT_6df22e37e397285dbacd455d63bb391d'

# Initialize API
api = KaggleApi()
api.authenticate()

print("✅ Kaggle API authenticated!")

# List datasets (dùng dataset_list, không phải datasets_list)
datasets = api.dataset_list(search='face')
print(f"\nFound {len(datasets)} face datasets")

for dataset in datasets[:5]:
    print(f"- {dataset.ref}: {dataset.title}")

# Download dataset example
# api.dataset_download_files('dataset-name', path='../data/datasets/', unzip=True)
