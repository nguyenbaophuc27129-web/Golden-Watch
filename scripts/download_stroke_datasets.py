import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Setup Kaggle API
os.environ['KAGGLE_USERNAME'] = '26'
os.environ['KAGGLE_KEY'] = 'KGAT_6df22e37e397285dbacd455d63bb391d'

api = KaggleApi()
api.authenticate()

print("📥 Downloading STROKE DETECTION datasets...\n")

# Dataset list với module
datasets_to_download = {
    'speech': [
        ('iamhungundji/dysarthria-detection', 'Dysarthria Detection'),
        ('aryashah2k/noise-reduced-uaspeech-dysarthria-dataset', 'UASpeech Dysarthria'),
    ],
    'gait': [
        ('vafaeii/kth-action-recognition-dataset', 'KTH Action Recognition'),
        ('rakeshkumarpudi/gait-analysis-dataset-cerebellar-ataxia', 'Gait Analysis Ataxia'),
    ],
    'pose': [
        ('trainingdatapro/pose-estimation', 'Human Pose Estimation'),
        ('nicolehoelzl/mpii-human-pose-data', 'MPII Human Pose'),
    ],
    'fall': [
        ('uttejkumarkandagatla/fall-detection-dataset', 'Fall Detection Dataset'),
    ]
}

base_path = '../data/datasets'

for module, dataset_list in datasets_to_download.items():
    print(f"\n📌 Module: {module.upper()}")
    module_path = os.path.join(base_path, module)
    os.makedirs(module_path, exist_ok=True)

    for dataset_name, description in dataset_list:
        print(f"   Downloading: {description}")
        try:
            api.dataset_download_files(
                dataset_name,
                path=module_path,
                unzip=True,
                quiet=False
            )
            print(f"   ✅ Downloaded: {description}\n")
        except Exception as e:
            print(f"   ❌ Error downloading {description}: {e}\n")

print("\n✅ All downloads complete!")
