import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Setup Kaggle API
os.environ['KAGGLE_USERNAME'] = '26'
os.environ['KAGGLE_KEY'] = 'KGAT_6df22e37e397285dbacd455d63bb391d'

api = KaggleApi()
api.authenticate()

print("🔍 Searching for STROKE DETECTION datasets...\n")

# Search keywords cho stroke detection
keywords = [
    'facial paralysis',
    'face asymmetry',
    'palsy',
    'dysarthria',
    'speech disorder',
    'stroke',
    'gait analysis',
    'walking pattern',
    'human pose',
    'fall detection'
]

for keyword in keywords:
    print(f"📌 Keyword: '{keyword}'")
    try:
        datasets = api.dataset_list(search=keyword)
        if datasets:
            print(f"   Found {len(datasets)} datasets:")
            for ds in datasets[:3]:  # Show top 3
                print(f"   - {ds.ref}: {ds.title}")
        else:
            print("   No datasets found")
    except Exception as e:
        print(f"   Error: {e}")
    print()
