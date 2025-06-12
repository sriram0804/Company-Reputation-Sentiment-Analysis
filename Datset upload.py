import pandas as pd


dataset_path = "/content/drive/MyDrive/dataset/"


file_names = {
    "After_Layoff": "After_Layoff.csv",
    "After_Layoff_General": "After_Layoff_General.csv",
    "Before_Layoff": "Before_Layoff.csv",
    "Fb_Training": "Fb_Training.xltx",
    "vect": "vect.csv"
}

datasets = {}

for name, file in file_names.items():
    file_path = dataset_path + file
    try:
        if file.endswith(".csv"):
            datasets[name] = pd.read_csv(file_path, on_bad_lines="skip")
        elif file.endswith(".xltx"):
            datasets[name] = pd.read_excel(file_path, engine="openpyxl")

        print(f" Loaded {name}: {datasets[name].shape}")
    except FileNotFoundError:
        print(f" File not found: {file_path}. Check if it exists in Google Drive.")
        datasets[name] = None
    except Exception as e:
        print(f" Failed to load {name}: {str(e)}")
        datasets[name] = None
