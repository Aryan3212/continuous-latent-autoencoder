import os
import shutil
import zipfile
import urllib.request
from huggingface_hub import snapshot_download, login
from kaggle.api.kaggle_api_extended import KaggleApi

# ==========================================
# ⚙️ CONFIGURATION SECTION
# ==========================================
BASE_DIR = r"C:\Bengali_Speech_Data"

# 1. Hugging Face Token (Read Access)
HF_TOKEN = "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"

# 2. Kaggle Credentials (Open your kaggle.json to see these)
KAGGLE_USERNAME = "aryanrahman"
KAGGLE_KEY = "KGAT_38471085ebbafd3d0c544e1954296b39"

# ==========================================

def setup_auth():
    """Sets up authentication for both Hugging Face and Kaggle."""
    print("🔑 Setting up authentication...")
    
    # Hugging Face Login
    try:
        if HF_TOKEN.startswith("PASTE"):
            print("⚠️ WARNING: You didn't paste your Hugging Face Token!")
        else:
            login(token=HF_TOKEN)
            print("✅ Hugging Face Logged In.")
    except Exception as e:
        print(f"❌ Hugging Face Login Failed: {e}")

    # Kaggle Login (Environment Variables Method)
    if KAGGLE_USERNAME.startswith("PASTE") or KAGGLE_KEY.startswith("PASTE"):
        print("⚠️ WARNING: You didn't paste your Kaggle Credentials!")
    else:
        # This tricks Kaggle into thinking the variables are set in the system
        os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
        os.environ['KAGGLE_KEY'] = KAGGLE_KEY
        print("✅ Kaggle Credentials Set.")

def download_hf_dataset(repo_id, folder_name, allow_patterns=None):
    """Downloads a dataset from Hugging Face."""
    print(f"\n------------------------------------------------")
    print(f"🤗 Downloading {repo_id}...")
    dest_dir = os.path.join(BASE_DIR, folder_name)
    
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=dest_dir,
            allow_patterns=allow_patterns,
            token=HF_TOKEN
        )
        print(f"✅ Finished: {folder_name}")
    except Exception as e:
        print(f"❌ Failed to download {repo_id}: {e}")

def download_openslr_part(part_id):
    """Downloads, unzips, and deletes a specific OpenSLR part."""
    dest_dir = os.path.join(BASE_DIR, "OpenSLR53")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    url = f"https://www.openslr.org/resources/53/asr_bengali_{part_id}.zip"
    zip_filename = f"asr_bengali_{part_id}.zip"
    zip_path = os.path.join(dest_dir, zip_filename)
    
    print(f"⬇️  OpenSLR: Downloading Part {part_id}...")
    
    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"📦 Unzipping Part {part_id}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        os.remove(zip_path)
        print(f"✅ Part {part_id} Done.")
    except Exception as e:
        print(f"❌ Failed OpenSLR Part {part_id}: {e}")

def download_kaggle_dataset(slug, folder_name, is_competition=False):
    """Downloads from Kaggle using API credentials."""
    print(f"\n------------------------------------------------")
    print(f"🦆 Downloading Kaggle Dataset: {slug}")
    dest_dir = os.path.join(BASE_DIR, folder_name)
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    try:
        api = KaggleApi()
        api.authenticate() # Uses the environment variables we set above
        
        print("⏳ Downloading zip (Please wait)...")
        if is_competition:
            api.competition_download_files(slug, path=dest_dir, quiet=False)
            zip_name = f"{slug}.zip" 
        else:
            api.dataset_download_files(slug, path=dest_dir, quiet=False, unzip=False)
            zip_name = f"{slug.split('/')[1]}.zip"
            
        zip_path = os.path.join(dest_dir, zip_name)
        
        # Fallback search
        if not os.path.exists(zip_path):
            files = [f for f in os.listdir(dest_dir) if f.endswith('.zip')]
            if files:
                zip_path = os.path.join(dest_dir, files[0])
            else:
                print("❌ Error: Zip file not found after download.")
                return

        print(f"📦 Unzipping {os.path.basename(zip_path)}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
            
        print("🗑️  Deleting Zip file...")
        os.remove(zip_path)
        print(f"✅ Finished: {folder_name}")
        
    except Exception as e:
        print(f"❌ Failed to download {slug}: {e}")

def main():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    setup_auth()

    # # 1. Shrutilipi (Grammar)
    # download_hf_dataset("ai4bharat/Shrutilipi", "Shrutilipi", allow_patterns="bn/*")
    #
    # # 2. SUBAK.KO (Clean Read)
    # download_hf_dataset("SUST-CSE-Speech/SUBAK.KO", "SUBAK_KO")
    #
    # 3. IndicVoices (Spontaneous)
    download_hf_dataset("ai4bharat/indicvoices_r", "IndicVoices", allow_patterns="Bengali/*")
    #
    # # 4. Kathbath (Probe Only)
    # download_hf_dataset("ai4bharat/Kathbath", "Kathbath_Probe", allow_patterns=["bn/test/*", "bn/valid/*"])
    #
    # # 5. OOD-Speech (Kaggle - Competition)
    # # WARNING: Accept rules at https://www.kaggle.com/c/bengaliai-speech/rules
    # download_kaggle_dataset("bengaliai-speech", "OOD_Speech", is_competition=True)
    #
    # # 6. RegSpeech12 (Kaggle - Dataset)
    # download_kaggle_dataset("mdrezuwanhassan/regspeech12", "RegSpeech12", is_competition=False)
    #
    # # 7. OpenSLR (The Loop)
    # print("\n------------------------------------------------")
    # print("⬇️ Starting OpenSLR Download...")
    # parts = list(range(10)) + ['a', 'b', 'c', 'd', 'e']
    # for p in parts:
    #     download_openslr_part(p)
    #
    print("\n✅ DONE! All datasets are in:", BASE_DIR)

if __name__ == "__main__":
    main()
