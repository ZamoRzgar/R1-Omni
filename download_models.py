from huggingface_hub import snapshot_download

# Download the vision model (siglip-base-patch16-224)
print("Downloading siglip-base-patch16-224...")
snapshot_download(repo_id="google/siglip-base-patch16-224", 
                  local_dir="C:/Users/zamor/models/siglip-base-patch16-224",
                  local_dir_use_symlinks=False)

# Download the audio model (whisper-large-v3)
print("Downloading whisper-large-v3...")
snapshot_download(repo_id="openai/whisper-large-v3", 
                  local_dir="C:/Users/zamor/models/whisper-large-v3",
                  local_dir_use_symlinks=False)

print("Both models downloaded successfully!")
