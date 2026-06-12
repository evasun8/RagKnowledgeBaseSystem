from modelscope.hub.snapshot_download import snapshot_download

# download models/bge-m3 folder
model_dir = snapshot_download('BAAI/bge-m3', cache_dir='/Users/evasun/users/eva/ai_models/modelscope_cache/models')
print(f"The model is downloaded at: {model_dir}")