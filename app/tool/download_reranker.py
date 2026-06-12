from modelscope.hub.snapshot_download import snapshot_download

local_dir = r"/Users/evasun/users/eva/ai_models/modelscope_cache/models"

snapshot_download(
    model_id="BAAI/bge-reranker-large",
    cache_dir=local_dir,
)

print("Download completed. The download path is：", local_dir)