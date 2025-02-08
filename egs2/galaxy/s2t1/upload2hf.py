import os
from huggingface_hub import HfApi
from glob import glob

# 配置参数
repo_id = "Yihan2023/galaxy_dump"  # 替换为您的仓库名称
file_dir = "/scratch/bbjs/ywu13/projects/Visual_ASR/galaxy_dataset_code/espnet/egs2/galaxy/s2t1/dump_yodas/raw"  # 文件目录路径

# 初始化 API
api = HfApi()

# 确保已经登录
try:
    api.whoami()
except:
    print("You need to log in to Hugging Face first. Run 'huggingface-cli login'.")
    exit(1)

# 遍历并上传文件，保留相对路径，只上传以 yodas 开头的文件夹
# for file_path in glob(f'{file_dir}/yodas*/**/*', recursive=True):
#     if os.path.isfile(file_path):  # 确保是文件而不是目录
#         # 获取相对路径
#         relative_path = os.path.relpath(file_path, file_dir)
        
#         try:
#             # 上传文件
#             api.upload_file(
#                 path_or_fileobj=file_path,
#                 path_in_repo=relative_path,
#                 repo_id=repo_id,
#                 repo_type="dataset",  # 数据集类型
#             )
#             print(f"Uploaded: {relative_path}")
#         except Exception as e:
#             print(f"Failed to upload {relative_path}. Error: {e}")
file_path = "/scratch/bbjs/ywu13/projects/Visual_ASR/owsm_visual/espnet/egs2/owsm_v3.1/s2t1/exp/s2t_train_s2t_ebf_conv2d_size1024_e18_d18_piecewise_lr2e-4_warmup60k_flashattn_vis_raw_bpe50000/pretrained.pth"
relative_path = "exp/s2t_train_s2t_ebf_conv2d_size1024_e18_d18_piecewise_lr2e-4_warmup60k_flashattn_vis_raw_bpe50000/pretrained.pth"
api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=relative_path,
        repo_id=repo_id,
        repo_type="dataset",  # 数据集类型
    )

print("All files uploaded successfully!")