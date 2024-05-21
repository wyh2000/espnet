import os

def load_keys_from_reference(ref_path):
    """ 从参照文件中加载keys。 """
    with open(ref_path, 'r') as file:
        keys = [line.split()[0] for line in file]
    return keys

def filter_and_save_files(file_path, keys, save_folder):
    """ 过滤并保存文件，使其只包含特定的keys，并保持顺序一致。 """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    with open(file_path, 'r') as file:
        content = file.readlines()
    
    # 创建一个字典来存储当前文件的键值对
    content_dict = {line.split()[0]: " ".join(line.split()[1:]) for line in content if line.strip()}
    
    # 过滤和排序内容
    filtered_content = [f"{key} {content_dict[key]}\n" for key in keys if key in content_dict]
    
    # 写入新文件
    save_path = os.path.join(save_folder, os.path.basename(file_path))
    with open(save_path, 'w') as file:
        file.writelines(filtered_content)

# 使用
ref_path = 'dump/raw/train_visual/clip_feature'  # 参照文件路径
folder_path = 'dump/raw/train_visual'  # 其他文件所在的文件夹
save_folder = 'dump/raw/train_visual'  # 保存过滤后文件的文件夹
file_list = ["spk2utt", "text", "utt2spk", "wav.scp", "text.ctc", "text.prev", "utt2num_samples"]
keys = load_keys_from_reference(ref_path)
for f in file_list:
    file_path=os.path.join(folder_path, f)
    filter_and_save_files(file_path, keys, save_folder)
