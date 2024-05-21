import os
import numpy as np
from tqdm import tqdm

clip_feature_path="/ocean/projects/cis210027p/ywu13/datasets/how2/clip_feature_4_frames"
wav_scp_path="/ocean/projects/cis210027p/ywu13/projects/VisualASR/OWSM_visual/espnet/egs2/owsm_v3.1/s2t1/dump/raw/train_visual/wav.scp"
visual_feature_path="/ocean/projects/cis210027p/ywu13/projects/VisualASR/OWSM_visual/espnet/egs2/owsm_v3.1/s2t1/dump/raw/train_visual/clip_feature"
wav_scp=open(wav_scp_path,'r')
visual_feature=open(visual_feature_path,'w')
for line in tqdm(wav_scp.readlines()):
    _line=line.strip().split(" ")
    sid=_line[0]
    real_sid="_".join(sid.split("_")[1:-4])
    npy_path=os.path.join(clip_feature_path, real_sid+'_clip_features.npy')
    if os.path.exists(npy_path):
        data = np.load(npy_path)

        if data.shape[0] == 4:
            visual_feature.write(sid+' '+npy_path+'\n')
            
    