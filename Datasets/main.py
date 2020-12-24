from Datasets.DataPreprocessing import SVDCompression
import os
from tqdm import tqdm
# 遍历所有图片，执行SVD压缩并保存

# ********************* 参数 s *********************
src_dir_root = "/home/harvey/Datasets/Face/19/AFLW/"
save_dir = "/home/harvey/Datasets/Face/19/LR"
K = 15
# ********************* 参数 e *********************

files = os.listdir(src_dir_root)


for f in tqdm(files):
    path = os.path.join(src_dir_root, f)
    SVDCompression(path, save_dir, K)

