import glob
import os

import tqdm

path = "C:\\Users\\Christoph\\Desktop\\dataset\\seg_mask128png\\*.png"

result = glob.glob(path)

for file_name in tqdm.tqdm(result):
    old_name = file_name
    new_name = old_name.replace("_seg", "")
    os.rename(old_name, new_name)
