"""process train_clean.csv

Train clean format:
[1]     landmark_id, images fields
[2:-]   landmark_id(class), images fields(file name)(space split)

Target format
im_dir, cont_id
"""
import os
import numpy as np
from tqdm import tqdm

def processing(input_file, output_file):
    """process train clean csv file"""
    print("Reading ...")
    with open(input_file, "r") as f_in:
        f_in.readline() # first line              
        all_lines = f_in.readlines()

    print("Processing ...")
    data_dict = {} # class id: image pathes
    class_id = "1620548888897360";
    img_paths = []
    for lines in tqdm(all_lines):
        folderandimage, dimension = lines.strip().split(":")         #删除空白符,分割突破信息和维度数
        #print(folderandimage)
        folder,blk,img = folderandimage.strip('"').split("\\")
        img_addr = os.path.join(folder, img)+".jpg"
        if(class_id == folder):
            img_paths.append(img_addr)
        else :
            data_dict[class_id] = img_paths
            class_id = folder
            img_paths = []
            img_paths.append(img_addr)


    print("Resorting class...")
    data_dict_sorted = {}                             # class id: image pathes
    data_class_num = len(data_dict.keys())
    data_img_num = 0
    for i, class_id in tqdm(zip(range(data_class_num), data_dict.keys())):
        data_dict_sorted[i] = data_dict[class_id]
        data_img_num += len(data_dict[class_id])
            
    print("Writing ...")
    with open(output_file, "w") as f_out:
        for class_id, img_pathes in data_dict_sorted.items():
            for img_path in img_pathes:
                f_out.write("{} {}\n".format(img_path, class_id))

    print("total class: {}".format(data_class_num))
    print("total img: {}".format(data_img_num))
    print("Done!\n")
    return None

def split_val(input_file, output_file_train, output_file_val):
    """Split train and val files"""
    print("Process the val and train file...\n")

    with open(input_file, "r")as f_in:
        lines = f_in.readlines()

    dataset_size = len(lines)
    val_size = round(dataset_size * 0.2)
    train_size = dataset_size - val_size
    train_indices = np.random.choice(dataset_size, train_size, replace=False)
    val_indices = np.array(list(set(range(dataset_size)) - set(train_indices)))

    with open(output_file_train, "w") as f_train:
        f_train.writelines(np.array(lines)[train_indices])
    
    with open(output_file_val, "w") as f_val:
        f_val.writelines(np.array(lines)[val_indices])
    
    print("Done\n")
        

if __name__ == "__main__":
    #input_dir = "./datasets/data/landmark/metadata/train_clean.csv"
    #output_dir = "./datasets/data/landmark/train/clean_all.txt"
    #output_dir_train = "./datasets/data/landmark/train/clean_train.txt"
    #output_dir_val = "./datasets/data/landmark/train/clean_val.txt"
    input_dir = "./LYMDataset/metadata/imageInformation.txt"
    output_dir = "./LYMDataset/clean_all.txt"
    output_dir_train = "./LYMDataset/clean_train.txt"
    output_dir_val = "./LYMDataset/clean_val.txt"
    processing(input_dir, output_dir)
    split_val(output_dir, output_dir_train, output_dir_val)