import os
from re import X
from tkinter import Y
import numpy as np
from tqdm import tqdm

def processing(input_file):
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
        print(folderandimage)
        folder = folderandimage.strip('"')
        X,Y,Z=folder.split("\\")
        print(folder)
        print(X)
        print(Y)
        print(Z)

     
        #img_addr = os.path.join(img[0],img[1], img[2], img)+".jpg"


if __name__ == "__main__":
    input_dir = "./testdata.txt"
    processing(input_dir)