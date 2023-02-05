# This file is no longer essential for main project file. But keep for now as needed for previous approach.

import pandas as pd
import os

def Data_Frame_Img(file_path):

    Dataset = os.listdir(file_path)
    img_df = pd.DataFrame(Dataset,columns=["Image"])

    image_path = []
    pic_dir = os.listdir(file_path)

    for item in pic_dir:
        image_path.append(file_path +item)

    image_path_df = pd.DataFrame(image_path,columns=["Image Path"])

    image_dataset_df = pd.concat([img_df,image_path_df],axis=1)

    return image_dataset_df

def Entire_dataframe (image_dataset_df, Labels_data):

    labels_data_df = pd.DataFrame(Labels_data,columns=["Labels"])
    labels_df = pd.DataFrame(Labels_data,columns=["Letters"])

    Labels_df = pd.concat([labels_data_df,labels_df],axis=1)

    Labels_df.Letters.replace({
    1:'A', 
    2:'B',
    3:'C',
    4:'D',
    5:'E',
    6:'F',
    7:'G',
    8:'H',
    9:'I',
    10:'J',
    11:'K',
    12:'L',
    13:'M',
    14:'N',
    15:'O',
    16:'P',
    17:'Q',
    18:'R',
    19:'S',
    20:'T',
    21:'U',
    22:'V',
    23:'W',
    24:'X',
    25:'Y',
    26:'Z'
    
    },inplace=True)




    Labels_df.Labels.replace({
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    19: 18,
    20: 19,
    21: 20,
    22: 21,
    23: 22,
    24: 23,
    25: 24,
    26: 25
    
    },inplace=True)

    Images_And_Labels_df = pd.concat([image_dataset_df,Labels_df],axis=1)
    
    


    return Images_And_Labels_df