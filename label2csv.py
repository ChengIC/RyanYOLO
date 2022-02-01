# Calcuate the results from after inference

import os
import pandas as pd

exp_label_folder = '/Users/rc/Documents/code/yolov5/runs/detect/exp/labels'
ground_truth_label_folder = '/Users/rc/Documents/code/THz_Dataset/train/labels'

# utils to transfer txt from file to array data
def readTXT(txtpath):
    data = []
    txtfile = open(txtpath, 'r')
    txtlines = txtfile.readlines()
    for single_line in txtlines:
        single_line_array = single_line.split()
        single_line_array = [float(s) for s in single_line_array]
        data.append(single_line_array)
    return data

def labelsToCSV(labels_folder, data_type='detection'):
    if data_type == 'detection':
        data_dict = {
                'ImageID':[],
                'Conf':[],
                'LabelName':[],
                'XMin': [],
                'XMax': [],
                'YMin': [],
                'YMax': [],
          }
    else:
        data_dict = {
                'ImageID':[],
                'LabelName':[],
                'XMin': [],
                'XMax': [],
                'YMin': [],
                'YMax': [],
          }

    for filename in os.listdir(labels_folder):

        label_path = os.path.join(labels_folder,filename)
        image_id  = filename.strip('.txt')

        if data_type == 'detection':
            for d in readTXT(label_path):
                data_dict ['ImageID'].append(image_id)
                data_dict ['Conf'].append(d[1])
                data_dict ['LabelName'].append(d[0])
                data_dict ['XMin'].append(d[2])
                data_dict ['XMax'].append(d[3])
                data_dict ['YMin'].append(d[4])
                data_dict ['YMax'].append(d[5])
        else:
            for d in readTXT(label_path):
                data_dict ['ImageID'].append(image_id)
                data_dict ['LabelName'].append(d[0])
                data_dict ['XMin'].append(d[1])
                data_dict ['XMax'].append(d[2])
                data_dict ['YMin'].append(d[3])
                data_dict ['YMax'].append(d[4])

    return data_dict


gt_dict = labelsToCSV(ground_truth_label_folder, data_type='ground_truth')
detect_dict = labelsToCSV(exp_label_folder, data_type='detection')

df1 = pd.DataFrame(gt_dict)
df2 = pd.DataFrame(detect_dict)

df1.to_csv('annotations.csv')
df2.to_csv('detections.csv')

