'''

THIS UTILS FLEXIBLY SPLITS THE DATASET BY OBJECTS CLASSES AND SIZE

'''

import os
import xml.etree.ElementTree as ET
import random
import shutil
import argparse
random.seed(2022)

classes =['GA', 'KK' , 'SS' , 
          'MD' , 'CK' , 'WB', 'KC' , 
          'CP' ,'CL' ,'LW' , 'UNKNOWN'] 
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--train_ratio', type=float, default = 0.8)

train_ratio = parser.parse_args().train_ratio
val_ratio = 0.1
train_val_ratio = train_ratio + val_ratio
test_ratio = 1 - train_val_ratio

# convert xml bbox to txt bbox 
# convert xyxy to normalized xywh
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

def convert_annotation(image_id,xml_folder,output_folder):
    in_file = open( xml_folder + '/%s.xml' % (image_id), encoding='UTF-8')
    out_file = open(output_folder + '/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    # correct the wrong info
    if w!=335:
        w=335
    if h!=880:
        h=880

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        
        # bbox correction is out of boundary
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)

        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def write_file(file_saved_path,image_ids,relative_img_path):
    f = open(file_saved_path, 'w')
    for image_id in image_ids:
        image_path = image_id.split('.')[0] + '.jpg'
        train_path = os.path.join(relative_img_path,image_path)
        f.write(train_path+'\n')

def move_file(filenames_list,parentpath,purpose='train',filetype='labels'):

    para_folder = '/'.join(parentpath.split('/')[0:-1])
    objective_dir = os.path.join(para_folder,filetype+'/'+purpose)
    if not os.path.exists(objective_dir):
        os.makedirs(objective_dir)

    for file in filenames_list:
        if filetype == 'labels':
            filename = file.split('.')[0]+'.txt'
        else:
            filename = file.split('.')[0]+'.jpg'
        src_filepath = os.path.join(parentpath,filename)
        dest_filepath = os.path.join(objective_dir,filename)
        shutil.copy(src_filepath, dest_filepath)


# reset
def DeleteFiles(path, remainDirsList):
    dirsList = []
    dirsList = os.listdir(path)
    for f in dirsList:
        if f not in remainDirsList:
            filepath = os.path.join(path,f)
            if os.path.isdir(filepath):
                shutil.rmtree(filepath, True)
            else:
                os.remove(filepath)
path='/Users/rc/Documents/code/datasets/THz_Dataset'
dirsList=['JPEGImages','Annotations']
DeleteFiles(path,remainDirsList=dirsList)

data_root = '/Users/rc/Documents/code'
image_folder = data_root + '/datasets/THz_Dataset/JPEGImages'
xml_annotation_folder = data_root + '/datasets/THz_Dataset/Annotations'
txt_annotation_folder = data_root + '/datasets/THz_Dataset/all_labels'
if not os.path.exists(txt_annotation_folder):
    os.makedirs(txt_annotation_folder)

# convert annotation xml to label txt
for filename in os.listdir(xml_annotation_folder):
    image_id = filename.split('.')[0]
    convert_annotation(image_id,xml_annotation_folder,txt_annotation_folder)

# trained with only objects
for filename in os.listdir(txt_annotation_folder):
    filepath = os.path.join(txt_annotation_folder,filename)
    if os.path.getsize(filepath) == 0:
        os.remove(filepath)

# generate train, val and test file
filename_list=os.listdir(txt_annotation_folder)
random.shuffle(filename_list)
length_list = len(filename_list)
train_filenames = filename_list [0: int(train_ratio*length_list)]
val_filenames = filename_list [int(train_ratio*length_list):int(train_val_ratio*length_list)]
test_filenames = filename_list [int(train_val_ratio*length_list):]

# move label files
move_file(train_filenames,txt_annotation_folder,purpose='train',filetype='labels')
move_file(val_filenames,txt_annotation_folder,purpose='val',filetype='labels')
move_file(test_filenames,txt_annotation_folder,purpose='test',filetype='labels')


# move img files
move_file(train_filenames,image_folder,purpose='train',filetype='images')
move_file(val_filenames,image_folder,purpose='val',filetype='images')
move_file(test_filenames,image_folder,purpose='test',filetype='images')

# write txt files
txtsavepath = data_root + '/datasets/THz_Dataset'
file_test = txtsavepath + '/test.txt'
file_train = txtsavepath + '/train.txt'
file_val = txtsavepath + '/val.txt'
relative_img_path = './images'

write_file(file_train,train_filenames,relative_img_path+'/train')
write_file(file_val,val_filenames,relative_img_path+'/val')
write_file(file_test,test_filenames,relative_img_path+'/test')

