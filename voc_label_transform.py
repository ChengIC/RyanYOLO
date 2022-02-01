# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
import shutil 
split_sets = ['train', 'val', 'test']
classes = ['HUMAN','GA', 'KK' , 'SS' , 'MD' , 'CK' , 'WB', 'KC' , 'CP' ,'CL' ,'LW' , 'UNKNOWN']  

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
    if w!=335:
        w=335
    if h!=880:
        h=880
    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        # difficult = obj.find('Difficult').text
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

data_root = '/Users/rc/Documents/code/THz_Dataset'

xml_folder = data_root + '/Annotations'
imgs_folder = data_root + '/JPEGImages' 
split_name_folder = data_root + '/split_name'

for image_set in split_sets:
    output_img_folder = data_root + '/' + image_set + '/images'
    output_label_folder = data_root + '/' + image_set + '/labels'
    if not os.path.exists(output_label_folder):
        os.makedirs(output_label_folder)
    if not os.path.exists(output_img_folder):
        os.makedirs(output_img_folder) 
        
    image_ids = open( split_name_folder + '/%s.txt' % (image_set)).read().strip().split()
    
    list_file = open(data_root + '/%s.txt' % (image_set), 'w')

    for image_id in image_ids:
        list_file.write(imgs_folder + '/%s.jpg\n' % (image_id))
        convert_annotation(image_id,xml_folder,output_label_folder)

        src_img_path = imgs_folder + '/' + image_id + '.jpg'
        dst_img_path = output_img_folder + '/' + image_id + '.jpg'
        shutil.copy(src_img_path,dst_img_path)

    list_file.close()
