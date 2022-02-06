
import os 

model_names = ['yolov5s.pt','yolov5m.pt','yolov5l.pt']
ratios = [0.5,0.55,0.6,0.65,0.7,0.75,0.8]
for model in model_names:
    for ratio in ratios:
        os.system('python flexible_split.py --train_ratio %s'%str(ratio))
        print (len(os.listdir('/Users/rc/Documents/code/datasets/THz_Dataset/images/train')))
        save_path = 'runs/train_ratio'+ str(int(100*ratio)) + '_' + model.split('.')[0]
        os.system('python train.py --project %s --weights %s' %(save_path,model))