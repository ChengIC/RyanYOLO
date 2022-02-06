from map_boxes import mean_average_precision_for_boxes
import pandas as pd

ann = pd.read_csv('annotations.csv')
det = pd.read_csv('detections.csv')

ann = ann[['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax']].values
det = det[['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax']].values
mean_ap, average_precisions = mean_average_precision_for_boxes(ann, det, iou_threshold=0.1)

print (mean_ap)

print (average_precisions)