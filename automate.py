import os
import shutil
import json
import re
import torch

def run_segmentation():
    '''
    Runs Segmentation on images
    '''
    path = 'output' # To save images
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)           # Removes all the subdirectories!
        os.makedirs(path)
    #os.chdir('./output')
    os.system('python -u test.py --gpu 0 --imgs ../output_frames')
    print('Segmentation Done!')
    return
def run_parsing():
    '''
    Parses Slrum file and produces json 
    of frame_no: {dict_of_segments}, where
    dict_of_segments = {'bulidings':40,...} 
    (percentage of each of them in that frame)
    '''
    lines_ = []
    segmented_json = {} 
    with open('slurm-9063714.out') as f:
        lines = f.readlines()
        lines_.append(lines)
    #parse = lines_.split('Predictions in ')
    parse = lines_[0]
    i = 0
    while i< len(parse):
        line = parse[i]
        i += 1
        if 'Predictions in [../output_frames/' in line:
            line = line.split('Predictions in [../output_frames/')[-1]            
            #Extract image(frame no)
            temp = re.findall(r'\d+', line)
            res = list(map(int, temp))
            frame_no = res[0]
            #print(frame_no)
            items = {}
            while i < len(parse) and len(parse[i].split(':')) == 2:
                splits = parse[i].split(':')
                items[splits[0].strip()] = float(splits[1][:-2])
                i += 1
            segmented_json[frame_no] = items
    segmented_json = dict(sorted(segmented_json.items()))
    with open('data.json','w') as f:
        json.dump(segmented_json,f)
    print('Parsing Done')
    return 

def run_yolo():
    '''
    This fn give OUT_OF_MEMORY error, so better to use yolo repo.
    Runs Yolo on images
    '''
    #Taken from - https://pytorch.org/hub/ultralytics_yolov5/
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    # Images
    imgs = ['../output_frames/'+ str(x)+'.0.jpg' for x in range(1,len(os.listdir('../output_frames'))+1)]  # batch of images
    model.device = '0,1'
    
    # Inference
    results = model(imgs)
    
    
    # Results
    #results.print()
    results.save()  # or .show()
    
    #print(results.xyxy[0])  # img1 predictions (tensor)
    #print(results.pandas().xyxy[0]['name'])  # img1 predictions (pandas)
    #      xmin    ymin    xmax   ymax  confidence  class    name
    # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
    # 1  433.50  433.50   517.5  714.5    0.687988     27     tie
    # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
    # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
    yolo_data = []
    for item in results.pandas().xyxy:
        counts = {}
        for name in item['name']:
            if name not in counts:
                counts[name] = 0
            counts[name] += 1
        yolo_data.append(counts)
    #print(yolo_data)
    with open('yolo_data.json','w') as f:
        json.dump(yolo_data,f)
    print('Yolo Done')
run_segmentation()
run_parsing()
# run_yolo()
