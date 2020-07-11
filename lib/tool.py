# Preprocessing of the input image
from torchvision import transforms
import torch
def preprocess_ImageNet(img):
    
    # define the preprocessor
    preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])


    # pass the input image through the preprocessor
    img_t = preprocess(img)
    
    # create a batch
    img_t = img_t[:3]
    batch_t = torch.unsqueeze(input=img_t, dim=0) # insert out image in batch
        
    return batch_t

def postprocess(output_data, topPred=1):
    # get class names
    with open('data/imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
        
    # calculate human-readable value by softmax
    confidences = torch.nn.functional.softmax(output_data, dim=1)[0] * 100

    # find top-5 predicted classes
    _, indices = torch.sort(output_data, descending=True)
    i = 0
    
    # print the top classes predicted by the model
    keepPredClass = []
    keepPredConf = []
    
    while i<topPred:
        class_idx = indices[0][i]
        
        print( "Predicted class: {0}\t Confidence: {1:2.2f}%".format(
                                classes[class_idx], 
                                confidences[class_idx].item()))
        
        keepPredClass.append(classes[class_idx])
        keepPredConf.append(float(confidences[class_idx]))
        i += 1    
    
    return {'class': keepPredClass,
            'conf': keepPredConf}

