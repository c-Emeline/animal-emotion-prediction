from torchvision import models
from PIL import Image
from torchvision import transforms
import torch
from transformers import pipeline
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


#get models and set evaluation mode
model121 = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
model121.eval()
model201 = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
model201.eval()


#gives class label
data_path = 'imagenet_classes.txt'


#process image before inference
def preprocess_image(image) :
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    processed_img = preprocess(image)
    return processed_img.unsqueeze(0)


#Densenet121
def get_prediction_img_121(image_path):
    image_bytes = Image.open(image_path)
    tensor_image = preprocess_image(image_bytes)

    #without gradient computation
    with torch.no_grad():
        output= model121(tensor_image)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    #retrieve categories from imagenet_classes.txt
    with open(data_path, 'r') as f:
        categories = [s.strip() for s in f.readlines()]

    category_nb, cat_id = torch.topk(probabilities, 1)
    category = categories[cat_id[0]]

    return category, cat_id


#Densenet201
def get_prediction_img_201(image_path):
    image_bytes = Image.open(image_path)
    tensor_image = preprocess_image(image_bytes)

    with torch.no_grad():
        output= model201(tensor_image)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    with open(data_path, 'r') as f:
        categories = [s.strip() for s in f.readlines()]

    category_nb, cat_id = torch.topk(probabilities, 1)
    category = categories[cat_id[0]]

    return category, cat_id


#EmoRoBERTa
def get_prediction_txt_roberta(text):
    emotion = pipeline('sentiment-analysis', 
                        model='arpanghoshal/EmoRoBERTa')

    emotion_labels = emotion(text)
    return emotion_labels[0]['label'], emotion_labels[0]['score']


#Distil Bert
def get_prediction_txt_distil_bert(text):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]
