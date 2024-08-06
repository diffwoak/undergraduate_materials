# ## CLIP zero-shot
# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image
# ## install packages: torch, transformers==4.23.1


# if __name__ == '__main__':
#     print('Loading Model, wait for a minute.')
#     model_name = "openai/clip-vit-base-patch32"
#     model = CLIPModel.from_pretrained(model_name)
#     model = model.eval()
#     for param in model.parameters():
#         param.requires_grad = False
#     processor = CLIPProcessor.from_pretrained(model_name)

#     image = Image.open(r"Photo path")
#     text_labels = ["a photo of a cat", "a photo of a dog", "a photo of a horse"]
#     inputs = processor(text=text_labels, images=image, return_tensors="pt", padding=True)
#     outputs = model(**inputs)
#     logits_per_image = outputs.logits_per_image
#     probs = logits_per_image.softmax(dim=1)
#     predicted_label_idx = probs.argmax()

#     print(predicted_label_idx)
#     print(text_labels[predicted_label_idx])


# ---------------------------------------------------------------------------------------


# import os
# import torch
# import clip
# from PIL import Image
# from torchvision import transforms
# from transformers import CLIPProcessor, CLIPModel

# if __name__ == '__main__':
    
#     images_dir = "datasets/CUB_200_2011/CUB_200_2011/images"
#     claz = []
#     for name in os.listdir(images_dir):
#         tmp = name.split('.')[-1].lower()
#         claz.append(tmp)
# #     print(claz)    
        
#     model_name = "ViT-L-14-336px.pt"
#     print(f'Loading Model {model_name}, wait for a minute.')
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     model, processor = clip.load(model_name, device=device)
#     tokenizer = clip.tokenize
#     image = Image.open("datasets/CUB_200_2011/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg")
    
#     process_image = processor(image).unsqueeze(0).to(device)
#     text = tokenizer(["a bird", "a dog", "a cat"]).to(device)
    
#     text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in claz]).to(device)
    
#     with torch.no_grad():
#         logits_per_image, logits_per_text = model(process_image, text)
#         probs = logits_per_image.softmax(dim=-1).cpu().numpy()
 
#     print("Label probs:", probs)
      
    
# ---------------------------------------------------------------------------------------

# 使用 ViT-B-16.pt 模型

# import os
# import torch
# import clip
# from PIL import Image
# from torchvision import transforms
# from transformers import CLIPProcessor, CLIPModel

# def load_images_and_labels(images_dir):
#     images = []
#     labels = []
#     label_dict = {}
#     idx = 0
    
#     for image_set in os.listdir(images_dir):
#         set_dir = images_dir + "/" + image_set
#         cnt = 0
#         for image_name in os.listdir(set_dir):
#             if cnt == 20:
#                 break
#             cnt += 1
#             img_path = set_dir + "/" + image_name
#             label = set_dir.split('.')[-1]
# #             label = set_dir
# #             print(label + " " + img_path)
#             if label not in label_dict:
#                 label_dict[label] = idx
#                 idx += 1
#             images.append(img_path)
#             labels.append(label_dict[label])

#     return images, labels, label_dict

# def predict(model, processor, image_path, device):
#     image = Image.open(image_path).convert("RGB")
#     image_input = processor(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         image_features = model.encode_image(image_input)
#     return image_features

# def main():
#     available_models = clip.available_models()
#     print("Available CLIP models:")
#     for model in available_models:
#         print(model)
#     images_dir = "datasets/CUB_200_2011/CUB_200_2011/images"
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     model_name = "ViT-L-14-336px.pt"
# #     model_name = "ViT-B-32.pt"
#     print(f'Loading Model {model_name}, wait for a minute.')
#     model, preprocess = clip.load(model_name, device=device)
#     print("Model Done")
    
#     images, labels, label_dict = load_images_and_labels(images_dir)
#     text_inputs = torch.cat([clip.tokenize(f"a photo of a {label}") for label in label_dict.keys()]).to(device)
    
#     correct_predictions = 0
#     total_images = len(images)
    
#     with torch.no_grad():
#         text_features = model.encode_text(text_inputs)
#         text_features /= text_features.norm(dim=-1, keepdim=True)
    
#     for i, (image_path, true_label) in enumerate(zip(images, labels)):
#         image_features = predict(model, preprocess, image_path, device)
        
#         # 对图像和文本的特征向量进行归一化处理
#         image_features /= image_features.norm(dim=-1, keepdim=True)
        
#         # 使用内积计算图像特征和文本特征之间的相似度。然后应用 softmax 函数将相似度转换为概率分布
#         similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
#         predicted_label_idx = similarity.argmax().item()
        
#         if predicted_label_idx == true_label:
#             correct_predictions += 1
        
#         # 每处理100张图像后打印一次当前的准确率
#         if (i + 1) % 50 == 0:
#             current_accuracy = correct_predictions / (i + 1)
#             print(f"Processed {i + 1}/{total_images} images, Current Accuracy: {current_accuracy:.4f}")
    
#     # 最终准确率
#     final_accuracy = correct_predictions / total_images
#     print(f"Final Accuracy: {final_accuracy:.4f}")

# if __name__ == '__main__':
#     main()

    
# ------------------------------------------------------------------------

# 微调

# import os
# import torch
# from torch.utils.data import Dataset, DataLoader, random_split
# from PIL import Image
# from torchvision import transforms
# from transformers import CLIPProcessor, CLIPModel
# import tqdm
# import random

# # 自定义数据集类
# class CustomDataset(Dataset):
#     def __init__(self, images, labels, processor):
#         self.images = images
#         self.labels = labels
#         self.processor = processor

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         image_path = self.images[idx]
#         label = self.labels[idx]
#         image = Image.open(image_path).convert("RGB")
#         image_input = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
#         return image_input, label

# # 加载图像和标签
# def load_images_and_labels(images_dir, num_samples_per_class=50):
#     images = []
#     labels = []
#     label_dict = {}
#     idx = 0
    
#     for image_set in os.listdir(images_dir):
#         set_dir = os.path.join(images_dir, image_set)
#         all_images = os.listdir(set_dir)
#         random.shuffle(all_images)  # 随机打乱顺序
        
#         for image_name in all_images[:num_samples_per_class]:  # 只使用部分数据
#             img_path = os.path.join(set_dir, image_name)
#             label = set_dir.split('.')[-1]
#             if label not in label_dict:
#                 label_dict[label] = idx
#                 idx += 1
#             images.append(img_path)
#             labels.append(label_dict[label])

#     return images, labels, label_dict

# # 训练函数
# def train(model, dataloader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0
#     for images, labels in tqdm.tqdm(dataloader, desc="Training"):
#         images, labels = images.to(device), labels.to(device)
#         outputs = model.get_image_features(images)
#         loss = criterion(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     return total_loss / len(dataloader)

# # 验证函数
# def validate(model, dataloader, criterion, device):
#     model.eval()
#     total_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for images, labels in tqdm.tqdm(dataloader, desc="Validating"):
#             images, labels = images.to(device), labels.to(device)
#             outputs = model.get_image_features(images)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#             preds = torch.argmax(outputs, dim=1)
#             correct += (preds == labels).sum().item()
#     accuracy = correct / len(dataloader.dataset)
#     return total_loss / len(dataloader), accuracy

# def main():
#     images_dir = "datasets/CUB_200_2011/CUB_200_2011/images"
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     model = CLIPModel.from_pretrained("./clip-vit-base-patch32").to(device)
#     processor = CLIPProcessor.from_pretrained("./clip-vit-base-patch32")
    
#     for name, param in model.named_parameters():
#         print(name)
    
#     images, labels, label_dict = load_images_and_labels(images_dir)
    
#     dataset = CustomDataset(images, labels, processor)
#     train_size = int(0.6 * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
#     train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
#     parameters_to_train = [
#     model.visual_projection.weight,
#     model.text_projection.weight,
#     model.vision_model.post_layernorm.weight,
#     model.vision_model.post_layernorm.bias,
#     model.text_model.final_layer_norm.weight,
#     model.text_model.final_layer_norm.bias,
#     model.vision_model.encoder.layers[11].self_attn.out_proj.weight,
#     model.vision_model.encoder.layers[11].self_attn.out_proj.bias,
#     model.text_model.encoder.layers[11].self_attn.out_proj.weight,
#     model.text_model.encoder.layers[11].self_attn.out_proj.bias,
#     model.vision_model.encoder.layers[11].mlp.fc2.weight,
#     model.vision_model.encoder.layers[11].mlp.fc2.bias,
#     model.text_model.encoder.layers[11].mlp.fc2.weight,
#     model.text_model.encoder.layers[11].mlp.fc2.bias,
#     model.logit_scale
#     ]
    
# #     for param in model.parameters():
# #         param.requires_grad = False

# #     for param in parameters_to_train:
# #         param.requires_grad = True

#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(parameters_to_train, lr=5e-5)
# #     optimizer = torch.optim.AdamW(parameters_to_train, lr=5e-5, weight_decay=5e-5)
# #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    
#     epochs = 100
#     for epoch in range(epochs):
#         train_loss = train(model, train_dataloader, optimizer, criterion, device)
#         val_loss, val_accuracy = validate(model, val_dataloader, criterion, device)
#         print(f"Epoch {epoch+1}/{epochs}")
#         print(f"Train Loss: {train_loss:.4f}")
#         print(f"Validation Loss: {val_loss:.4f}")
#         print(f"Validation Accuracy: {val_accuracy:.4f}")
    
# if __name__ == '__main__':
#     main()


#----------------------------------------------------------------------

# 使用 clip-vit-base-patch32 模型

import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

def load_images_and_labels(images_dir):
    images = []
    labels = []
    label_dict = {}
    idx = 0
    
    for image_set in os.listdir(images_dir):
        set_dir = os.path.join(images_dir, image_set)
        cnt = 0
        for image_name in os.listdir(set_dir):
            if cnt == 20:
                break
            cnt += 1
            img_path = os.path.join(set_dir, image_name)
            label = set_dir[32:]
#             label = set_dir.split('/')[-1]
            if label not in label_dict:
                label_dict[label] = idx
                print(label)
                idx += 1
            images.append(img_path)
            labels.append(label_dict[label])

    return images, labels, label_dict

def predict(model, processor, image_path, device):
    image = Image.open(image_path).convert("RGB")
    image_input = processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        image_features = model.get_image_features(image_input)
    return image_features

def main():
    images_dir = "datasets/CUB_200_2011/CUB_200_2011/images"
#     images_dir = "datasets/StanfordDogs"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 使用 Hugging Face 的 CLIP 模型和处理器
    model_name = "./clip-vit-base-patch32"
#     model_name = "./CLIP-ViT-g-14-laion2B-s12B-b42K"
    print(f'Loading Model {model_name}, wait for a minute.')
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    print("Model Done")
    
    images, labels, label_dict = load_images_and_labels(images_dir)
    text_inputs = processor(text=[f"a photo of a {label}" for label in label_dict.keys()], return_tensors="pt", padding=True).input_ids.to(device)
    
    correct_predictions = 0
    total_images = len(images)
    
    with torch.no_grad():
        text_features = model.get_text_features(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    for i, (image_path, true_label) in enumerate(zip(images, labels)):
        image_features = predict(model, processor, image_path, device)
        
        # 对图像和文本的特征向量进行归一化处理
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # 使用内积计算图像特征和文本特征之间的相似度。然后应用 softmax 函数将相似度转换为概率分布
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        predicted_label_idx = similarity.argmax().item()
        
        if predicted_label_idx == true_label:
            correct_predictions += 1
        
        # 每处理100张图像后打印一次当前的准确率
        if (i + 1) % 50 == 0:
            current_accuracy = correct_predictions / (i + 1)
            print(f"Processed {i + 1}/{total_images} images, Current Accuracy: {current_accuracy:.4f}")
    
    # 最终准确率
    final_accuracy = correct_predictions / total_images
    print(f"Final Accuracy: {final_accuracy:.4f}")

if __name__ == '__main__':
    main()

    
#-----------------------------------------------------------------------


# open_clip


# import os
# import torch
# from PIL import Image
# from transformers import CLIPProcessor, CLIPModel
# import open_clip

# def load_images_and_labels(images_dir):
#     images = []
#     labels = []
#     label_dict = {}
#     idx = 0
    
#     for image_set in os.listdir(images_dir):
#         set_dir = images_dir + "/" + image_set
#         cnt = 0
#         for image_name in os.listdir(set_dir):
#             if cnt == 20:
#                 break
#             cnt += 1
#             img_path = set_dir + "/" + image_name
#             label = set_dir.split('.')[-1]
#             if label not in label_dict:
#                 label_dict[label] = idx
#                 idx += 1
#             images.append(img_path)
#             labels.append(label_dict[label])

#     return images, labels, label_dict

# def predict(model, preprocess, image_path, device):
#     image = Image.open(image_path).convert("RGB")
#     image_input = preprocess(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         image_features = model.encode_image(image_input)
#     return image_features

# def main():
#     available_models = open_clip.list_pretrained()
#     print("Available CLIP models:")
#     for model in available_models:
#         print(model)

#     images_dir = "datasets/CUB_200_2011/CUB_200_2011/images"
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     model_name = "ViT-H-14-378-quickgelu"
#     pretrained_dataset = "laion2b_s34b_b79k"
#     print(f'Loading Model {model_name} with pretrained dataset {pretrained_dataset}, wait for a minute.')
#     model, preprocess = open_clip.create_model_and_transforms(model_name)
#     model = model.to(device)
#     print("Model Done")
    
#     images, labels, label_dict = load_images_and_labels(images_dir)
#     tokenizer = open_clip.get_tokenizer(model_name)
#     text_inputs = tokenizer([f"a photo of a {label}" for label in label_dict.keys()]).to(device)
    
#     correct_predictions = 0
#     total_images = len(images)
    
#     with torch.no_grad():
#         text_features = model.encode_text(text_inputs)
#         text_features /= text_features.norm(dim=-1, keepdim=True)
    
#     for i, (image_path, true_label) in enumerate(zip(images, labels)):
#         image_features = predict(model, preprocess, image_path, device)
        
#         # 对图像和文本的特征向量进行归一化处理
#         image_features /= image_features.norm(dim=-1, keepdim=True)
        
#         # 使用内积计算图像特征和文本特征之间的相似度。然后应用 softmax 函数将相似度转换为概率分布
#         similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
#         predicted_label_idx = similarity.argmax().item()
        
#         if predicted_label_idx == true_label:
#             correct_predictions += 1
        
#         # 每处理50张图像后打印一次当前的准确率
#         if (i + 1) % 50 == 0:
#             current_accuracy = correct_predictions / (i + 1)
#             print(f"Processed {i + 1}/{total_images} images, Current Accuracy: {current_accuracy:.4f}")
    
#     # 最终准确率
#     final_accuracy = correct_predictions / total_images
#     print(f"Final Accuracy: {final_accuracy:.4f}")

# if __name__ == '__main__':
#     main()
