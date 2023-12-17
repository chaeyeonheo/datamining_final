import requests
import io, sys
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image, ImageEnhance

# st.set_option('deprecation.showfileUploaderEncoding', False)

###############################################
import argparse
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont
import os

print(os.getcwd())
sys.path.append('./Grounded-Segment-Anything')



# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(Image):
    # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(Image, None)  # 3, h, w
    return Image, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > int(box_threshold)
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


# cfg
config_file = './Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py' # change the path of the model config file
grounded_checkpoint = './Grounded-Segment-Anything/groundingdino_swint_ogc.pth'  # change the path of the model
sam_version = "vit_b"
sam_checkpoint = './Grounded-Segment-Anything/sam_vit_b_01ec64.pth'
sam_hq_checkpoint = None
use_sam_hq = False
output_dir = "mask"
device = "cuda"


os.makedirs(output_dir, exist_ok=True)
model = load_model(config_file, grounded_checkpoint, device=device)
# make dir






############################################33######











# Upload an image and set some options for demo purposes
st.header("Zero-Shot Labeling")
img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
# realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
# box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
box_color ='#0000FF'
# aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
# contrast_level = st.sidebar.slider('Contrast level', min_value=0.5, max_value=3.5, value=1.0)
# brightness_level = st.sidebar.slider('Brightness level', min_value=0.5, max_value=3.5, value=1.0)
# sharpness_level = st.sidebar.slider('Sharpness level', min_value=0.5, max_value=3.5, value=1.0)
# aspect_dict = {
#     "1:1": (1, 1),
#     "16:9": (16, 9),
#     "4:3": (4, 3),
#     "2:3": (2, 3),
#     "Free": None
# }
# aspect_ratio = aspect_dict[aspect_choice]

def pil_to_binary(image, enc_format = "png"):
	"""Convert PIL Image to base64-encoded image"""
	buffer = io.BytesIO()
	image.save(buffer, format=enc_format)
	buffer.seek(0)
	return buffer

if img_file:
	img = Image.open(img_file)
	image_pil, image = load_image(img)
	st.image(img)


	text_prompt = f"dog, cat"
	text_prompt = st.text_input('Object_input',text_prompt)


	box_threshold = f"0.3"
	box_threshold = st.text_input('box_threshold',box_threshold)
	box_threshold = float(box_threshold)

	text_threshold = f"0.25"
	text_threshold = st.text_input('text_threshold',text_threshold)
	text_threshold = float(text_threshold)
	


	if st.button('Go!!'):
	# visualize raw image
		image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

		# run grounding dino model
		boxes_filt, pred_phrases = get_grounding_output(
			model, image, text_prompt, float(box_threshold), float(text_threshold), device=device)

		predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
		image = np.array(image_pil)
		image = image[:, :, ::-1].copy()
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		predictor.set_image(image)


		size = image_pil.size
		H, W = size[1], size[0]
		##############################
		for i in range(boxes_filt.size(0)):
			boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
			boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
			boxes_filt[i][2:] += boxes_filt[i][:2]

		boxes_filt = boxes_filt.cpu()
		transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
		##################################################3
		masks, _, _ = predictor.predict_torch(
			point_coords = None,
			point_labels = None,
			boxes = transformed_boxes.to(device),
			multimask_output = False,
		)


		plt.figure(figsize=(10, 10))
		plt.imshow(image)
		for mask in masks:
			show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
		for box, label in zip(boxes_filt, pred_phrases):
			show_box(box.numpy(), plt.gca(), label)

		plt.axis('off')
		plt.savefig(
			os.path.join(output_dir, "grounded_sam_output.jpg"),
			bbox_inches="tight", dpi=300, pad_inches=0.0
		)

		save_mask_data(output_dir, masks, boxes_filt, pred_phrases)


		output_image = Image.open(os.path.join(output_dir, "grounded_sam_output.jpg"))
		st.image(output_image)
		output_mask = Image.open(os.path.join(output_dir, "mask.jpg"))
		st.image(output_mask)



	# contr_enhancer = ImageEnhance.Contrast(img)
	# img = contr_enhancer.enhance(contrast_level)
	# bright_enhancer = ImageEnhance.Brightness(img)
	# img = bright_enhancer.enhance(brightness_level)
	# sharp_enhancer = ImageEnhance.Sharpness(img)
	# img = sharp_enhancer.enhance(sharpness_level)

	# if not realtime_update:
	# 	st.write("Double click to save crop")
	# Get a cropped image from the frontend
	# cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
	# 							aspect_ratio=aspect_ratio)
	# thumnail_img = cropped_img.copy()
	# Manipulate cropped image at will
	# st.write("Preview")
	# _ = thumnail_img.thumbnail((150,150))
	# st.image(thumnail_img)
	# if st.button('submit'):
	# 	# Convert the PIL image to base64 format
	# 	buffer= pil_to_binary(cropped_img)
 
	####bed####
		# POST the image data as json to the FastAPI server
	# url = "http://127.0.0.1:8000/predict"
	# file = {"file": buffer}
	# response = requests.post(url, files=file)

	# 	# Print the response or do something else...
	# grad_img = Image.open(response.json()['grad_img_path'])
	# st.image(grad_img)
	# st.code(response.text,'json')
	####bed####