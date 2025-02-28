import cv2
import numpy as np
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt

DEVICE = 'cpu'
model = FastSAM('FastSAM/FastSAM-x.pt')


def transfer_illumination(source_img, target_img):
    gray_source = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    hist_source, _ = np.histogram(gray_source.flatten(), bins=256, range=[0, 256])
    hist_target, _ = np.histogram(gray_target.flatten(), bins=256, range=[0, 256])

    mean_source = np.mean(gray_source)
    mean_target = np.mean(gray_target)

    illumination_difference = mean_source - mean_target
    corrected_target = cv2.add(target_img, illumination_difference)

    return corrected_target


def replace_sky(in_img, sky_img):
    print("Start_process")
    everything_results = model(in_img, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9, )
    prompt_process = FastSAMPrompt(in_img, everything_results, device=DEVICE)

    ann = prompt_process.everything_prompt()
    ann = prompt_process.text_prompt(text='all merged sky')

    mask_tensor = ann[0].masks.data[0]
    mask = mask_tensor.cpu().numpy().astype(np.uint8) * 255

    mask_inv = cv2.bitwise_not(mask)

    removed_bg = cv2.bitwise_and(in_img, in_img, mask=mask_inv)
    sky_region = cv2.bitwise_and(sky_img, sky_img, mask=mask)
    result_img = cv2.add(removed_bg, sky_region)
    transfer_illumination(sky_img, result_img)
    return result_img


image = cv2.imread(r"C:\Users\aginity\Desktop\SHIT_5_2\OPEN_CV\sky_segmmentation\data\sky_1.jpg")
sky_image = cv2.imread(r"C:\Users\aginity\Desktop\SHIT_5_2\OPEN_CV\sky_segmmentation\data\bg_1.jpg")
res = replace_sky(image, sky_image)
cv2.imshow("Removed sky", res)
