import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
from config import data_files, output_video_name

def stitch_frames(image_folder, video_name):
    def key(img_name):
        num = int(img_name.split('.')[0])
        return num

    base, ext = os.path.splitext(video_name)
    if ext.lower() != ".mp4":
        video_name = base + ".mp4"

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, 30, (width,height))

    for image in tqdm(sorted(images, key=key), desc="Stitching Heatmap Video"):
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def generate_frames():
    # Load your image
    image = cv2.imread('base_image.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    h, w, _ = image.shape

    json_file_paths = data_files
    json_files = []

    for i in range(len(json_file_paths)):
        with open(json_file_paths[i]) as f:
            json_files.append(json.load(f))

    search_range = min([len(j) for j in json_files])

    # Create an empty "heatmap" array initialized with zeros
    heatmap = np.zeros((h, w), dtype=int)

    for i in tqdm(range(search_range), desc="Generating Heatmap"):
        locations = {}
        for j in json_files:
            detections = j[i]
            for detection in detections:
                key = list(detection.keys())[0]
                if key in locations.keys():
                    locations[key] = [
                        (locations[key][0] + detection[key][0]) // 2,
                        (locations[key][1] + detection[key][1]) // 2,
                    ]
                else:
                    locations[key] = detection[key]

                if locations[key][0] < 100:
                    locations[key][0] = 100
                elif locations[key][0] > 550:
                    locations[key][0] = 550
                if locations[key][1] < 50:
                    locations[key][1] = 50
                elif locations[key][1] > 550:
                    locations[key][1] = 550

        for id, box in locations.items():
            heatmap[box[1]-25:box[1]+25, box[0]-25:box[0]+25] += 1

        # Visualization
        plt.imshow(image, cmap='gray', interpolation='bilinear')
        plt.imshow(heatmap, cmap='jet', interpolation='bilinear', alpha=0.5, vmin=0, vmax=30)  # 0.5 alpha for transparency
        
        plt.savefig(f"./frames/{i*2}.jpg")
        plt.savefig(f"./frames/{i*2+1}.jpg")
        plt.close()

# generate_frames()
stitch_frames("./frames", output_video_name)