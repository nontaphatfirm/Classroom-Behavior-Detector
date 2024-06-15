import gradio as gr
import cv2
import requests
import os

from ultralytics import YOLO

file_urls = [
    'https://www.dropbox.com/scl/fi/5i223blrvkarlczd6v175/3002177.jpg?rlkey=mcfhy20cpu0hltqdcy7jtohin&st=noqsik1v&dl=1',
    'https://www.dropbox.com/scl/fi/7u4e56zdz5i5gsbneo9jv/Screenshot-2024-06-12-161636.png?rlkey=0tuqjo51ptop3aa4rzissvmio&st=2oh7ko6z&dl=1',
    'https://www.dropbox.com/scl/fi/jdeana2t9lzec0czwvafu/1116839_Lesson_Hand_1280x720.mp4?rlkey=qbahgmbk3jjm2g6t0obl53uw0&st=w2jlo6oc&dl=1'
]

def download_file(url, save_name):
    if not os.path.exists(save_name):
        file = requests.get(url)
        open(save_name, 'wb').write(file.content)

for i, url in enumerate(file_urls):
    if 'mp4' in file_urls[i]:
        download_file(file_urls[i], f"video.mp4")
    else:
        download_file(file_urls[i], f"image_{i}.jpg")

model = YOLO('best.pt')
path  = [['image_0.jpg'], ['image_1.jpg']]
video_path = [['video.mp4']]

# Define a color map for different classes
color_map = {
    0: (209, 54, 40),  # Red
    1: (37, 194, 45),  # Green
    2: (34, 92, 240),  # Blue
    # Add more colors as needed for additional classes
}

def show_preds_image(image_path):
    if isinstance(image_path, dict):
        image_path = image_path['name']
    image = cv2.imread(image_path)
    outputs = model.predict(source=image_path)
    results = outputs[0]
    for i, det in enumerate(results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, det)
        cls = int(results.boxes.cls[i].item())
        label = results.names[cls]
        confidence = results.boxes.conf[i].item()
        color = color_map.get(cls, (255, 255, 255))  # Default to white if class not in color_map
        cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

inputs_image = [
    gr.components.Image(type="filepath", label="Input Image"),
]
outputs_image = [
    gr.components.Image(type="numpy", label="Output Image"),
]
interface_image = gr.Interface(
    fn=show_preds_image,
    inputs=inputs_image,
    outputs=outputs_image,
    title="Student-Behavior-Detector-in-Classroom",
    examples=path,
    cache_examples=False,
)

def show_preds_video(video_path):
    if isinstance(video_path, dict):
        video_path = video_path['name']
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_video_path = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            outputs = model.predict(source=frame)
            results = outputs[0]
            for i, det in enumerate(results.boxes.xyxy):
                x1, y1, x2, y2 = map(int, det)
                cls = int(results.boxes.cls[i].item())
                label = results.names[cls]
                confidence = results.boxes.conf[i].item()
                color = color_map.get(cls, (255, 255, 255))  # Default to white if class not in color_map
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            out.write(frame)
        else:
            break

    cap.release()
    out.release()
    return output_video_path

inputs_video = [
    gr.components.Video(label="Input Video"),
]
outputs_video = [
    gr.components.Video(label="Output Video"),
]
interface_video = gr.Interface(
    fn=show_preds_video,
    inputs=inputs_video,
    outputs=outputs_video,
    title="Student-Behavior-Detector-in-Classroom",
    examples=video_path,
    cache_examples=False,
)

gr.TabbedInterface(
    [interface_image, interface_video],
    tab_names=['Image inference', 'Video inference']
).queue().launch()
