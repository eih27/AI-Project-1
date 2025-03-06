import socket
import time
import array
import gc
import network
from machine import Pin
from camera import Camera, PixelFormat, FrameSize
# Import necessary image preprocessing functions
from image_preprocessing import resize_96x96_to_32x32_averaged_and_threshold, strip_bmp_header

import emlearn_cnn_int8 as emlearn_cnn  # Import CNN model for inference

# 📡 WiFi Configuration
WIFI_SSID = "EIH"  # Update WiFi name
WIFI_PASSWORD = "228467eh"  # Update WiFi password

# 📷 Camera Configuration Parameters
CAMERA_CONFIG = {
    "data_pins": [15, 17, 18, 16, 14, 12, 11, 48],
    "vsync_pin": 38,
    "href_pin": 47,
    "sda_pin": 40,
    "scl_pin": 39,
    "pclk_pin": 13,
    "xclk_pin": 10,
    "xclk_freq": 20000000,  # Frequency for XCLK
    "powerdown_pin": -1,
    "reset_pin": -1,
    "frame_size": FrameSize.R96X96,
    "pixel_format": PixelFormat.GRAYSCALE
}

# 🧠 Load Pre-trained CNN Model
MODEL_FILE = "model.tmdl"
CLASS_LABELS = ["Rock", "Paper", "Scissors"]

def connect_to_wifi():
    """Establish a connection to the specified WiFi network."""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)

    print("🌍 Connecting to WiFi...")
    while not wlan.isconnected():
        time.sleep(1)
    
    print(f"✅ WiFi Connected! IP Address: {wlan.ifconfig()[0]}")

def get_max_index(arr):
    """Find the index of the highest value in an array."""
    return max(range(len(arr)), key=lambda i: arr[i])

def log_debug_info(data, description, sample_size=10):
    """Log debug information about a given data object."""
    print(f"🔍 DEBUG: {description} → Type: {type(data)}, Size: {len(data)}, Sample: {data[:sample_size]}")

def start_video_stream():
    """Initialize the camera, load the CNN model, and start streaming classification data."""
    print("📷 Setting up Camera...")
    cam = Camera(**CAMERA_CONFIG)
    cam.init()
    cam.set_bmp_out(True)
    print("✅ Camera Ready!")

    # Load pre-trained model
    print("🟢 Loading CNN model from:", MODEL_FILE)
    with open(MODEL_FILE, "rb") as f:
        model_bytes = array.array("B", f.read())
        model = emlearn_cnn.new(model_bytes)
    print("✅ Model loaded successfully!")

    # Open network socket for client communication
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", 8080))
    server.listen(1)
    print("🚀 Awaiting client connection on port 8080...")

    client, addr = server.accept()
    print(f"✅ Client connected from: {addr}")

    # 📊 Classification storage
    probabilities = array.array("f", (0.0 for _ in range(len(CLASS_LABELS))))

    try:
        while True:
            # 📷 Capture Image
            image_data = cam.capture()
            log_debug_info(image_data, "Raw Captured Image")

            # Convert to bytearray if necessary
            image_data = bytearray(image_data)
            log_debug_info(image_data, "Converted Image Data")

            # 📏 Resize and Apply Thresholding
            resized_image = resize_96x96_to_32x32_averaged_and_threshold(image_data, threshold=128)
            log_debug_info(resized_image, "Resized Image (32x32)")

            # ✂️ Remove BMP Header before inference
            processed_image = strip_bmp_header(resized_image)
            log_debug_info(processed_image, "Processed Image Without Header")

            # Convert bytearray to array for model processing
            image_array = array.array('B', processed_image)
            log_debug_info(image_array, "Final Image Data for CNN Model")

            # 🔮 Perform Classification using the CNN Model
            model.run(image_array, probabilities)
            predicted_index = get_max_index(probabilities)
            prediction = CLASS_LABELS[predicted_index]

            # 📡 Send processed frame + classification result to client
            client.sendall(b"FRAME_START" + resized_image + b"FRAME_END")
            client.sendall(f"PREDICTION: {prediction}\n".encode())

            print(f"🖼️ Prediction: {prediction}")
            
            gc.collect()  # Free up memory
    
    except Exception as err:
        print(f"❌ Error occurred: {err}")
        client.close()
        server.close()
        cam.deinit()

if __name__ == "__main__":
    connect_to_wifi()
    start_video_stream()
