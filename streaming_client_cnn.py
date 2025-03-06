import cv2
import socket
import numpy as np
import struct

# 🎯 **ESP32 Server Configuration**
ESP32_IP = "172.20.10.5"  # Replace with your ESP32 IP
PORT = 8080  # Must match ESP32 server

# 🖥 **Initialize OpenCV Window**
cv2.namedWindow("ESP32 Camera", cv2.WINDOW_NORMAL)  # Resizable window
cv2.resizeWindow("ESP32 Camera", 800, 600)  # Set initial window size

# 🔗 **Connect to ESP32 Server**
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((ESP32_IP, PORT))

# 🛠 **Buffer for Incoming Data**
buffer = bytearray()

while True:
    try:
        # 📩 **Receive Incoming Data**
        chunk = sock.recv(4096)
        if not chunk:
            break
        buffer.extend(chunk)

        # 🔍 **Find BMP Header (BMP starts with 'BM')**
        start = buffer.find(b'BM')
        if start == -1:
            continue  # No BMP header found yet

        # 📏 **Extract BMP File Size from Header**
        if len(buffer) >= start + 6:
            file_size = struct.unpack("<I", buffer[start+2:start+6])[0]  # Read BMP size
        else:
            continue  # Wait for more data

        # ✅ **Check if Full BMP is Received**
        if len(buffer) < start + file_size:
            continue  # Keep receiving until we have the full BMP frame

        # ✂ **Extract BMP Frame**
        bmp_data = buffer[start:start + file_size]
        buffer = buffer[start + file_size:]  # Remove processed frame

        # 🔄 **Convert BMP to OpenCV Format**
        img = cv2.imdecode(np.frombuffer(bmp_data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        if img is None:
            print("❌ ERROR: BMP Frame Decoding Failed")
            continue

        # 🖼 **Resize & Display**
        img_resized = cv2.resize(img, (800, 600))
        cv2.imshow("ESP32 Camera", img_resized)

        # 🛑 **Exit on 'q' Press**
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except Exception as e:
        print(f"❌ Exception: {e}")
        break

# 🔄 **Cleanup**
sock.close()
cv2.destroyAllWindows()
