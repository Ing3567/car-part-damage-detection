FROM python:3.11-slim
WORKDIR /app

# lib ที่จำเป็นกับ OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

# อัปเกรด pip ให้ใหม่สุด (ลดปัญหารุ่น wheel)
RUN python -m pip install --upgrade pip

# ติดตั้ง PyTorch/torchvision (CPU wheels) จาก index ของ PyTorch
RUN pip install --no-cache-dir \
    torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cpu

# ติดตั้งไลบรารีที่เหลือ
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ด FastAPI
COPY routing.py ./routing.py

EXPOSE 8000
CMD ["uvicorn", "routing:app", "--host", "0.0.0.0", "--port", "8000"]
