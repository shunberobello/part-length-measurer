
import streamlit as st
import numpy as np
from PIL import Image
import cv2
from skimage import io, color, segmentation, feature, filters
from skimage.future import graph

def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

# タイトルを設定
st.title('物体の長さ計測アプリ')

# 画像をアップロード
uploaded_file = st.file_uploader("画像ファイルをアップロードしてください (jpg, png)", type=['jpg', 'png'])

if uploaded_file is not None:
    # アップロードされた画像を表示
    st.image(uploaded_file, caption='アップロードされた画像', use_column_width=True)
    
    # Convert uploaded file to an array
    image = np.array(Image.open(uploaded_file))
    
    # トリムした画像
    trimmed_image = image[1500:2500, 800:3500]
    st.image(trimmed_image, caption='トリムした画像', use_column_width=True)
    
    # グレースケールに変換
    gray_image = cv2.cvtColor(trimmed_image, cv2.COLOR_BGR2GRAY)
    
    # エッジ検出
    edge_image = detect_edges(trimmed_image)
    
    # Display the edge detected image
    st.image(edge_image, caption='エッジ', use_column_width=True, channels="GRAY")
