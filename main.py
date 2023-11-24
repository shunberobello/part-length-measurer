import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import time
import cv2
from skimage import io, color, segmentation, feature, filters
from skimage import graph
from skimage.measure import regionprops, label, regionprops_table
import matplotlib.pyplot as plt
import subprocess
import os

# 画像アップロード


def upload_image():
    return st.file_uploader("画像ファイルをアップロードしてください (jpg, png)", type=['jpg', 'png'])


# """
# アップロードされたファイルから画像を読み込み、トリムして返します。

# :param uploaded_file: アップロードされたファイル。
# :return: トリムされた画像。
# """
def read_and_trim_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    trimmed_image = image[1500:2500, 800:3500]
    return trimmed_image


# """
# GrabCutアルゴリズムを使用して前景抽出を行います。

# :param I_trimmed: トリムされた入力画像。
# :param mask_path: 初期マスク画像へのパス。
# :return: セグメンテーションされた画像。
# """
def apply_grabcut(I_trimmed, mask_path='grabcut_mask.png'):

    # 前景抽出のためのマスクの準備
    mask = cv2.imread(mask_path, 0)

    # maskをI_trimmedのサイズにリサイズ
    mask = cv2.resize(
        mask, (I_trimmed.shape[1], I_trimmed.shape[0]), interpolation=cv2.INTER_NEAREST)

    # I_trimmedの色空間をRGBに変換
    img = cv2.cvtColor(I_trimmed, cv2.COLOR_BGR2RGB)

    # GrabCutのためのモデルの初期化
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)

    # GrabCutの適用
    cv2.grabCut(img, mask, None, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)

    # 最終的なマスクの作成
    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')

    # セグメンテーションされたイメージの取得
    segmented = cv2.bitwise_and(img, img, mask=mask2)

    return segmented

# """
# エッジ検出を行い、フィルタリングして結果の画像を返します。

# :param I_trimmed: トリムされた入力画像。
# :param segmented: セグメンテーションされた画像。
# :param min_area: 領域の最小面積。
# :return: フィルタリング後のエッジが描画された画像、フィルタリングされたマスク。
# """


def detect_edges(I_trimmed, segmented, min_area=50):
    # グレースケールに変換
    image_gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)

    # 外側と内側の領域のマスクを作成
    _, outer_mask = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY)
    _, inner_mask = cv2.threshold(image_gray, 254, 255, cv2.THRESH_BINARY_INV)

    # # オープニング処理を適用してノイズや小さな点を除去
    # kernel = np.ones((3, 3), np.uint8)
    # outer_mask = cv2.morphologyEx(outer_mask, cv2.MORPH_OPEN, kernel)
    # inner_mask = cv2.morphologyEx(inner_mask, cv2.MORPH_OPEN, kernel)

    # エッジ検出を行う
    edges_outer = cv2.Canny(outer_mask, 100, 200)
    edges_inner = cv2.Canny(inner_mask, 100, 200)

    # 2つのエッジマップを結合
    edges_combined = cv2.bitwise_or(edges_outer, edges_inner)

    # 連結成分の分析を行い、各領域の統計情報を取得
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        edges_combined, connectivity=8)

    # 面積が一定値以上の領域のみを残す
    filtered_mask = np.zeros_like(edges_combined)
    for i in range(1, num_labels):  # 0は背景のためスキップ
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            filtered_mask[labels == i] = 255

    # 膨張処理でエッジを太くする
    kernel = np.ones((3, 3), np.uint8)  # このカーネルのサイズで太さが決まります。必要に応じて調整してください。
    dilated_mask = cv2.dilate(filtered_mask, kernel, iterations=1)

    # フィルタリング後のエッジを元の画像に描画
    result_image = I_trimmed.copy()
    result_image[dilated_mask == 255] = [255, 0, 0]

    return result_image, filtered_mask
  
# def detect_ellipses(filtered_mask, min_area=100, min_aspect_ratio=0.6, max_aspect_ratio=1.4):
    # 輪郭を検出
    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 輪郭ごとに楕円をフィットさせる
    ellipse_data = []
    for cnt in contours:
        if len(cnt) >= 5:  # fitEllipseは少なくとも5点必要
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (MA, ma), angle = ellipse

            # 楕円の面積
            area = np.pi * (MA / 2) * (ma / 2)

            # 楕円のアスペクト比
            aspect_ratio = max(MA, ma) / min(MA, ma) if min(MA, ma) > 0 else 0

            # 楕円の面積とアスペクト比に基づいてフィルタリング
            if area >= min_area and min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                ellipse_info = {
                    'ellipse': ellipse,
                    'area': area,
                    'aspect_ratio': aspect_ratio
                }
                ellipse_data.append(ellipse_info)
    
    # 面積が小さい順に楕円をソート
    ellipse_data.sort(key=lambda x: x['area'])
    
    # 面積が小さい上位3つの楕円を取得
    selected_ellipses = ellipse_data[:3]

    return [e['ellipse'] for e in selected_ellipses]


def detect_ellipses(filtered_mask, min_area=100, min_aspect_ratio=0.6, max_aspect_ratio=1.4):
    # 輪郭を検出
    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 輪郭ごとに楕円をフィットさせる
    ellipse_data = []
    for cnt in contours:
        if len(cnt) >= 5:  # fitEllipseは少なくとも5点必要
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (MA, ma), angle = ellipse

            # 楕円の面積
            area = np.pi * (MA / 2) * (ma / 2)

            # 楕円のアスペクト比
            aspect_ratio = max(MA, ma) / min(MA, ma) if min(MA, ma) > 0 else 0

            # 楕円の面積とアスペクト比に基づいてフィルタリング
            if area >= min_area and min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                ellipse_info = {
                    'ellipse': ellipse,
                    'area': area,
                    'aspect_ratio': aspect_ratio
                }
                ellipse_data.append(ellipse_info)
    
    # 面積が小さい順に楕円をソート
    ellipse_data.sort(key=lambda x: x['area'])
    
    # 面積が小さい上位3つの楕円を取得
    selected_ellipses = ellipse_data[:3]

    # 完全な楕円情報を返す
    return selected_ellipses
  
def draw_max_width_box(image, mask):
    # 輪郭を検出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 最大幅とそのバウンディングボックスを初期化
    max_width = 0
    max_width_box = None
    
    # 各輪郭のバウンディングボックスを計算し、最大のものを見つける
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width > max_width:
            max_width = width
            max_width_box = (x, y, width, height)
    
    # 最大幅のバウンディングボックスを描画
    if max_width_box:
        x, y, width, height = max_width_box
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        # 幅の数値を描画
        text = f"Width: {width}"
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
     # 画像と最大幅を返す
    return image, max_width

# この関数を使用して楕円を検出し、結果を使用するコード...

# メイン


def main():
    st.title('物体の長さ計測アプリ')

    # ---------------------------------------------
    # 画像アップロード
    # ---------------------------------------------
    uploaded_file = upload_image()
    if uploaded_file is None:
        return
    st.image(uploaded_file, caption='アップロードされた画像', use_column_width=True)

    # 計測ボタンを押されるまで待つ
    if not st.button('計測'):
        return

    # ---------------------------------------------
    # トリミング
    # ---------------------------------------------
    I_trimmed = read_and_trim_image(uploaded_file)
    st.image(I_trimmed, caption='トリム画像', use_column_width=True)

    # ---------------------------------------------
    # 前景抽出
    # ---------------------------------------------
    segmented = apply_grabcut(I_trimmed, 'grabcut_mask.png')

    # ---------------------------------------------
    # エッジ抽出
    # ---------------------------------------------
    result_image2, filtered_mask = detect_edges(I_trimmed, segmented)
    st.image(result_image2, caption='エッジ検出', use_column_width=True)
    
    # ------------------------------------------------------
    # 円の認識
    # ------------------------------------------------------
    result_image = segmented.copy()  # 描画用の画像をコピー

    # クロージング処理を適用して小さな穴を埋める
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)

    # すべての輪郭を検出
    contours, _ = cv2.findContours(
        closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 最大の輪郭を取得
    max_contour = max(contours, key=cv2.contourArea)

    # 最大の輪郭のみを描画したマスクを作成
    outer_edge_mask = np.zeros_like(filtered_mask)
    cv2.drawContours(outer_edge_mask, [max_contour], -1, 255, thickness=1)

    # マスクを表示
    # st.image(outer_edge_mask, caption='Outer Edge Mask', use_column_width=True)

    # 外周エッジ上の座標を取得
    y, x = np.where(outer_edge_mask == 255)
    edge_points = np.column_stack((x, y))
    print("Result Image Shape: ", result_image.shape)
    print("Outer Edge Mask Shape: ", outer_edge_mask.shape)

    # edge_pointsの座標に赤い点を描画
    outer_edge_image = I_trimmed.copy()  # 描画用の画像をコピー
    for point in edge_points:
        cv2.circle(outer_edge_image, tuple(point), 1, (255, 0, 0), -1)

    # 結果の表示
    # st.image(outer_edge_image, caption='Edge Points', use_column_width=True)

    # ------------------------------------------------------
    # 楕円の認識
    # ------------------------------------------------------
    # 楕円を検出してそのデータを取得
    ellipses_data = detect_ellipses(filtered_mask)

    # 最大の楕円を見つける
    max_ellipse_data = max(ellipses_data, key=lambda e: e['area'])
    max_ellipse = max_ellipse_data['ellipse']
    max_area = max_ellipse_data['area']
    # reference_aspect_ratio = max_ellipse_data['aspect_ratio']
    reference_aspect_ratio = 1
    max_long_axis = max(max_ellipse[1])  # 長軸の長さ
    max_short_axis = min(max_ellipse[1])  # 短軸の長さ

    # 結果を描画するための画像をコピー
    result_image = I_trimmed.copy()
    result_image_corrected = I_trimmed.copy()

    # 校正された距離を保存するリスト
    distances_data = []

    # 楕円ごとに距離の校正を行う
    for ellipse_data in ellipses_data:
        ellipse, area, aspect_ratio = ellipse_data['ellipse'], ellipse_data['area'], ellipse_data['aspect_ratio']

        # 楕円の中心点を取得
        (x_center, y_center), (MA, ma), angle = ellipse
        long_axis = max(MA, ma)
        short_axis = min(MA, ma)
        # print('max_long_axis：' + str(max_long_axis))
        # print('max_short_axis：' + str(max_short_axis))
        # print('MA：' + str(long_axis))
        # print('ma：' + str(short_axis))
        # print('area：' + str(area))
        center = (int(x_center), int(y_center))

        # 中心からエッジへの最短距離を測定
        distances = np.linalg.norm(edge_points - np.array(center), axis=1)
        min_distance = np.min(distances)
        print('最短距離（校正前）：' + str(min_distance))

        # 校正係数を計算
        if area < max_ellipse_data['area']:  # 最大楕円以外の場合に校正を行う
          
            # # 校正係数 = ルート (最大円の長軸 / 円の長軸） * (最大円の短軸 / 円の短軸）
            correction_factor = np.sqrt((max_long_axis / long_axis) * (max_short_axis / short_axis))
            
            # # 校正係数 = ルート (最大円の面積 / 円の面積） * (基準アスペクト比=1 / 円のアスペクト比）
            # correction_factor = np.sqrt((max_area / area) * (reference_aspect_ratio / aspect_ratio))
            
            corrected_distance = min_distance * correction_factor
            # print('補正係数：' + str(correction_factor))
        else:
            corrected_distance = min_distance  # 最大楕円の場合は校正不要

        # 校正された距離をリストに追加
        distances_data.append({
          'ピクセル（校正前）': min_distance,
          'mm（校正前）': None,  # まだ計算しない
          'ピクセル（校正後）': corrected_distance,
          'mm（校正後）': None,  # まだ計算しない
        })
        print('最短距離（校正後）：' + str(corrected_distance))

        # 中心からエッジへの線を描画
        min_index = np.argmin(distances)
        edge_point = tuple(edge_points[min_index])
        cv2.line(result_image, center, edge_point, (0, 255, 0), 2)
        cv2.line(result_image_corrected, center, edge_point, (0, 255, 0), 2)

        # 校正された距離を画像上に表示
        text_position = (edge_point[0], edge_point[1] - 10)
        cv2.putText(result_image, f"{min_distance:.2f}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(result_image_corrected, f"{corrected_distance:.2f}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # 楕円を描画
        cv2.ellipse(result_image, ellipse, (0, 255, 0), 2)
        cv2.ellipse(result_image_corrected, ellipse, (0, 255, 0), 2)

    # 結果の表示
    st.image(result_image, caption='円の中心からエッジへの最短距離（校正前）', use_column_width=True)
    st.image(result_image_corrected, caption='円の中心からエッジへの最短距離（校正後）', use_column_width=True)
    

    # ------------------------------------------------------
    # 幅の計測
    # ------------------------------------------------------
    width_image = I_trimmed.copy()  # 描画用の画像をコピー
    image_with_box, max_width_pixels = draw_max_width_box(width_image, filtered_mask)
    st.image(image_with_box, caption='物体の最大幅', use_column_width=True)
    
    # ------------------------------------------------------
    # 結果表示
    # ------------------------------------------------------
    # 1ピクセルあたりの実際の長さの計算
    length = 157
    pixel_length_mm = length / max_width_pixels  # 実際の幅は157mm
    st.write(f"物体の最大幅（手動で計測した値）:　{length} mm")
    st.write(f"1ピクセルあたりの長さ（{length} / {max_width_pixels}）:　{pixel_length_mm:.4f} mm")
    
    # 中心からエッジへの実際の長さの計算
    for data in distances_data:
        data['mm（校正前）'] = data['ピクセル（校正前）'] * pixel_length_mm
        data['mm（校正後）'] = data['ピクセル（校正後）'] * pixel_length_mm

    # データをDataFrameに変換して表示
    distances_df = pd.DataFrame(distances_data)
    
    st.write("中心からエッジへの最短距離")
    st.dataframe(distances_df)

if __name__ == "__main__":
    main()
