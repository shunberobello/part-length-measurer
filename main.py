import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import time
import cv2
from skimage import io, color, segmentation, feature, filters
from skimage import graph
from skimage.measure import regionprops, label;
import matplotlib.pyplot as plt
import subprocess
import os


def measure_object_length(image_path):
    """
    この関数は、提供された画像から物体の長さを計測することを目的としています。
    現在は、指定された結果の画像とダミーの計測データをそのまま返しています。
    """
    # ダミーの結果画像
    result_img_path = 'result.png'
    
    # 固定番号を持たないダミーの計測データ
    measurements = [259.0284, 260.1915, 244.7583]
    
    return result_img_path, measurements


def create_superpixels(image, n_segments=13619):
    # Convert the image to LAB color space
    image_lab = color.rgb2lab(image)
    
    # Use SLIC (Simple Linear Iterative Clustering) for superpixel segmentation
    segments = segmentation.slic(image_lab, n_segments=n_segments, compactness=30, sigma=1, start_label=1)
    
    return segments

def apply_graphcut(image, segments, foreground_mask, background_mask):
    g = graph.rag_mean_color(image, segments, mode='similarity')
    labels = graph.cut_normalized(segments, g, num_cuts=10)
    
    output = np.zeros_like(image, dtype=np.uint8)
    for region in regionprops(labels):
        if region.label in foreground_mask:
            for coord in region.coords:
                output[coord[0], coord[1]] = image[coord[0], coord[1]]
                
    return output

def detect_edges(image):
    gray = color.rgb2gray(image)
    edges = feature.canny(gray)
    return edges

foregroundInd = [189525, 191529, 193533, 195541, 199547, 199549, 201555, 205553, 205569, 207575, 213589, 215599, 225622, 225624, 227630, 229634, 238653, 242663, 244669, 248679, 254695, 260709, 262713, 266725, 270734, 282764, 296800, 300810, 304820, 306824, 311835, 311837, 319853, 319855, 327874, 329880, 335894, 341908, 343912, 351934, 357948, 363964, 373986, 377999, 387022, 397050, 397056, 397058, 397062, 397066, 397068, 397070, 399080, 405096, 409111, 411121, 417141, 418758, 419149, 419151, 419153, 419163, 419167, 421177, 421179, 423189, 423198, 423202, 423210, 423216, 423218, 425222, 425232, 427236, 427240, 429246, 429248, 431252, 433254, 439260, 447266, 449268, 459278, 466285, 468287, 470289, 476295, 478297, 480299, 487819, 488307, 490307, 495827, 500317, 502317, 514327, 518329, 528337, 536343, 547352, 555358, 561362, 563362, 583378, 605394, 611400, 615404, 634419, 638423, 641951, 646429, 663970, 666447, 682461, 691992, 694471, 698997, 699476, 700999, 705001, 709484, 723015, 727500, 751041, 782542, 786546, 788074, 798082, 798554, 802556, 818566, 824570, 830574, 838118, 844586, 853131, 859597, 861599, 873609, 875609, 877151, 895625, 905633, 917643, 928196, 936658, 938204, 962682, 966230, 984702, 996712, 1015275, 1021735, 1033745, 1035291, 1037747, 1053307, 1063771, 1090342, 1100806, 1116822, 1126832, 1140846, 1154858, 1163867, 1189891, 1203443, 1207907, 1229929, 1238474, 1250948, 1272970, 1294992, 1304533, 1330025, 1344039, 1345570, 1362057, 1386079, 1399089, 1403093, 1408629, 1431119, 1434655, 1449135, 1473692, 1474158, 1480164, 1497716, 1508192, 1509728, 1516200, 1531748, 1542226, 1546763, 1570785, 1582797, 1590805, 1597279, 1609291, 1620833, 1643856, 1646328, 1668350, 1684366, 1687898, 1702913, 1715397, 1718929, 1758967, 1761443, 1769451, 1787996, 1790474, 1802010, 1802488, 1826512, 1834522, 1857545, 1875081, 1883571, 1899589, 1907113, 1907597, 1913119, 1925615, 1934140, 1938144, 1938628, 1948638, 1956649, 1966174, 1966659, 1978190, 1988681, 2000216, 2004697, 2009227, 2009702, 2017239, 2027720, 2035728, 2037263, 2049280, 2061756, 2063296, 2065298, 2092791, 2096335, 2102803, 2114817, 2120361, 2128371, 2134841, 2146391, 2154399, 2177428, 2183896, 2197458, 2205470, 2211480, 2215486, 2221938, 2225506, 2231950, 2237532, 2250560, 2264590, 2268598, 2273001, 2276616, 2282628, 2294652, 2297029, 2305041, 2312688, 2326066, 2327716, 2329720, 2335732, 2345756, 2347760, 2350094, 2363796, 2368114, 2383836, 2389848, 2393857, 2403155, 2404878, 2411165, 2420914, 2427185, 2428934, 2447211, 2465027, 2478060, 2482068, 2484074, 2486263, 2488082, 2490267, 2494096, 2498277, 2500108, 2502114, 2506123, 2510291, 2512135, 2516145, 2522159, 2526169, 2530179, 2542205, 2544321, 2548219, 2555234, 2557240, 2565257, 2567340, 2569267, 2569340, 2571271, 2573342, 2575281, 2575341, 2579293, 2579343, 2581301, 2581343, 2583305, 2587317, 2587325, 2587327, 2587329, 2587333];
backgroundInd = [49334, 49340, 51312, 51314, 51316, 51354, 51358, 55373, 55375, 57314, 57379, 59314, 59397, 59401, 61411, 63423, 69322, 69445, 71322, 71453, 73457, 78471, 84487, 86491, 96513, 98345, 102527, 106351, 108543, 116559, 120569, 122365, 122575, 128589, 138610, 140614, 150634, 152638, 155390, 155643, 163657, 165663, 173681, 175405, 179691, 187413, 191713, 199727, 209745, 215437, 215756, 217760, 219441, 227778, 238797, 242807, 246817, 250470, 250825, 256837, 260845, 264853, 268863, 276883, 280496, 282900, 282902, 284910, 284958, 288922, 288926, 290930, 290936, 290956, 290958, 290960, 290962, 292942, 292948, 292954, 292956, 292964, 309525, 322046, 325539, 336090, 340104, 342115, 343555, 346123, 348133, 360171, 360173, 362177, 362179, 364189, 366193, 366195, 368207, 368209, 368211, 368224, 368234, 368236, 368240, 368242, 368244, 368254, 368256, 368258, 368260, 368266, 368270, 368276, 370280, 376290, 384302, 392598, 399324, 407336, 408614, 426630, 433364, 445378, 465665, 480413, 509705, 520442, 530448, 542734, 545455, 546738, 551046, 560750, 571082, 573465, 579098, 579469, 581100, 581471, 585108, 593475, 594782, 608794, 609485, 617491, 626498, 634502, 636504, 655835, 687863, 700876, 709567, 720894, 723579, 729585, 737591, 747599, 748920, 786629, 789959, 805973, 832669, 859024, 863696, 873706, 881046, 905734, 911074, 919744, 942105, 946769, 978797, 992811, 994157, 1007168, 1039200, 1045860, 1051210, 1067226, 1088901, 1102261, 1104917, 1122933, 1140297, 1161318, 1196002, 1210016, 1217372, 1228034, 1229384, 1248403, 1275081, 1288443, 1301105, 1302457, 1341493, 1345773, 1345781, 1345785, 1347797, 1349799, 1350152, 1371523, 1376176, 1386184, 1411207, 1423217, 1450598, 1462610, 1463253, 1470260, 1481627, 1492282, 1493639, 1522310, 1531671, 1549337, 1554692, 1568706, 1573361, 1610746, 1615401, 1630414, 1636420, 1641777, 1652436, 1682464, 1687821, 1705485, 1712846, 1739519, 1755535, 1760894, 1780562, 1812594, 1832614, 1833967, 1843975, 1883665, 1925054, 1940069, 1956085, 1972756, 1986770, 2021148, 2053472, 2053494, 2055506, 2055508, 2057851, 2063857, 2065859, 2083881, 2090890, 2102231, 2116247, 2130936, 2138273, 2140946, 2158299, 2163973, 2177324, 2196007, 2209377, 2213383, 2221397, 2227405, 2245058, 2252444, 2253068, 2254448, 2260460, 2261076, 2264466, 2267084, 2268472, 2274482, 2279098, 2297120, 2308539, 2328157, 2339588, 2340171, 2347600, 2350183, 2351606, 2355614, 2374209, 2375646, 2387668, 2391676, 2415254, 2416721, 2422734, 2434754, 2445286, 2448780, 2462804, 2463306, 2469315, 2481839, 2485847, 2492342, 2499873, 2503883, 2514370, 2524384, 2531940, 2536398, 2541960, 2547974, 2575033, 2579046, 2579451, 2583056, 2589078, 2591084, 2591467, 2595094, 2601479, 2605122, 2609489, 2611134, 2619501, 2623163, 2623507, 2628512, 2634188, 2634518, 2636520, 2638196, 2638522, 2642208, 2646526, 2648526, 2650228, 2650526, 2652526, 2658528, 2662520, 2662522, 2664487, 2664493, 2664495, 2664497, 2664503, 2664507, 2664514, 2666469, 2666471, 2666475, 2666481, 2666487, 2668289, 2668445, 2668459, 2668463, 2670424, 2670428, 2670434, 2670441, 2672416, 2672418, 2672420, 2672422, 2674410, 2674412, 2674416, 2674418, 2676380, 2676382, 2676384, 2676394, 2676404, 2676406, 2676410, 2678335, 2678343, 2678349, 2678357, 2678365, 2678372, 2678374, 2678376];

# タイトルを設定
st.title('物体の長さ計測アプリ')



# 画像をアップロード
uploaded_file = st.file_uploader("画像ファイルをアップロードしてください (jpg, png)", type=['jpg', 'png'])

if uploaded_file is not None:
    # アップロードされた画像を表示
    st.image(uploaded_file, caption='アップロードされた画像', use_column_width=True)
    
    # 計測ボタン
    if st.button('計測'):
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        I = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  
        
        # #画像のサイズ縮小
        I_trimmed = I[1500:2500, 800:3500]
        st.image(I_trimmed, caption='トリム画像', use_column_width=True)
        
        # ------------------------------------------------------
        # 前景抽出
        # ------------------------------------------------------
        # 前景抽出のためのマスクの準備
        mask = cv2.imread('grabcut_mask.png', 0)
        
        # I_trimmedをmaskのサイズにリサイズ
        I_resized = cv2.resize(I_trimmed, (mask.shape[1], mask.shape[0]))
        
        img = cv2.cvtColor(I_resized, cv2.COLOR_BGR2RGB)   
        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)
        cv2.grabCut(img, mask, None, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')
        segmented = cv2.bitwise_and(img, img, mask=mask2)
        # st.image(output, caption='Segmented Image', use_column_width=True)
        
        # ------------------------------------------------------
        # エッジ検出
        # ------------------------------------------------------
        image_gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)

        # 外側と内側の領域のマスクを作成
        _, outer_mask = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY)
        _, inner_mask = cv2.threshold(image_gray, 254, 255, cv2.THRESH_BINARY_INV)
        
        # オープニング処理を適用してノイズや小さな点を除去
        kernel = np.ones((3,3), np.uint8)
        outer_mask = cv2.morphologyEx(outer_mask, cv2.MORPH_OPEN, kernel)
        inner_mask = cv2.morphologyEx(inner_mask, cv2.MORPH_OPEN, kernel)

        # Cannyを使用してエッジ検出を行う
        edges_outer = cv2.Canny(outer_mask, 100, 200)
        edges_inner = cv2.Canny(inner_mask, 100, 200)

        # 2つのエッジマップを結合
        edges_combined = cv2.bitwise_or(edges_outer, edges_inner)

        # エッジを細くするために収縮を使用
        # kernel = np.ones((1,1), np.uint8)
        # thin_edges = cv2.erode(edges_combined, kernel, iterations=1)

        # 細いエッジを元の画像に描画
        image_copy = img.copy()
        result_thin_edges = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        result_thin_edges[edges_combined == 255] = [255, 0, 0]
        
        st.image(result_thin_edges, caption='エッジ検出', use_column_width=True)


        # ------------------------------------------------------
        # 楕円の認識
        # ------------------------------------------------------
        # # 楕円の認識
        # image_gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
        # _, binary_mask = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY)

        # # ラベリング
        # labeled_image = label(binary_mask)

        # # 楕円のプロパティを取得
        # properties = regionprops(labeled_image)

        # # 楕円を塗りつぶすためのマスクを初期化
        # fill_mask = np.zeros_like(binary_mask, dtype=np.uint8)

        # # 領域ごとにチェックし、楕円として認識されるもののみを塗りつぶす
        # for prop in properties:
        #     if prop.major_axis_length > 10 and prop.minor_axis_length > 10:  # この値は調整するかもしれません
        #         fill_mask[labeled_image == prop.label] = 255

        # # 楕円の中を青で塗りつぶす
        # result_image = img.copy()
        # indices = np.where(fill_mask == 255)
        # result_image[indices[0], indices[1], :] = [0, 0, 255]

        # st.image(result_image, caption='楕円を塗りつぶした画像', use_column_width=True)        
        
        # 関数を使用
        result_image_path, measurements = measure_object_length(uploaded_file)

        # 計測結果の画像を表示
        result_image = Image.open(result_image_path)
        st.image(result_image, caption='計測結果', use_column_width=True)

        # 番号を動的に採番して計測データのテーブルを作成
        data = {
            'No.': list(range(1, len(measurements) + 1)),  # ここで 'No.' カラムを明示的に作成
            '長さ': measurements
        }
        df = pd.DataFrame(data).set_index('No.')

        st.write(df)  # DataFrameを表示
        
        
        
        
# st.write('DataFrame')

# df = pd.DataFrame({
#   '1列目': [1, 2, 3, 4],
#   '2列目': [10, 20, 30, 40],
# })
# st.write(df)
# st.dataframe(df.style.highlight_max(axis=0), width=300, height=None)
# st.table(df.style.highlight_max(axis=0))

# """
# # 章
# ## 節
# ### 項

# ```python
# import streamlit as st
# import numpy as np
# import pandas as pd
# ```
# """

# 表
# df = pd.DataFrame(
#   np.random.rand(20, 3),
#   columns=['a', 'b', 'c']
# )
# st.line_chart(df)
# st.area_chart(df)
# st.bar_chart(df)

# マップ
# df = pd.DataFrame(
#   np.random.rand(100, 2) / [50, 50] + [35.69, 139.70],
#   columns=['lat', 'lon']
# )
# st.map(df)

# イメージ
# st.write('Display Image')
# img = Image.open('image.jpg')
# st.image(img, caption='対象画像', use_column_width=True)

# チェックボックス 
# st.write('Display Image')
# if st.checkbox('Show Image'):
#   img = Image.open('image.jpg')
#   st.image(img, caption='対象画像', use_column_width=True)
  
# チェックボックス
# option = st.selectbox(
#   'あなたが好きな数字を教えてください。',
#   list(range(1, 11))
# )
# 'あなたの好きな数字は', option, 'です。'

# テキスト
# st.write('Interactive Widgets')
# text = st.text_input('あなたの趣味を教えて下さい。')
# st.write('あなたの趣味：', text)

# スライダー
# condition = st.slider('あなたの今の調子は？', 0, 100, 50)
# 'コンディション：', condition

# サイドバー
# text = st.sidebar.text_input('あなたの趣味を教えて下さい。')
# condition = st.sidebar.slider('あなたの今の調子は？', 0, 100, 50)
# st.write('あなたの趣味：', text)
# st.write('コンディション：', condition)

# 2カラムレイアウト
# left_column, right_column = st.columns(2)
# button = left_column.button('右カラムに文字を表示')

# if button:
#   right_column.write('ここは右カラムです')
  
# #エクスパンダー
# expander = st.expander('問い合わせ')
# expander.write('問い合わせを書く')

#プログレスバー
# st.write('プログレスバーの表示')
# st.write('Start!!')

# latest_iteration = st.empty()
# bar = st.progress(0)

# for i in range(100):
#   latest_iteration.text(f'Iteration {i+1}')
#   bar.progress(i + 1)
#   time.sleep(0.1)

# st.write('Done!!')

