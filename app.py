import streamlit as st
import cv2
import numpy as np
from PIL import Image

def detect_aruco_corners(img_np):
    """
    ArUcoマーカー（ID: 0=左上, 1=右上, 2=右下, 3=左下）を検出して中心座標を取得する
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(aruco_dict)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
        
    marker_centers = {}
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in [0, 1, 2, 3]:
                pts = corners[i][0]
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                marker_centers[marker_id] = (cx, cy)
                
    if len(marker_centers) == 4:
        tl = marker_centers[0]  # ID 0: 左上
        tr = marker_centers[1]  # ID 1: 右上
        br = marker_centers[2]  # ID 2: 右下
        bl = marker_centers[3]  # ID 3: 左下
        return True, tl, tr, bl, br
        
    return False, None, None, None, None

def perspective_transform_with_offset(img, pts1, margin_px=25):
    """
    マーカー4点から補正を行い、マーカーから1の1までの余白(margin_px)を除外して600x600に正規化する
    """
    pts2 = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped_full = cv2.warpPerspective(img, M, (600, 600))
    
    # マーカーの内側にある「19x19の盤面領域」をトリミングして再拡大
    x1, y1 = margin_px, margin_px
    x2, y2 = 600 - margin_px, 600 - margin_px
    
    if x2 > x1 and y2 > y1:
        cropped = warped_full[y1:y2, x1:x2]
        warped_board = cv2.resize(cropped, (600, 600))
    else:
        warped_board = warped_full
        
    return warped_board

def detect_stones(warped_img, black_L=70, white_L=130, max_chroma=18):
    lab = cv2.cvtColor(warped_img, cv2.COLOR_RGB2LAB)
    board_state = np.zeros((19, 19), dtype=int)
    cell_size = 600.0 / 18.0
    r = 8
    
    for row in range(19):
        for col in range(19):
            cx = int(col * cell_size)
            cy = int(row * cell_size)
            if cx >= 600: cx = 599
            if cy >= 600: cy = 599
            
            y1, y2 = max(0, cy-r), min(600, cy+r)
            x1, x2 = max(0, cx-r), min(600, cx+r)
            
            patch_lab = lab[y1:y2, x1:x2]
            
            med_L = np.median(patch_lab[:, :, 0])
            med_A = np.median(patch_lab[:, :, 1]).astype(float) - 128
            med_B = np.median(patch_lab[:, :, 2]).astype(float) - 128
            chroma = np.sqrt(med_A**2 + med_B**2)
            
            if chroma > max_chroma:
                board_state[row, col] = 0
            else:
                if med_L < black_L:
                    board_state[row, col] = 1
                elif med_L > white_L:
                    board_state[row, col] = 2
                else:
                    board_state[row, col] = 0
                    
    return board_state

def calculate_territory_simple(board):
    rows, cols = 19, 19
    visited = np.zeros((rows, cols), dtype=bool)
    territory_map = np.zeros((rows, cols), dtype=int)
    
    black_score = 0
    white_score = 0
    
    for r in range(rows):
        for c in range(cols):
            if board[r, c] == 0 and not visited[r, c]:
                region = []
                adjacent_colors = set()
                queue = [(r, c)]
                visited[r, c] = True
                
                while queue:
                    curr_r, curr_c = queue.pop(0)
                    region.append((curr_r, curr_c))
                    
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            val = board[nr, nc]
                            if val == 0:
                                if not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    queue.append((nr, nc))
                            else:
                                adjacent_colors.add(val)
                
                if adjacent_colors == {1}:
                    for rr, cc in region:
                        territory_map[rr, cc] = 1
                    black_score += len(region)
                elif adjacent_colors == {2}:
                    for rr, cc in region:
                        territory_map[rr, cc] = 2
                    white_score += len(region)
                elif 1 in adjacent_colors and 2 in adjacent_colors:
                    for rr, cc in region:
                        territory_map[rr, cc] = 3
                        
    return territory_map, black_score, white_score

def render_detected_stones(warped_img, board_state, territory_map=None, target_row=None, target_col=None, show_grid=False):
    result_img = warped_img.copy()
    cell_size = 600.0 / 18.0
    
    # 格子ガイドラインの描画（位置合わせ用）
    if show_grid:
        for i in range(19):
            pos = int(i * cell_size)
            if pos >= 600: pos = 599
            cv2.line(result_img, (pos, 0), (pos, 600), (255, 100, 100), 1)
            cv2.line(result_img, (0, pos), (600, pos), (255, 100, 100), 1)

    if territory_map is not None:
        for row in range(19):
            for col in range(19):
                cx = int(col * cell_size)
                cy = int(row * cell_size)
                if cx >= 600: cx = 599
                if cy >= 600: cy = 599
                
                t = territory_map[row, col]
                if t == 1:
                    cv2.circle(result_img, (cx, cy), 4, (0, 200, 0), -1)
                elif t == 2:
                    cv2.circle(result_img, (cx, cy), 4, (0, 0, 255), -1)
                elif t == 3:
                    cv2.circle(result_img, (cx, cy), 3, (255, 0, 0), -1)

    for row in range(19):
        for col in range(19):
            cx = int(col * cell_size)
            cy = int(row * cell_size)
            if cx >= 600: cx = 599
            if cy >= 600: cy = 599
            
            stone = board_state[row, col]
            if stone == 1:
                cv2.circle(result_img, (cx, cy), 12, (0, 255, 0), 2)
            elif stone == 2:
                cv2.circle(result_img, (cx, cy), 12, (255, 0, 0), 2)
                
    if target_row is not None and target_col is not None:
        cx = int(target_col * cell_size)
        cy = int(target_row * cell_size)
        if cx >= 600: cx = 599
        if cy >= 600: cy = 599
        cv2.rectangle(result_img, (cx-14, cy-14), (cx+14, cy+14), (255, 255, 0), 3)
        cv2.circle(result_img, (cx, cy), 4, (255, 255, 0), -1)
        
    return result_img

st.set_page_config(layout="wide")
st.title('囲碁勝敗判定アプリ（ArUco全自動位置合わせ対応）')
uploaded_file = st.file_uploader('四隅にArUcoマーカーを貼った写真をアップロードしてください', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    
    # ArUcoマーカー検出
    success, tl, tr, bl, br = detect_aruco_corners(img_np)
    
    if success:
        st.toast("🎯 ArUcoマーカー4点を自動検出しました！", icon="✅")
        pts1 = np.float32([tl, tr, bl, br])
    else:
        st.error("⚠️ ArUcoマーカーが検出できませんでした。マーカー全4点（ID 0〜3）が画面内に写っているか確認してください。")
        st.stop()

    if "cur_row" not in st.session_state: st.session_state.cur_row = 0
    if "cur_col" not in st.session_state: st.session_state.cur_col = 0
    if "margin_px" not in st.session_state: st.session_state.margin_px = 25

    # サイドバー調整項目
    st.sidebar.subheader("📐 位置合わせ微調整")
    st.session_state.margin_px = st.sidebar.slider("マーカーから1の1（最外郭線）までの距離", 0, 80, st.session_state.margin_px)
    show_grid = st.sidebar.checkbox("19x19の格子ガイド線を表示する", value=True)

    warped = perspective_transform_with_offset(img_np, pts1, margin_px=st.session_state.margin_px)
    st.session_state.board_state = detect_stones(warped)
        
    tab1, tab2 = st.tabs(['🏆 勝敗判定・地計算', '🟢⚪ 石認識結果＆手動修正'])
    territory_map, black_territory, white_territory = calculate_territory_simple(st.session_state.board_state)
    
    with tab1:
        col_img1, col_res = st.columns([3, 2])
        
        with col_res:
            st.subheader('📊 判定結果・目数計算')
            komi = st.number_input('コミ (目数)', value=0.0, step=0.5)
            c_capt1, c_capt2 = st.columns(2)
            with c_capt1: black_captures = st.number_input('黒のアゲハ (石数)', min_value=0, value=0, step=1)
            with c_capt2: white_captures = st.number_input('白のアゲハ (石数)', min_value=0, value=0, step=1)
                
            st.markdown('---')
            total_black = black_territory + black_captures
            total_white = white_territory + white_captures + komi
            
            st.markdown(f'### 🟢 **黒の合計**: **{total_black:.1f}** 目')
            st.write(f'・地の数: {black_territory} 目')
            st.write(f'・アゲハ: {black_captures} 目')
            
            st.markdown(f'### 🔴 **白の合計**: **{total_white:.1f}** 目')
            st.write(f'・地の数: {white_territory} 目')
            st.write(f'・アゲハ: {white_captures} 目')
            st.write(f'・コミ: {komi} 目')
            
            st.markdown('---')
            diff = abs(total_black - total_white)
            if total_black > total_white:
                st.balloons()
                st.success(f'🏆 **【結果】黒の {diff:.1f} 目勝ち！**')
            elif total_white > total_black:
                st.balloons()
                st.error(f'🏆 **【結果】白の {diff:.1f} 目勝ち！**')
            else:
                st.info('⚖️ **【結果】持碁（引き分け）**')
                
        with col_img1:
            result_img_t = render_detected_stones(warped, st.session_state.board_state, territory_map=territory_map, show_grid=show_grid)
            st.image(result_img_t, use_container_width=True)

    with tab2:
        col_img, col_edit = st.columns([3, 2])
        
        with col_img:
            result_img = render_detected_stones(warped, st.session_state.board_state, target_row=st.session_state.cur_row, target_col=st.session_state.cur_col, show_grid=show_grid)
            st.image(result_img, use_container_width=True)

        with col_edit:
            st.subheader('🎯 石の手動修正')
            c_r, c_c = st.columns(2)
            with c_r: sel_row = st.number_input('行 (縦 1〜19)', min_value=1, max_value=19, value=st.session_state.cur_row+1, step=1)
            with c_c: sel_col = st.number_input('列 (横 1〜19)', min_value=1, max_value=19, value=st.session_state.cur_col+1, step=1)
            
            st.session_state.cur_row = int(sel_row - 1)
            st.session_state.cur_col = int(sel_col - 1)
            
            curr_state = st.session_state.board_state[st.session_state.cur_row, st.session_state.cur_col]
            state_label = "❌ なし（空点）" if curr_state == 0 else ("🟢 黒石" if curr_state == 1 else "🔴 白石")
            st.info(f'選択中の座標: **({sel_row}行, {sel_col}列)** ➔ 現在の判定: **{state_label}**')
            
            btn_c1, btn_c2, btn_c3 = st.columns(3)
            with btn_c1:
                if st.button('❌ なしに変更'):
                    st.session_state.board_state[st.session_state.cur_row, st.session_state.cur_col] = 0
                    st.rerun()
            with btn_c2:
                if st.button('🟢 黒石に変更'):
                    st.session_state.board_state[st.session_state.cur_row, st.session_state.cur_col] = 1
                    st.rerun()
            with btn_c3:
                if st.button('🔴 白石に変更'):
                st.session_state.board_state[st.session_state.cur_row, st.session_state.cur_col] =
                    st.rerun()
```[cite: 1]

<FollowUp label="新しい app.py の保存が無事に完了しましたら、次の手順についてお声がけください。" query="新しいapp.pyの保存が完了しました。次の手順を教えてください。"/>

<ElicitationsGroup message="ファイルを保存する中で、ご不明な点があれば以下をお選びください：">
  <Elicitation label="「すべてのファイル」への変え方で迷っている" query="メモ帳で名前を付けて保存する際、「すべてのファイル」にする具体的なやり方を教えてください。"/>
  <Elicitation label="保存完了したので次に進みたい" query="app.py の保存が終わりました。GitHubへのアップロード手順を詳しく教えてください。"/>
</ElicitationsGroup>
