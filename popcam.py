import cv2
import numpy as np

# --- 設定 ---
# ミックスモード時のフレームの重み（小さいほど残像が強く残る）
MIX_ALPHA = 0.9

# --- 初期化 ---
# モード管理用のフラグ (False: ネガポジ&輪郭モード, True: ミックスモード)
mix_mode = False
# ミックスモードで使用する前のフレームを保持する変数
prev_frame = None

# カメラのキャプチャを開始 (0は内蔵カメラを意味します)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("エラー: カメラを開けませんでした。")
    exit()

print("カメラを起動しました。")
print("------------------------------------")
print("q: プログラムを終了します。")
print("m: エフェクトモードを切り替えます。")
print("------------------------------------")


# メインループ
while True:
    # カメラから1フレーム読み込む
    ret, frame = cap.read()
    if not ret:
        print("エラー: フレームをキャプチャできませんでした。")
        break

    # キー入力を待つ (1ミリ秒)
    key = cv2.waitKey(1) & 0xFF

    # 'q'キーが押されたらループを抜ける
    if key == ord('q'):
        print("終了します。")
        break
    
    # 'm'キーが押されたらモードを切り替える
    if key == ord('m'):
        mix_mode = not mix_mode
        # ミックスモードに切り替わった直後は前のフレームがないので初期化
        if mix_mode:
            print("モード: ミックスエフェクト")
            # 現在のフレームのサイズと型で黒い画像を生成して初期化
            prev_frame = np.zeros_like(frame, dtype=np.float32)
        else:
            print("モード: ネガポジ反転 & 輪郭")


    # 処理結果を表示するフレームを入れる変数
    output_frame = None

    # モードに応じて映像を処理
    if mix_mode:
        # --- ミックスモードの処理 ---
        # 「コーヒーにクリームを溶かす」ようなエフェクト
        
        # 現在のフレームをfloat32に変換
        current_frame_float = frame.astype(np.float32)

        # 現在のフレームと前のフレームをブレンドする
        # cv2.addWeighted(src1, alpha, src2, beta, gamma)
        # output = src1 * alpha + src2 * beta + gamma
        mixed_frame = cv2.addWeighted(current_frame_float, 1.0 - MIX_ALPHA, prev_frame, MIX_ALPHA, 0)
        
        # 次のループのために現在のブレンド結果を保存
        prev_frame = mixed_frame
        
        # 表示用にuint8に型を戻す
        output_frame = mixed_frame.astype(np.uint8)

    else:
        # --- ネガポジ反転 & 輪郭モードの処理 ---
        
        # 1. 映像をネガポジ反転させる
        negative_frame = cv2.bitwise_not(frame)

        # 2. 輪郭線を検出する
        # グレースケールに変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ノイズを減らすために少しぼかす
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # Canny法でエッジ（輪郭）を検出
        edges = cv2.Canny(gray, 50, 150)

        # 3. ネガポジ反転した映像に、検出した輪郭を白で描画する
        # 輪郭（edges）が白(0以外)の部分だけ、negative_frameを白([255, 255, 255])で上書き
        negative_frame[edges != 0] = [255, 255, 255]
        
        output_frame = negative_frame


    # 結果のフレームをウィンドウに表示
    cv2.imshow('Pop Art Camera', output_frame)


# --- 終了処理 ---
# カメラを解放
cap.release()
# すべてのウィンドウを閉じる
cv2.destroyAllWindows()
