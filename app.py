import json
import os
from pathlib import Path

import gradio as gr
import pandas as pd

# ベースディレクトリのパス
BASE_DIR = "/nas/k_ishikawa/datasets/HappyRat/synthesis"
JSON_SUBDIR = "evaluation/illogical/jsons"

# 現在のファイルインデックスをグローバル変数として保持
current_file_index = 0


def get_iteration_dirs():
    """利用可能なiterationディレクトリのリストを取得"""
    iter_dirs = []
    if os.path.exists(BASE_DIR):
        for item in os.listdir(BASE_DIR):
            item_path = os.path.join(BASE_DIR, item)
            if os.path.isdir(item_path):
                # estimation/satisfaction/jsonsが存在するかチェック
                json_path = os.path.join(item_path, JSON_SUBDIR)
                if os.path.exists(json_path):
                    iter_dirs.append(item)
        iter_dirs.sort()  # ソート
    return iter_dirs


def get_json_dir(iteration):
    """指定されたiterationのJSONディレクトリパスを取得"""
    return os.path.join(BASE_DIR, iteration, JSON_SUBDIR)


def get_json_files(iteration="iter_0"):
    """JSONファイルのリストを取得"""
    json_dir = get_json_dir(iteration)
    json_files = []
    if os.path.exists(json_dir):
        json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
        json_files.sort()  # ファイル名でソート
    return json_files


def load_json_data(filename, iteration="iter_0"):
    """JSONファイルを読み込み、データを返す"""
    if not filename:
        return None, None, "", "", "", "", ""

    json_dir = get_json_dir(iteration)
    filepath = os.path.join(json_dir, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Messages データを抽出（nameとutteranceのみ）
        messages_data = []
        if "messages" in data:
            for msg in data["messages"]:
                name = msg.get("name", "")
                utterance = msg.get("utterance", "")
                messages_data.append({"name": name, "utterance": utterance})

        # DataFrameに変換
        df = pd.DataFrame(messages_data) if messages_data else pd.DataFrame()

        # 他のフィールドを抽出
        reason = data.get("reason", "")
        chosen = data.get("chosen", "")
        rejected = data.get("rejected", "")
        score = str(data.get("score", ""))
        speaker = data.get("speaker", "")
        scene = data.get("scene", "")

        return df, scene, reason, chosen, rejected, score, speaker

    except Exception as e:
        return None, None, f"Error loading file: {str(e)}", "", "", "", ""


def get_current_file_data(iteration="iter_0"):
    """現在のファイルインデックスに基づいてデータを取得"""
    json_files = get_json_files(iteration)
    if (
        not json_files
        or current_file_index < 0
        or current_file_index >= len(json_files)
    ):
        return "", None, None, "", "", "", "", ""

    current_filename = json_files[current_file_index]
    df, scene, reason, chosen, rejected, score, speaker = load_json_data(
        current_filename, iteration
    )
    return current_filename, df, scene, reason, chosen, rejected, score, speaker


def navigate_to_previous(iteration):
    """前のファイルに移動"""
    global current_file_index
    json_files = get_json_files(iteration)
    if json_files and current_file_index > 0:
        current_file_index -= 1
    return get_current_file_data(iteration)


def navigate_to_next(iteration):
    """次のファイルに移動"""
    global current_file_index
    json_files = get_json_files(iteration)
    if json_files and current_file_index < len(json_files) - 1:
        current_file_index += 1
    return get_current_file_data(iteration)


def refresh_files(iteration):
    """ファイルリストを更新し、最初のファイルに戻る"""
    global current_file_index
    current_file_index = 0
    return get_current_file_data(iteration)


def on_iteration_change(iteration):
    """iterationが変更された時の処理"""
    global current_file_index
    current_file_index = 0
    return get_current_file_data(iteration)


def create_interface():
    """Gradioインターフェースを作成"""

    with gr.Blocks(title="JSON サンプル ビューア", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📄 JSON サンプル ビューア")
        gr.Markdown("指定されたディレクトリのJSONファイルを確認できます。")

        with gr.Row():
            with gr.Column(scale=3):
                # 現在のファイル名表示
                current_file_textbox = gr.Textbox(
                    label="現在のファイル", interactive=False, value=""
                )

            with gr.Column(scale=1):
                # リフレッシュボタン
                refresh_btn = gr.Button("🔄 ファイルリストを更新", variant="secondary")

        with gr.Row():
            # シーン情報
            scene_textbox = gr.Textbox(
                label="シーン (Scene)", lines=10, max_lines=20, interactive=False
            )

        with gr.Row():
            # メッセージ表示
            messages_dataframe = gr.Dataframe(
                headers=["name", "utterance"],
                label="メッセージ (Messages)",
                interactive=False,
                wrap=True,
            )

        with gr.Row():
            with gr.Column():
                # Chosen
                chosen_textbox = gr.Textbox(
                    label="選択された回答 (Chosen)", lines=4, interactive=False
                )

            with gr.Column():
                # Rejected
                rejected_textbox = gr.Textbox(
                    label="却下された回答 (Rejected)", lines=4, interactive=False
                )

        with gr.Row():
            # Speaker, Score, Reason を1行で
            speaker_textbox = gr.Textbox(
                label="話者 (Speaker)", interactive=False, scale=1
            )
            score_textbox = gr.Textbox(
                label="スコア (Score)", interactive=False, scale=1
            )
            reason_textbox = gr.Textbox(
                label="理由 (Reason)", interactive=False, scale=3
            )

        with gr.Row():
            # ナビゲーションボタンを下部に配置
            prev_btn = gr.Button("⬅️ 前へ", variant="secondary", scale=1)
            next_btn = gr.Button("➡️ 次へ", variant="secondary", scale=1)

        with gr.Row():
            # ディレクトリ選択
            available_dirs = get_iteration_dirs()
            initial_dir = available_dirs[0] if available_dirs else None
            iteration_dropdown = gr.Dropdown(
                label="ディレクトリ",
                choices=available_dirs,
                value=initial_dir,
                interactive=True,
            )

        # イベントハンドラー
        outputs = [
            current_file_textbox,
            messages_dataframe,
            scene_textbox,
            reason_textbox,
            chosen_textbox,
            rejected_textbox,
            score_textbox,
            speaker_textbox,
        ]

        # ボタンイベント
        prev_btn.click(
            fn=navigate_to_previous, inputs=[iteration_dropdown], outputs=outputs
        )
        next_btn.click(
            fn=navigate_to_next, inputs=[iteration_dropdown], outputs=outputs
        )
        refresh_btn.click(
            fn=refresh_files, inputs=[iteration_dropdown], outputs=outputs
        )
        iteration_dropdown.change(
            fn=on_iteration_change, inputs=[iteration_dropdown], outputs=outputs
        )

        # 初期データ読み込み
        available_dirs = get_iteration_dirs()
        initial_iteration = available_dirs[0] if available_dirs else "iter_0"
        json_files = get_json_files(initial_iteration)
        if json_files:
            (
                initial_filename,
                initial_df,
                initial_scene,
                initial_reason,
                initial_chosen,
                initial_rejected,
                initial_score,
                initial_speaker,
            ) = get_current_file_data(initial_iteration)

            current_file_textbox.value = initial_filename
            messages_dataframe.value = initial_df
            scene_textbox.value = initial_scene
            reason_textbox.value = initial_reason
            chosen_textbox.value = initial_chosen
            rejected_textbox.value = initial_rejected
            score_textbox.value = initial_score
            speaker_textbox.value = initial_speaker

    return app


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, debug=True)
