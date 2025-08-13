import json
import random

from datasets import load_dataset
from langdetect import detect
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# バックアップファイルと出力ファイルの設定
BACKUP_FILE_NAME = "./backup.jsonl"
OUTPUT_FILE_NAME = "./synthesized_scenarios_new.jsonl"

# モデルの設定
# MODEL_NAME = "Aratako/Llama-Gemma-2-27b-SFT-trial1"
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8"
TENSOR_PARALLEL_SIZE = 8  # 利用環境のGPUの数に合わせる
MAX_NUM_SEQS = 500  # バッチサイズに合わせる？
MAX_NUM_BATCHED_TOKENS = 32768
MAX_MODEL_LEN = 16384
DOWNLOAD_DIR = "./cache"

# サンプリングパラメータの設定
SAMPLING_PARAMS = SamplingParams(
    temperature=1.0,
    top_p=0.9,
    max_tokens=2048,
    repetition_penalty=1.0,
)

# データ合成の設定
BATCH_SIZE = 50  # バッチサイズ


def is_japanese(text):
    """
    langdetectによる日本語判定
    """
    try:
        return detect(text) == "ja"
    except:
        return False


def is_zh(text):
    """
    SJISに変換して文字数が減れば簡体字があるので中国語と判定する
    一部旧字体や異体字が残る可能性もあるようだが無いよりはマシなものとして使う
    """
    return (set(text) - set(text.encode("sjis", "ignore").decode("sjis"))) != set([])


def get_random_characters():
    """
    ランダムに2体のキャラクターを取得
    """
    character_list = ["チュン太", "次郎", "KITT", "マーロ姫", "ククリ"]
    return random.sample(character_list, 2)


def _generate_base_instruction():
    """
    ベースとなるプロンプトを作成
    """
    _prompt = """
### 指示:
あなたはシナリオ作家です。与えられた2体のキャラクターのプロフィールを元に、これらのキャラクター間で展開される対話のシナリオを作成してください。出力の際には"#シナリオ:"に続ける形で対話のシナリオを出力してください。"
"""
    return _prompt


def _scene_instruction():
    """
    シーンの指示をランダムに選択
    """
    scenario = random.choice(
        [
            "動物園ワンダーズーの中のシナリオを作成してください。",
            "居酒屋でのシナリオを作成してください。",
            "SFシナリオを作成してください。",
            "会社でのシナリオを作成してください。",
            "海外旅行でのシナリオを作成してください。",
            "喧嘩しているシナリオを作成してください。",
            "感動する話を作成してください。",
            "悲しい話を作成してください。",
            "怖い話を作成してください。",
            "笑い話を作成してください。",
            "家族の話を作成してください。",
            "スラップスティクコメディを作成してください。",
            "駅での出会いのシナリオを作成してください。",
            "イベントでのシナリオを作成してください。",
            "恋人とのデートのシナリオを作成してください。",
            "ハロウィンパーティーでのシナリオを作成してください。",
            "クリスマスの夜のシナリオを作成してください。",
            "テレビ局がやってきたシナリオを作成してください。",
            "サーカス団がやってきたシナリオを作成してください。",
            "マフィアの世界のシナリオを作成してください。",
            "鬼ごっこをしているシナリオを作成してください。",
            "観覧車の中でのシナリオを作成してください。",
            "運動会でのシナリオを作成してください。",
            "電車の中でのシナリオを作成してください。",
            "海辺のキャンプのシナリオを作成してください。",
            "ゲームの中に入り込むシナリオを作成してください。",
            "荒廃した都市のシナリオを作成してください。",
            "海底でのシナリオを作成してください。",
            "森の奥で迷子になるシナリオを作成してください。",
            "船上でのシナリオを作成してください。",
            "魅惑のジャングルでのシナリオを作成してください。",
            "ダンスパーティーでのシナリオを作成してください。",
            "ミステリー小説のようなシナリオを作成してください。",
            "スポーツの試合でのシナリオを作成してください。",
            "宝探しの冒険シナリオを作成してください。",
            "島を脱出するシナリオを作成してください。",
            "病院でのシナリオを作成してください。",
            "動物大戦争のシナリオを作成してください。",
            "冬休みの思い出シナリオを作成してください。",
            "子供だけの冒険シナリオを作成してください。",
            "小さな町の謎解きシナリオを作成してください。",
            "魔法学校でのシナリオを作成してください。",
            "城での生活を描いたシナリオを作成してください。",
            "戦場でのシナリオを作成してください。",
            "サーカスでのシナリオを作成してください。",
            "不思議な図書館のシナリオを作成してください。",
            "モンスターと戦うシナリオを作成してください。",
            "おとぎ話風のシナリオを作成してください。",
            "舞台裏でのシナリオを作成してください。",
            "探偵のシナリオを作成してください。",
            "豪華客船でのシナリオを作成してください。",
            "嵐の夜のシナリオを作成してください。",
            "寮生活のシナリオを作成してください。",
            "修学旅行中のシナリオを作成してください。",
            "古代遺跡のシナリオを作成してください。",
            "山のふもとの村でのシナリオを作成してください。",
            "特殊能力を持つキャラクターのシナリオを作成してください。",
            "犯罪捜査のシナリオを作成してください。",
            "時間制限があるシナリオを作成してください。",
            "学校の試験前夜のシナリオを作成してください。",
            "幽霊の出る館のシナリオを作成してください。",
            "宇宙船の中でのシナリオを作成してください。",
            "ドラゴンとの共存する世界のシナリオを作成してください。",
            "動物たちの視点でのシナリオを作成してください。",
            "自然災害からの避難のシナリオを作成してください。",
            "高層ビルの中でのシナリオを作成してください。",
            "森の中の小屋でのシナリオを作成してください。",
            "無人島での生活のシナリオを作成してください。",
            "異星人との交流のシナリオを作成してください。",
            "天才少年の物語のシナリオを作成してください。",
            "伝説の武器を求めるシナリオを作成してください。",
            "不思議な祭壇のシナリオを作成してください。",
            "人間と動物が話せる世界のシナリオを作成してください。",
            "古代文明の復活シナリオを作成してください。",
            "仮想空間でのシナリオを作成してください。",
            "楽器にまつわるシナリオを作成してください。",
            "裏社会のシナリオを作成してください。",
            "冒険家たちの旅のシナリオを作成してください。",
            "魔法のアイテムを見つけるシナリオを作成してください。",
            "時代劇のようなシナリオを作成してください。",
            "海賊の冒険シナリオを作成してください。",
            "駅前商店街でのシナリオを作成してください。",
            "一夜限りのシナリオを作成してください。",
            "孤独なヒーローのシナリオを作成してください。",
            "崩壊する世界でのシナリオを作成してください。",
            "幻想的な雪国のシナリオを作成してください。",
            "ある日突然のシナリオを作成してください。",
            "村の伝説を元にしたシナリオを作成してください。",
            "学校の怪談を元にしたシナリオを作成してください。",
            "動物園の舞台裏のシナリオを作成してください。",
            "秘密の地下室でのシナリオを作成してください。",
            "学校の屋上でのシナリオを作成してください。",
            "屋台巡りのシナリオを作成してください。",
            "世界を救う鍵を巡るシナリオを作成してください。",
            "コンサートの舞台裏のシナリオを作成してください。",
            "神話のようなシナリオを作成してください。",
            "高校生の部活動のシナリオを作成してください。",
            "町中で大追跡のシナリオを作成してください。",
            "母と子の心温まるシナリオを作成してください。",
            "天界と地上の物語のシナリオを作成してください。",
            "街中で起こる怪奇現象のシナリオを作成してください。",
            "火山島でのサバイバルシナリオを作成してください。",
            "農村での生活のシナリオを作成してください。",
            "パズルを解くシナリオを作成してください。",
            "人狼ゲームをやっているシナリオを作成してください。",
            "テレビ出演をしているシナリオを作成してください。",
            "ラジオのMCをしているようなシナリオを作成してください。",
            "Youtubeに出演しているようなシナリオを作成してください。",
            "テレビのコメンテーターをしているようなシナリオを作成してください。",
            "ラジオで視聴者からのお便りに回答するようなシナリオを作成してください。",
            "テレビのバラエティー番組に出演しているのようなシナリオを作成してください。",
            "人間のお客さんが来て喜ぶシナリオを作成してください。",
            "動物園の舞台裏のシナリオを作成してください。",
            "トリビア披露会のシナリオを作成してください。",
        ]
    )
    return f"- {scenario}\n"


def _format_instruction():
    """
    プロンプトフォーマットをランダムに選択。汎化性能を最大化するため、10kパターンくらいを目指す。
    キャラクターの個性情報を付加するかどうかというところまで汎化性能を持たせている
    """
    # どのような要素をシナリオに含めるか
    youso = [
        "場所",
        "日時",
        "状況",
        "経緯",
        "関係性",
        "どう振る舞うべきか",
        "どういう感情か",
        "展開",
        "話のオチ",
        "結末",
        "結果",
    ]
    youso_list = random.sample(youso, random.randint(1, len(youso)))

    # シナリオの行数をランダムに選択
    num_lines = random.randint(1, 10)

    prompt = f"- {num_lines}行程度でシナリオを作成してください。その中には、{', '.join(youso_list)}などの要素を含めてください。\n"

    # 口調
    prompt += random.choice(
        [
            "- 説明口調で書いてください。\n",
            "- くだけた口調で書いてください。\n",
            "- 女子高生の口調で書いてください。\n",
            "- 編集者の指示的な口調で書いてください。\n",
            "- お役所的な口調で書いてください。\n",
            "- 理系男子風の論理的な口調で書いてください。\n",
            "- 劇作家の口調で書いてください。\n",
            "- ドキュメンタリーの口調で書いてください。\n",
            "",
        ]
    )

    # 書き方の指示
    prompt += random.choice(
        [
            "- 文章で書いてください。\n",
            "- 改行多めの文章で書いてください。\n",
            "- 箇条書きも交えて書いてください。\n",
            "- シナリオの概要を書いた後に、箇条書きで詳細を補足してください。\n",
            "- 箇条書きだけで書いてください。\n",
            "- 構造化された文章で書いてください。\n",
        ]
    )

    # 展開指示
    if random.random() < 0.5:
        prompt += random.choice(
            [
                "- 「N対話目でAAはXXをする」などと時系列を意識して書いてください。\n",
                "- 「N対話かけてAAは徐々にXXしていきます」など、時系列を意識して書いてください。\n",
                "- 「AAがXXという発言をしたら、BBはYYをする」などとキャラクターの条件指示を書いてください。\n",
                "- 「AAがXXをしたら、BBはYYをする」などとキャラクターの条件指示を書いてください。\n",
                "- 「AAさんがXXをしたら、BBはYYをする」などとキャラクターの条件指示を複数書いてください。\n",
                "- 「AAさんがXXをしたら、BBはYYをする」などとキャラクターの条件指示を中心に書いてください。\n",
            ]
        )

    return prompt


def _generate_scenario_instruction():
    """
    シナリオを生成するためのプロンプトを作成
    """
    _prompt = """
### シナリオの形式に関する指示
- 出力は必ず日本語で行ってください。
- 余分な説明は付け加えず、すぐに生成結果を出力してください。
"""
    # どういうシチュエーションのシナリオを作成するかを指示
    _prompt += _scene_instruction()
    # シナリオの書き方のクセを指示
    _prompt += _format_instruction()
    return _prompt


def _generate_profile_instruction(persona_1, persona_2, relation):
    """
    各キャラクターの設定をpromptに追加
    """
    prompt = "各キャラクターのプロフィールは以下の通りです。\n"
    prompt += "1人目の設定：\n"
    prompt += persona_1
    prompt += "\n"
    prompt += "2人目の設定：\n"
    prompt += persona_2
    prompt += "\n"
    prompt += f"2人目の1人目に対する関係：{relation}\n"
    return prompt


def _generate_suffix_instruction():
    """
    最後にもう一度指示のプロンプトを書くことで追従性を高める
    """
    prompt = """
それでは、上述した指示に従いキャラクター間で展開される対話のシナリオを作成してください。出力の際には"#シナリオ:"に続ける形で対話のシナリオを出力してください。"
"""
    return prompt


def initialize_model():
    """vLLMでモデルを初期化する"""
    return LLM(
        model=MODEL_NAME,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_model_len=MAX_MODEL_LEN,
        download_dir=DOWNLOAD_DIR,
        gpu_memory_utilization=0.9,
    )


def process_data(batch_size, model, tokenizer, data_batch):
    """vLLMによるバッチ推論を使ったデータ合成"""
    results = []
    prompts = []
    persona_1s = []
    persona_2s = []
    relations = []

    # バッチ推論用のプロンプトを構成する
    for i, data in enumerate(data_batch):
        persona_1 = data["generated_persona"]
        persona_2 = data["new_persona"]
        relation = data["relation"]
        system_prompt = _generate_base_instruction()
        user_prompt = _generate_scenario_instruction()
        user_prompt += _generate_profile_instruction(persona_1, persona_2, relation)
        user_prompt += _generate_suffix_instruction()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # Assistant PrefillによりConsistencyを確保
        prompt = (
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            + "#シナリオ:\n"
        )

        # デバッグ用
        if i == 0:
            print(system_prompt)
            print(user_prompt)
            print(prompt)

        prompts.append(prompt)
        persona_1s.append(persona_1)
        persona_2s.append(persona_2)
        relations.append(relation)

    outputs = model.generate(prompts, SAMPLING_PARAMS)

    for i, (persona_1, persona_2, relation, prompt, output) in enumerate(
        zip(persona_1s, persona_2s, relations, prompts, outputs)
    ):
        text = output.outputs[0].text.strip()
        # デバッグ用
        if i == 0:
            print(text)
        # 出力がちゃんとstopしており日本語である場合のみ有効なデータとして追加
        if (
            (output.outputs[0].finish_reason == "stop")
            and is_japanese(text)
            and not is_zh(text)
        ):
            new_data = {
                "persona_1": persona_1,
                "persona_2": persona_2,
                "relation": relation,
                "prompt": prompt,
                "scenario": text,
                "scenario_gen_model": MODEL_NAME,
            }
            results.append(new_data)

    return results


def save_backup(dataset, file_name="./backup.jsonl"):
    """バックアップを保存する関数"""
    with open(file_name, "w", encoding="utf-8") as f:
        for item in dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def process_dataset(model, tokenizer, profile_dataset, batch_size):
    """バッチサイズごとにデータを合成し、バックアップを保存する"""
    new_dataset = []

    for i in tqdm(range(0, len(profile_dataset), batch_size)):
        base_indices = list(range(i, min(i + batch_size, len(profile_dataset))))

        # インデックスを5倍に複製し、要素ごとにまとめる
        indices = []
        for idx in base_indices:
            indices.extend([idx] * 10)
        batch = profile_dataset.select(indices)
        new_data_list = process_data(batch_size, model, tokenizer, batch)
        new_dataset.extend(new_data_list)
        save_backup(new_dataset, BACKUP_FILE_NAME)
        print(f"現在のデータ数: {len(new_dataset)}")
    return new_dataset


def main():
    """メイン処理"""
    # モデルとtokenizerをロード
    model = initialize_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    profile_dataset = load_dataset(
        "Spiral-AI/Synthesized-Persona-20250103", split="train"
    )

    # 1ペアあたり10種類
    # 関係あるキャラ同士、関係ないキャラ同士

    # BATCH_SIZE分のデータ合成をN_TIMES回実行
    new_dataset = process_dataset(model, tokenizer, profile_dataset, BATCH_SIZE)

    print(f"作成されたデータ数: {len(new_dataset)}")

    # 結果の保存
    with open(OUTPUT_FILE_NAME, "w", encoding="utf-8") as f:
        for item in new_dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
