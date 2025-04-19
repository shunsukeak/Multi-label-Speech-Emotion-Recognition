import os
import pandas as pd
from collections import defaultdict

# 有効な感情のみ（neutralも追加OK）
VALID_EMOTIONS = {'ang', 'hap', 'sad', 'neu', 'fru', 'exc'}  # merged 'exc' into 'hap' if needed

def parse_iemocap_multi(iemocap_dir):
    records = []
    sessions = [f'Session{i}' for i in range(1, 6)]

    for sess in sessions:
        emo_dir = os.path.join(iemocap_dir, sess, 'dialog', 'EmoEvaluation')
        wav_base = os.path.join(iemocap_dir, sess, 'sentences', 'wav')

        for file in os.listdir(emo_dir):
            if not file.endswith('.txt'): continue
            with open(os.path.join(emo_dir, file), 'r') as f:
                for line in f:
                    if line.startswith('['):
                        parts = line.strip().split()
                        utt_id, emo = parts[1], parts[2]
                        if emo not in VALID_EMOTIONS:
                            continue
                        wav_path = os.path.join(wav_base, utt_id[:5], utt_id + '.wav')
                        if os.path.exists(wav_path):
                            records.append({'path': wav_path, 'emotion': emo})

    # multi-label化
    label_map = defaultdict(set)
    for row in records:
        label_map[row['path']].add(row['emotion'])

    # binaryラベルでDataFrame作成
    all_emotions = sorted(VALID_EMOTIONS)
    data = []
    for path, emotions in label_map.items():
        entry = {'path': path}
        for emo in all_emotions:
            entry[emo] = int(emo in emotions)
        data.append(entry)

    return pd.DataFrame(data)

import pandas as pd
import os

def parse_cremad(cremad_dir, metadata_path, threshold=0.3):
    df = pd.read_csv(metadata_path)  # 評価者のラベルを含むCSV
    emotion_classes = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
    label_cols = [col for col in df.columns if col.startswith('rater_')]

    processed = []
    for _, row in df.iterrows():
        emo_counts = {e: 0 for e in emotion_classes}
        total_raters = 0
        for col in label_cols:
            label = row[col]
            if label in emo_counts:
                emo_counts[label] += 1
                total_raters += 1

        # 閾値超えた感情だけを1に
        entry = {'path': os.path.join(cremad_dir, row['file'])}
        for emo in emotion_classes:
            entry[emo.lower()] = int(emo_counts[emo] / total_raters >= threshold)
        processed.append(entry)

    return pd.DataFrame(processed)

# ラベルの統一名を揃える（ang = angry, hap = happy など）
iemocap_df = parse_iemocap_multi('/path/to/IEMOCAP')
cremad_df = parse_cremad('/path/to/CREMA-D/audio', '/path/to/CREMA-D/metadata.csv')

# ラベル名のマッピング（例：hap → happy）
rename_map = {'ang': 'angry', 'hap': 'happy', 'sad': 'sad', 'neu': 'neutral', 'fru': 'frustration', 'exc': 'excitement'}
iemocap_df = iemocap_df.rename(columns=rename_map)

# ラベルの共通化（存在しないラベルは0で埋める）
all_labels = sorted(set(iemocap_df.columns).union(cremad_df.columns) - {'path'})
iemocap_df = iemocap_df[['path'] + [l if l in iemocap_df.columns else iemocap_df.assign(**{l: 0})[l] for l in all_labels]]
cremad_df = cremad_df[['path'] + [l if l in cremad_df.columns else cremad_df.assign(**{l: 0})[l] for l in all_labels]]

# 統合
full_df = pd.concat([iemocap_df, cremad_df], ignore_index=True).fillna(0).astype({l: int for l in all_labels})
