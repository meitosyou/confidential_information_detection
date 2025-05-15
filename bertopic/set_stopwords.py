import nltk
from nltk.corpus import stopwords
from typing import Set

def load_custom_stopwords(file_path: str) -> Set[str]:
    """カスタムストップワードをファイルから読み込む

    Args:
        file_path (str): ストップワードファイルのパス

    Returns:
        Set[str]: カスタムストップワードのセット
    """
    custom_stopwords = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # コメント行をスキップ
            if line.strip() and not line.startswith('#'):
                custom_stopwords.add(line.strip())
    return custom_stopwords

def get_stopwords_en(custom_stopwords_path: str) -> Set[str]:
    """ストップワードのセットを取得

    Args:
        custom_stopwords_path (str): カスタムストップワードファイルのパス

    Returns:
        Set[str]: NLTKの標準ストップワードとカスタムストップワードを統合したセット
    """
    # NLTKのストップワードをダウンロード
    nltk.download('stopwords')

    # NLTKの英語ストップワード
    stop_words = set(stopwords.words('english'))
    # print(len(stop_words))
    # カスタムストップワードを読み込んで統合
    custom_stopwords = load_custom_stopwords(custom_stopwords_path)
    stop_words.update(custom_stopwords)
    # print(len(stop_words))
    return stop_words
