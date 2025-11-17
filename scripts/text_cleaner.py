
import unicodedata


def clean_text(text):
    text = text.replace('|', ' ')
    return text


def title_to_index(col):
    col = col.apply(lambda x: unicodedata.normalize("NFKD", x))
    return {title: i for i, title in enumerate(col)}


def clean_pipe_text(x):
    if pd.isna(x):
        return ""
    return " ".join(x.split)