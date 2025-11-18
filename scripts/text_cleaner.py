
import unicodedata


def clean_text(text):
    text = text.replace('|', ' ')
    text.lower()
    return text


def title_to_index(col):
    col = col.apply(lambda x: unicodedata.normalize("NFKD", x))
    return {title: i for i, title in enumerate(col)}
