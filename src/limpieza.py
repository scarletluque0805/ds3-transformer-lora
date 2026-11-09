"""Limpieza estructural del texto (la misma regex del Módulo 2), sin spaCy.

Para el Transformer usamos SOLO esta limpieza: se quitan restos de HTML, URLs, firmas de agencia y
datelines (ruido y atajos), pero NO se lematiza, NO se pasa a minúsculas ni se quitan stop-words.
El modelo preentrenado necesita el texto natural: el orden, las mayúsculas (si el modelo es cased),
las preposiciones y las negaciones son justamente el contexto que la auto-atención aprovecha.
"""
import html
import re

# Agencias y fuentes que firman las noticias de AG News
AGENCIAS = r"Reuters|AP|AFP|Canadian Press|CP|SPACE\.com|Sports Network|NewsFactor|TechWeb|Forbes\.com|washingtonpost\.com|USATODAY\.com|CNET|PC World|Ziff Davis|MarketWatch|Business Week|eWEEK|Investor's Business Daily"

_RE_ENTIDAD_ROTA = re.compile(r"(?<![&\w])#(\d+);")      # "#39;"  -> "&#39;"  (apóstrofe)
_RE_QUOT_ROTO = re.compile(r"(?<![&\w])(quot|amp|lt|gt);")  # "quot;" -> "&quot;"
_RE_HTML = re.compile(r"<.*?>", flags=re.S)                 # etiquetas HTML (no codicioso)
_RE_URL = re.compile(r"https?://\S+|www\.\S+|\b[\w.-]+\.(?:com|net|org)\b\S*", flags=re.I)
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
_RE_FUENTE_PAREN = re.compile(rf"\((?:{AGENCIAS})\)", flags=re.I)        # "(Reuters)"
_RE_FUENTE_GUION = re.compile(rf"(?:^|(?<=\. ))(?:{AGENCIAS})\s+-\s+", flags=re.I)  # "AP - " al inicio del cuerpo
_RE_QUOTE_PROFILE = re.compile(r"\(?\s*(?:[A-Z0-9.]+\s*:\s*)?Quote,\s*Profile,\s*Research\s*\)?")  # links de Reuters
_RE_UPDATE = re.compile(r"\bUPDATE\s*\d*\s*-\s*")                    # "UPDATE 1-" (versión del cable)
_RE_FUENTE_CIUDAD = re.compile(r"\([A-Za-z ,.]+-AP\)")                 # "(Washington-AP)"
# Dateline al inicio del cuerpo: "NEW YORK - ", "BAGHDAD, Iraq -- ", "LONDON (Reuters) - "
_RE_DATELINE = re.compile(r"(?:^|(?<=\. ))[A-Z][A-Z.'\- ]{2,}(?:,\s*[A-Z][A-Za-z.]+)*\s*(?:\([^)]*\))?\s*-{1,2}\s+")
_RE_TICKER = re.compile(r"\b[A-Z]{2,5}\.(?:N|O|OQ|PK|OB|L|TO|DE|PA|AS|T|HK|MI|F)\b")  # tickers "PEP.N", "AKLM.PK" (no toca U.S. ni U.N.)
_RE_ESPACIOS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Limpieza estructural con regex (no toca el significado del texto)."""
    text = _RE_ENTIDAD_ROTA.sub(r"&#\1;", text)
    text = _RE_QUOT_ROTO.sub(r"&\1;", text)
    text = html.unescape(html.unescape(text))   # dos veces: hay entidades doblemente escapadas (&amp;lt;)
    text = _RE_HTML.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = _RE_EMAIL.sub(" ", text)
    text = _RE_QUOTE_PROFILE.sub(" ", text)
    text = _RE_UPDATE.sub(" ", text)
    text = _RE_FUENTE_CIUDAD.sub(" ", text)
    for _ in range(2):                          # 2 pasadas: "AP - BEIJING - ..." tiene firma + dateline encadenados
        text = _RE_FUENTE_GUION.sub("", text)   # "Canadian Press - "
        text = _RE_DATELINE.sub(" ", text)      # antes de quitar "(Reuters)", que forma parte del dateline
        text = _RE_ESPACIOS.sub(" ", text)
    text = _RE_FUENTE_PAREN.sub(" ", text)
    text = _RE_FUENTE_GUION.sub(" ", text)
    text = _RE_TICKER.sub(" ", text)
    text = _RE_ESPACIOS.sub(" ", text)
    return text.strip()
