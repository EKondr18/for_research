"""PDF text extraction and chunking."""
import logging

from pypdf import PdfReader
from pypdf.errors import PdfReadError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from io import BytesIO

logger = logging.getLogger("pdf_processing")


class Chunk:
    __slots__ = ("text", "page")

    def __init__(self, text: str, page: int):
        self.text = text
        self.page = page


def extract_pages_text(pdf_bytes: bytes, file_name: str) -> list[tuple[int, str]]:
    """Returns a list of (page_number_1_indexed, text). Pages with no
    extractable text (e.g. scanned images without a text layer) are skipped
    with a warning -- no OCR is performed at this stage."""
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except PdfReadError as e:
        logger.warning("Не удалось открыть PDF '%s': %s. Файл пропущен.", file_name, e)
        return []

    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:  # pypdf can raise various parsing errors on malformed pages
            logger.warning("Ошибка извлечения текста со страницы %d файла '%s': %s", i, file_name, e)
            continue
        text = text.strip()
        if not text:
            continue
        pages.append((i, text))

    if not pages:
        logger.warning(
            "В файле '%s' не найдено текстового слоя ни на одной странице "
            "(похоже на скан без OCR). Файл пропущен при индексации.",
            file_name,
        )
    return pages


def chunk_pages(pages: list[tuple[int, str]], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Chunk] = []
    for page_number, text in pages:
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if piece:
                chunks.append(Chunk(text=piece, page=page_number))
    return chunks
