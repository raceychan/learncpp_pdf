import os
from pathlib import Path


class Config:
    LEARNCPP: str = "https://www.learncpp.com"
    PROJECT_ROOT = Path(__file__).parent.parent
    HTML_FOLDER: Path = PROJECT_ROOT / "html"
    HTML_CHAPTER: Path = HTML_FOLDER / "chapters"
    CHAPTER_OUTLINE: Path = HTML_FOLDER / "outline.html"
    PDF_FOLDER: Path = PROJECT_ROOT / "pdf"
    PDF_CHAPTER: Path = PDF_FOLDER / HTML_CHAPTER.name
    PDF_BOOK_FOLDER: Path = PDF_FOLDER / "learncpp"
    DOWNLOAD_CONCURRENT_MAX: int = 100
    COMPUTE_PROCESS_MAX: int = os.cpu_count() or 1
