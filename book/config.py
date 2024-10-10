from dataclasses import dataclass

frozen = dataclass(frozen=True, slots=True, kw_only=True)

import os
from pathlib import Path

from dotenv import dotenv_values


@frozen
class Config:
    DOWNLOAD_CONCURRENT_MAX: int = 200
    COMPUTE_PROCESS_MAX: int = os.cpu_count() or 1
    COMPUTE_PROCESS_TIMEOUT: int = 60
    DOWNLOAD_CONTENT_RETRY: int = 6
    PDF_CONVERTION_MAX_RETRY: int = 3
    BOOK_NAME: str = "learncpp.pdf"
    REMOVE_CACHE_ON_SUCCESS: bool = False

    LEARNCPP: str = "https://www.learncpp.com"
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    CACHE_FOLDER: Path = PROJECT_ROOT / ".tmp"
    ERROR_LOG: Path = CACHE_FOLDER / "error"
    HTML_FOLDER: Path = CACHE_FOLDER / "html"
    HTML_CHAPTER: Path = HTML_FOLDER / "chapters"
    CHAPTER_OUTLINE: Path = HTML_FOLDER / "outline.html"
    PDF_FOLDER: Path = CACHE_FOLDER / "pdf"
    PDF_CHAPTER: Path = PDF_FOLDER / HTML_CHAPTER.name
    PDF_MERGED_CHAPTER_FOLDER: Path = PDF_FOLDER / "learncpp"

    @property
    def BOOK_PATH(self) -> Path:
        return self.PROJECT_ROOT / self.BOOK_NAME

    @classmethod
    def from_env(cls, env_file: Path = Path.cwd() / ".env") -> "Config":
        values = dotenv_values(env_file)
        for k, v in values.items():
            if val_type := cls.__annotations__.get(k):
                try:
                    values[k] = val_type(v)
                except ValueError as e:
                    raise ValueError(
                        f"Invalid value for {k}, make sure it can be parsed as {val_type}"
                    )
        return cls(**values)
