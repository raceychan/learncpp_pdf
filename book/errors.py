from pathlib import Path

class ProcessingError(Exception): ...


class MissingDependencyError(ProcessingError):

    def __init__(self, dep: str) -> None:
        super().__init__(f"Missing dependency: {dep}")


class HTMLParsingError(ProcessingError): ...


class DownloadError(ProcessingError):
    code: int
    detail: str

    def __init__(self, code: int, detail: str) -> None:
        super().__init__(f"Download error: {code} {detail}")


class MergingError(ProcessingError):
    file: Path


class PDFNotFoundError(MergingError):

    def __init__(self, file: Path):
        super().__init__(
            f"Failed to find {file}, make sure it exists or re-run the program to convert or download"
        )


class CorruptedPDFError(MergingError):

    def __init__(self, file: Path):
        super().__init__(
            f"Failed to convert {file} as the file is corrupted, please re-run the program to convert it again"
        )


class ConvertionError(ProcessingError):
    def __init__(self, failed_num: int, error_path: Path):
        super().__init__(
            f"Failed to convert {failed_num} HTMLs after retries, check {error_path} for failed htmls"
        )


class FileMissingError(ProcessingError): ...