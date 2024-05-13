import argparse
import asyncio
import os
import shutil
import sys
import typing as ty
from dataclasses import dataclass
from multiprocessing.pool import Pool
from pathlib import Path

import aiohttp
import pdfkit
import pypdf
import pypdf.errors
from dotenv import dotenv_values
from rich.progress import Progress
from selectolax.lexbor import LexborHTMLParser, LexborNode

frozen = dataclass(frozen=True, slots=True, kw_only=True)

type SrcDstPairs = list[tuple[Path, Path]]


class _Sentinel: ...


SENTINEL = _Sentinel()


class ProcessingError(Exception): ...


class HTMLParsingError(ProcessingError): ...


class DownloadError(ProcessingError): ...


class MergingError(ProcessingError): ...


class ConvertionError(ProcessingError): ...


class FileMissingError(ProcessingError): ...


@frozen
class Config:
    DOWNLOAD_CONCURRENT_MAX: int = 200
    COMPUTE_PROCESS_MAX: int = os.cpu_count() or 1
    COMPUTE_PROCESS_TIMEOUT: int = 60
    PDF_CONVERTION_MAX_RETRY: int = 3
    BOOK_NAME: str = "learncpp.pdf"
    REMOVE_CACHE_ON_SUCCESS: bool = False

    LEARNCPP: str = "https://www.learncpp.com"
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    CACHE_FOLDER = PROJECT_ROOT / ".tmp"
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


def namesort(name: str) -> tuple[int, int | str]:
    """Used to sort filenames
    0 -> 9 -> "a" -> "z"
    """
    return (0, int(name)) if name.isdigit() else (1, name)


def _html_to_pdf(
    html: Path,
    dst_f: Path,
    options: dict = {"enable-local-file-access": ""},
) -> tuple[Exception | None, Path, Path]:
    post = Post.parse_html(name=html.stem, text=html.read_text())
    post.remove_elements()
    try:
        pdfkit.from_string(post.html, str(dst_f), options=options)
    except Exception as exc:
        return (exc, html, dst_f)
    else:
        return (None, html, dst_f)


def _merge_chapters(pdfs: list[Path], out: Path) -> Path:
    merger = pypdf.PdfWriter()
    for file in pdfs:
        try:
            merger.append(file)
        except FileNotFoundError as fe:
            raise MergingError(
                f"Failed to find {file}, make sure it exists or re-run the program to convert or download"
            ) from fe
        except pypdf.errors.PdfStreamError as pe:
            file.unlink()
            raise MergingError(
                f"Failed to merge {file} as the file is corrupted, please re-run the program to convert it again"
            )

    merger.write(out)
    merger.close()
    return out


@frozen
class Post:
    _REMOVE_ELEMENTS: ty.ClassVar[list[str]] = [
        "header.cryout",
        ".entry-content > .prevnext",
        ".comments-area",
    ]
    name: str
    root: LexborNode

    @property
    def html(self) -> str:
        assert self.root.html
        return self.root.html

    def remove_elements(self: "Post") -> None:
        for ele in self._REMOVE_ELEMENTS:
            if node := self.root.css_first(ele):
                node.decompose()

    @classmethod
    def parse_html(cls, *, name: str, text: str) -> "Post":
        root = LexborHTMLParser(text).root
        assert root
        return cls(name=name, root=root)


@frozen
class Chapter:
    no: str
    title: str
    link: str

    @property
    def filename(self) -> str:
        no = self.no.replace(".", "_").replace(" ", "_")
        return f"{no}.html"

    @classmethod
    def from_node(cls, chapter_node: LexborNode) -> "Chapter":
        chapter_no = chapter_node.css("div.lessontable-row-number")[0].text()
        title_node = chapter_node.css("div.lessontable-row-title")[0]
        title = title_node.text()
        link = title_node.css('a[href^="https://www.learncpp.com/cpp-tutorial/"]')[
            0
        ].attributes["href"]
        if not link:
            raise HTMLParsingError(f"Invalid link for {title}: {link}")
        return cls(no=chapter_no, title=title, link=link)


@frozen
class ChapterTable:
    "a groupd of chapters, e.g chapter 28.*"
    name: str
    chapters: tuple[Chapter, ...]

    @classmethod
    def from_node(cls, table_node: LexborNode) -> "ChapterTable":
        table_name = table_node.css('div.lessontable-header > a[name*="Chapter"]')[
            0
        ].attributes["name"]
        if not table_name:
            raise HTMLParsingError(f"Invalid table name: {table_name}")
        chapter_nodes = table_node.select("div.lessontable-row").matches
        chapters = tuple(
            Chapter.from_node(chapter_node) for chapter_node in chapter_nodes
        )
        return cls(name=table_name, chapters=chapters)


@frozen
class OutLinePage:
    root: LexborNode

    def content_tables(self):
        tables = self.root.select("div.lessontable").matches
        for table in tables:
            yield ChapterTable.from_node(table)

    @classmethod
    def parse_html(cls, text: str) -> "OutLinePage":
        outline_dom = LexborHTMLParser(text)
        if not outline_dom.body:
            raise HTMLParsingError("Failed to parse the html of the outline page")
        content = outline_dom.body.select("div.entry-content").matches[0]
        return cls(root=content)


class DownloadService:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        sems: asyncio.Semaphore,
        home_url: str,
        progress: Progress,
    ):
        self._session = session
        self._sems = sems
        self._home_url = home_url
        self._progress = progress

    async def get_content(self, url: str = "/") -> str:
        async with self._sems:
            async with self._session.get(url) as response:
                res = await response.text()
        return res

    async def download_outline(self) -> OutLinePage:
        html = await self.get_content(self._home_url)
        outline = OutLinePage.parse_html(html)
        return outline

    async def download_chapter(self, link: str, dst_f: Path) -> None:
        res = await self.get_content(link)
        dst_f.write_text(res)
        self._progress.update(self.__download_task, advance=1)

    async def download_chapters(self, chapters_folder: Path, use_cache: bool) -> None:
        outline = await self.download_outline()
        todo = set()
        for table in outline.content_tables():
            table_folder = chapters_folder / table.name
            table_folder.mkdir(exist_ok=True)
            for chapter in table.chapters:
                dst_f = table_folder / chapter.filename
                if use_cache and dst_f.exists():
                    continue
                coro = self.download_chapter(chapter.link, dst_f)
                todo.add(coro)

        if not todo:
            self._progress.log("Using cached htmls, skip download")
            return

        self.__download_task = self._progress.add_task("[red]Downloading HTMLs...")
        self._progress.update(self.__download_task, total=len(todo))
        await asyncio.gather(*todo)

    async def __aenter__(self) -> "DownloadService":
        await self._session.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tbx) -> None:
        await self._session.__aexit__(exc_type, exc, tbx)


class FileManager:
    def __init__(
        self,
        cache_folder: Path,
        html_folder: Path,
        html_chapter: Path,
        pdf_folder: Path,
        pdf_chapter: Path,
        pdf_merged_chapter_folder: Path,
        error_log: Path,
    ):
        self.cache_folder = cache_folder
        self.html_folder = html_folder
        self.html_chapter = html_chapter
        self.pdf_folder = pdf_folder
        self.pdf_chapter = pdf_chapter
        self.pdf_merged_chapter_folder = pdf_merged_chapter_folder
        self.error_log = error_log
        self.__setup()

    def __setup(self) -> None:
        self.cache_folder.mkdir(exist_ok=True)
        self.html_folder.mkdir(exist_ok=True)
        self.html_chapter.mkdir(exist_ok=True)
        self.pdf_folder.mkdir(exist_ok=True)
        self.pdf_chapter.mkdir(exist_ok=True)
        self.pdf_merged_chapter_folder.mkdir(exist_ok=True)
        self.error_log.touch()

    @property
    def chapter_folders(self) -> list[Path]:
        chapter_folders = sorted(
            self.html_chapter.iterdir(),
            key=lambda f: namesort(f.stem.split("Chapter")[1]),
        )
        return chapter_folders

    def sorted_dst_dirs(self) -> list[list[Path]]:
        dst_dirs = []
        for chapter_folder in self.chapter_folders:
            pdf_chapter = self.pdf_chapter / chapter_folder.name
            pdf_chapter.mkdir(exist_ok=True)
            htmls = sorted(
                chapter_folder.iterdir(), key=lambda h: namesort(h.stem.split("_")[1])
            )
            pdf_files = [pdf_chapter / f"{src_f.stem}.pdf" for src_f in htmls]
            dst_dirs.append(pdf_files)
        return dst_dirs

    def sorted_dir_pairs(self, use_cache: bool) -> SrcDstPairs:
        res: SrcDstPairs = []
        if not self.chapter_folders:
            raise FileMissingError(
                "Missing HTMLs of every chapters, please download them first"
            )
        for chapter_folder in self.chapter_folders:
            pdf_chapter = self.pdf_chapter / chapter_folder.name
            pdf_chapter.mkdir(exist_ok=True)
            htmls = sorted(
                chapter_folder.iterdir(), key=lambda h: namesort(h.stem.split("_")[1])
            )
            for src_f in htmls:
                dst_f = pdf_chapter / f"{src_f.stem}.pdf"
                if use_cache and dst_f.exists():
                    continue
                res.append((src_f, dst_f))
        return res

    def remove_cache(self) -> None:
        shutil.rmtree(self.cache_folder)

    def append_error(self, error_info: str) -> None:
        with self.error_log.open(mode="a") as f:
            f.write(f"{error_info} \n")

    def append_errors(self, error_infos: list[Exception | None]) -> None:
        with self.error_log.open(mode="a") as f:
            for error_info in error_infos:
                if not error_info:
                    continue
                f.write(f"{error_info} \n")

    def read_errors(self) -> str:
        return self.error_log.read_text()


class Application:
    def __init__(
        self,
        *,
        bookfile: Path,
        cvt_max_retries: int,
        dl_service: DownloadService,
        file_mgr: FileManager,
        progress: Progress,
        worker_pool: Pool,
        worker_timeout: int,
        remove_cache_on_success: bool = False,
    ):
        self._bookfile = bookfile
        self._cvt_max_retries = cvt_max_retries
        self._dl_service = dl_service
        self._file_mgr = file_mgr
        self._progress = progress
        self._worker_pool = worker_pool
        self._worker_timeout = worker_timeout
        self._remove_cache_on_success = remove_cache_on_success
        self.__task_succeed = False

    async def __aenter__(self) -> ty.Self:
        self._progress.__enter__()
        self._worker_pool.__enter__()
        return self

    async def __aexit__(
        self, exctype: type[Exception], exc: Exception, traceback
    ) -> None:
        self.__exit_log()
        self._worker_pool.__exit__(exctype, exc, traceback)
        await self._dl_service.__aexit__(exctype, exc, traceback)
        self._progress.__exit__(exctype, exc, traceback)

    def __exit_log(self):
        self._progress.log("Cleaning up ...")
        if self.__task_succeed:
            self._progress.console.rule("[green]Application succeeded")
        else:
            self._progress.console.rule(f"[red]Application Failed")
            self._progress.console.log(
                f"[red]See details in [bold]{self._file_mgr.error_log}[/bold]",
            )

    def succeed(self) -> None:
        self.__task_succeed = True

    async def download_chapters(self, use_cache: bool = True) -> None:
        await self._dl_service.download_chapters(
            self._file_mgr.html_chapter, use_cache=use_cache
        )

    def _convert_to_pdf(
        self, dir_pairs: SrcDstPairs
    ) -> list[tuple[Exception | None, Path, Path]]:
        results: list[tuple[Exception | None, Path, Path]] = []
        if not dir_pairs:
            self._progress.log("Using cached pdfs, skip converting")
            return results

        convert_task = self._progress.add_task("[green]Converting HTMLs...")

        tasks = [
            self._worker_pool.apply_async(_html_to_pdf, args=(src_f, dst_f))
            for src_f, dst_f in dir_pairs
        ]
        self._progress.update(convert_task, total=len(tasks))
        for task in tasks:
            res = task.get(timeout=self._worker_timeout)
            results.append(res)
            if not isinstance(res[0], Exception):
                self._progress.update(convert_task, advance=1)
        return results

    def _failed_cvt_filter(
        self, results: list[tuple[Exception | None, Path, Path]]
    ) -> SrcDstPairs:
        return [(src_f, dst_f) for _, src_f, dst_f in results if not dst_f.exists()]

    def convert_and_retry(self, use_cache: bool = True) -> None:
        results = self._convert_to_pdf(self._file_mgr.sorted_dir_pairs(use_cache))
        failed_dirs = self._failed_cvt_filter(results)
        retries = 0
        errors: list[Exception | None] = []

        while failed_dirs and retries < self._cvt_max_retries:
            self._progress.log(
                f"{len(failed_dirs)} HTMLs can't be converted, retyring, {self._cvt_max_retries - retries} retries left"
            )
            new_results = self._convert_to_pdf(failed_dirs)
            errors = [exc for exc, _, _ in results]
            failed_dirs = self._failed_cvt_filter(new_results)
            retries += 1

        if failed_dirs:
            self._file_mgr.append_errors(errors)
            raise ConvertionError(
                f"Failed to convert {len(failed_dirs)} HTMLs after retries, check {self._file_mgr.error_log} for failed htmls"
            )

    def _merging_pdfs(self, merging_folder: Path) -> list[Path]:
        dst_dirs = self._file_mgr.sorted_dst_dirs()
        if not dst_dirs:
            raise FileMissingError("Missing HTMLs to convert, download them first")

        merging_task = self._progress.add_task("[cyan]Merging PDFs...")

        tasks = []
        merged: list[Path] = []

        for pdf_dirs in dst_dirs:
            chapter_idx = pdf_dirs[0].parent.stem.split("Chapter")[1]
            dst_f = merging_folder / f"chapter_{chapter_idx}.pdf"
            if dst_f.exists():
                merged.append(dst_f)
                continue
            task = self._worker_pool.apply_async(
                _merge_chapters, args=(pdf_dirs, dst_f)
            )
            tasks.append(task)

        if not tasks:
            self._progress.log("Using cached chapters, skip merging")
            self._progress.remove_task(merging_task)
            return merged

        self._progress.update(merging_task, total=len(tasks))

        for task in tasks:
            try:
                res = task.get(timeout=self._worker_timeout)
            except Exception as e:
                raise MergingError(f"{e}")
            merged.append(res)
            self._progress.update(merging_task, advance=1)
        return merged

    def merge_chapters(self, use_cache: bool = True):
        bookfile = self._bookfile
        if use_cache and bookfile.exists():
            self._progress.log(f"Book '{bookfile.name}' alreasy exists, skip merging")
            return bookfile
        merged = self._merging_pdfs(self._file_mgr.pdf_merged_chapter_folder)
        learncpp = _merge_chapters(merged, bookfile)
        self._file_mgr.pdf_merged_chapter_folder.rmdir()
        return learncpp

    def show_errors(self) -> None:
        error_logs = self._file_mgr.error_log.read_text()
        if not error_logs:
            self._progress.log("No errors found in error log")
        for error_log in error_logs.split("\n"):
            self._progress.log(error_log)

    def application_succeeded(self):
        return not self._file_mgr.error_log.exists() or (
            self._file_mgr.read_errors() == ""
        )

    async def run(self, args: argparse.Namespace | _Sentinel = SENTINEL):
        if isinstance(args, _Sentinel):
            await self.download_chapters()
            self.convert_and_retry()
            self.merge_chapters()
        elif args.all:
            await self.download_chapters(use_cache=False)
            self.convert_and_retry(use_cache=False)
            self.merge_chapters(use_cache=False)
        else:
            if args.download:
                await self.download_chapters(use_cache=False)
                self._progress.log("Chapters downloaded")
            if args.convert:
                self.convert_and_retry(use_cache=False)
                self._progress.log("HTMLs Converted")
            if args.merge:
                self.merge_chapters(use_cache=False)
                self._progress.log("PDFs merged")
            if args.rmcache:
                self._file_mgr.remove_cache()
                self._progress.log("Cache removed")
            if args.showerrors:
                self.show_errors()

        if self.application_succeeded():
            self.succeed()
            if self._remove_cache_on_success:
                self._file_mgr.remove_cache()


def sessoin_factory(timeout: int = 120) -> aiohttp.ClientSession:
    timeout = aiohttp.ClientTimeout(total=timeout, connect=30)
    session = aiohttp.ClientSession()
    return session


def app_factory(config: Config) -> Application:
    sems = asyncio.Semaphore(value=config.DOWNLOAD_CONCURRENT_MAX)
    progress = Progress()
    dl_service = DownloadService(
        session=sessoin_factory(),
        sems=sems,
        home_url=config.LEARNCPP,
        progress=progress,
    )
    file_mgr = FileManager(
        cache_folder=config.CACHE_FOLDER,
        html_folder=config.HTML_FOLDER,
        html_chapter=config.HTML_CHAPTER,
        pdf_folder=config.PDF_FOLDER,
        pdf_chapter=config.PDF_CHAPTER,
        pdf_merged_chapter_folder=config.PDF_MERGED_CHAPTER_FOLDER,
        error_log=config.ERROR_LOG,
    )
    pool = Pool(processes=config.COMPUTE_PROCESS_MAX)

    app = Application(
        bookfile=config.BOOK_PATH,
        cvt_max_retries=config.PDF_CONVERTION_MAX_RETRY,
        dl_service=dl_service,
        file_mgr=file_mgr,
        progress=progress,
        worker_pool=pool,
        worker_timeout=config.COMPUTE_PROCESS_TIMEOUT,
    )
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-D",
        "--download",
        dest="download",
        help="Downloading articles from learcpp.com",
        action="store_true",
    )

    parser.add_argument(
        "-C",
        "--convert",
        dest="convert",
        help="Converting downloaded htmls to pdfs",
        action="store_true",
    )
    parser.add_argument(
        "-M", "--merge", help="Merging Chapters into a single book", action="store_true"
    )
    parser.add_argument(
        "-A", "--all", help="Download, convert and merge", action="store_true"
    )
    parser.add_argument("-R", "--rmcache", help="Remove cache", action="store_true")
    parser.add_argument("-S", "--showerrors", help="Show errors", action="store_true")
    args = parser.parse_args()
    return args


async def main():
    config = Config.from_env()
    args = parse_args() if len(sys.argv) > 1 else SENTINEL
    async with app_factory(config) as app:
        await app.run(args)


if __name__ == "__main__":
    asyncio.run(main())
