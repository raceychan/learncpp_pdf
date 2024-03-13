import asyncio
import os
import shutil
import typing as ty
from dataclasses import dataclass
from functools import cached_property
from multiprocessing.pool import Pool
from pathlib import Path

import aiohttp
import pdfkit
import pypdf
from dotenv import dotenv_values
from loguru import logger
from rich.progress import Progress
from selectolax.lexbor import LexborHTMLParser, LexborNode

frozen = dataclass(frozen=True, slots=True, kw_only=True)

type SrcDstPairs = list[tuple[Path, Path]]


@frozen
class Config:
    DOWNLOAD_CONCURRENT_MAX: int = 200
    COMPUTE_PROCESS_MAX: int = os.cpu_count() or 1
    PDF_CONVERTION_MAX_RETRY: int = 3

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


def append_error(error_log: Path, error_info: str) -> None:
    with error_log.open(mode="a") as f:
        f.write(f"{error_info} \n")


def namesort(name: str) -> tuple[int, int | str]:
    """Used to sort filenames
    0 -> 9 -> "a" -> "z"
    """
    return (0, int(name)) if name.isdigit() else (1, name)


def html_to_pdf(
    html: Path,
    dst_f: Path,
    options: dict = {"enable-local-file-access": ""},
) -> tuple[int, Path, Path]:
    post = Post.parse_html(name=html.stem, text=html.read_text())
    post.remove_elements()
    try:
        pdfkit.from_string(post.html, str(dst_f), options=options)
    except Exception as ge:
        return (0, html, dst_f)
    else:
        return (1, html, dst_f)


def merge_chapters(pdfs: list[Path], out: Path) -> Path:
    merger = pypdf.PdfWriter()
    for file in pdfs:
        merger.append(file)

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
            raise ValueError(f"Invalid link: {link}")
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
            raise ValueError(f"Invalid table name: {table_name}")
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
            raise ValueError("Failed to parse html")
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

    async def download_chapters(self, chapters_folder: Path) -> None:
        self.__download_task = self._progress.add_task("[red]Downloading HTMLs...")
        outline = await self.download_outline()
        todo = set()
        for table in outline.content_tables():
            table_folder = chapters_folder / table.name
            table_folder.mkdir(exist_ok=True)
            for chapter in table.chapters:
                dst_f = table_folder / chapter.filename
                if dst_f.exists():
                    continue
                coro = self.download_chapter(chapter.link, dst_f)
                todo.add(coro)

        if not todo:
            self._progress.log("Using cached htmls, skip download")
            self._progress.remove_task(self.__download_task)
            return
        self._progress.update(self.__download_task, total=len(todo))
        await asyncio.gather(*todo)

    async def close(self) -> None:
        await self._session.close()


class FileManager:
    def __init__(self, config: Config):
        self._config = config
        self.__setup()

    def __setup(self) -> None:
        self._config.CACHE_FOLDER.mkdir(exist_ok=True)
        self._config.HTML_FOLDER.mkdir(exist_ok=True)
        self._config.HTML_CHAPTER.mkdir(exist_ok=True)
        self._config.PDF_FOLDER.mkdir(exist_ok=True)
        self._config.PDF_CHAPTER.mkdir(exist_ok=True)
        self._config.PDF_MERGED_CHAPTER_FOLDER.mkdir(exist_ok=True)
        self._config.ERROR_LOG.touch()

    @property
    def chapter_folders(self) -> list[Path]:
        chapter_folders = sorted(
            self._config.HTML_CHAPTER.iterdir(),
            key=lambda f: namesort(f.stem.split("Chapter")[1]),
        )
        return chapter_folders

    def sorted_dst_dirs(self):
        for chapter_folder in self.chapter_folders:
            pdf_chapter = self._config.PDF_CHAPTER / chapter_folder.name
            pdf_chapter.mkdir(exist_ok=True)
            htmls = sorted(
                chapter_folder.iterdir(), key=lambda h: namesort(h.stem.split("_")[1])
            )
            pdf_files = [pdf_chapter / f"{src_f.stem}.pdf" for src_f in htmls]
            yield pdf_files

    def sorted_dir_pairs(self) -> SrcDstPairs:
        res: SrcDstPairs = []
        for chapter_folder in self.chapter_folders:
            pdf_chapter = self._config.PDF_CHAPTER / chapter_folder.name
            pdf_chapter.mkdir(exist_ok=True)
            htmls = sorted(
                chapter_folder.iterdir(), key=lambda h: namesort(h.stem.split("_")[1])
            )
            for src_f in htmls:
                dst_f = pdf_chapter / f"{src_f.stem}.pdf"
                if dst_f.exists():
                    continue
                res.append((src_f, dst_f))
        return res

    def convert_failed_htmls(self):
        res: SrcDstPairs = []
        error_logs = self._config.ERROR_LOG.read_text()
        for error_log in error_logs.split():
            src_f = Path(error_log.strip())
            dst_f = self._config.PDF_CHAPTER / src_f.parent.name / f"{src_f.stem}.pdf"
            res.append((src_f, dst_f))
        return res

    def remove_cached(self):
        shutil.rmtree(self._config.CACHE_FOLDER)


class Application:
    def __init__(
        self,
        *,
        config: Config,
        dl_service: DownloadService,
        file_mgr: FileManager,
        progress: Progress,
        worker_pool: Pool,
    ):
        self._config = config
        self._dl_service = dl_service
        self._file_mgr = file_mgr
        self._progress = progress
        self._worker_pool = worker_pool
        self.__task_succeed = False

    @cached_property
    def config(self):
        return self._config

    async def __aenter__(self):
        self._progress.__enter__()
        return self

    async def __aexit__(self, exc, excval, traceback):
        self._progress.__exit__(exc, excval, traceback)
        await self.close()
        if self.__task_succeed:
            logger.success("Application suceeded and now exiting")
        else:
            logger.error(
                f"Application interrupted due to unknown error, check {self._config.ERROR_LOG} for missing chapters"
            )

    async def download_chapters(self):
        await self._dl_service.download_chapters(self._config.HTML_CHAPTER)

    def _convert_to_pdf(self, dir_pairs: SrcDstPairs):
        convert_task = self._progress.add_task("[green]Converting HTMLs...")
        res: list[tuple[int, Path, Path]] = []

        if not dir_pairs:
            self._progress.log("Using cached pdfs, skip converting")
            self._progress.remove_task(convert_task)
            return res

        tasks = [
            self._worker_pool.apply_async(html_to_pdf, args=(src_f, dst_f))
            for src_f, dst_f in dir_pairs
        ]
        self._progress.update(convert_task, total=len(tasks))
        for task in tasks:
            res.append(task.get())
            self._progress.update(convert_task, advance=1)
        return res

    def convert_and_retry(self, dir_pairs: SrcDstPairs):
        fail_filter = lambda res: [
            (src_f, dst_f) for flag, src_f, dst_f in res if flag == 1
        ]
        res = self._convert_to_pdf(dir_pairs)
        for _ in range(self._config.PDF_CONVERTION_MAX_RETRY):
            if not res:
                break
            res = self._convert_to_pdf(fail_filter(res))
        else:
            failed = fail_filter(res)
            for src_f, _ in failed:
                append_error(self._config.ERROR_LOG, str(src_f))

            self._progress.log(
                f"{len(failed)} htmls can't be converted after retries, check {self._config.ERROR_LOG} for details"
            )

    def _merging_pdfs(self, merging_folder: Path) -> list[Path]:
        merging_task = self._progress.add_task("[cyan]Merging PDFs...")

        tasks = []
        merged: list[Path] = []
        for pdf_dirs in self._file_mgr.sorted_dst_dirs():
            chapter_idx = pdf_dirs[0].parent.stem.split("Chapter")[1]
            dst_f = merging_folder / f"chapter_{chapter_idx}.pdf"
            if dst_f.exists():
                merged.append(dst_f)
                continue
            task = self._worker_pool.apply_async(merge_chapters, args=(pdf_dirs, dst_f))
            tasks.append(task)

        if not tasks:
            self._progress.log("Using cached chapters, skip merging")
            self._progress.remove_task(merging_task)
            return merged

        self._progress.update(merging_task, total=len(tasks))

        for task in tasks:
            merged.append(task.get())
            self._progress.update(merging_task, advance=1)

        return merged

    def merging_chapters(self, merging_folder: Path, res_path: Path):
        merged = self._merging_pdfs(merging_folder)
        if res_path.exists():
            return res_path
        learncpp = merge_chapters(merged, res_path)
        return learncpp

    def application_succeeded(self):
        errors = self._config.ERROR_LOG.read_text()
        return not errors

    async def close(self):
        self._worker_pool.close()
        await self._dl_service.close()

    async def run(self):
        await self.download_chapters()
        self.convert_and_retry(self._file_mgr.sorted_dir_pairs())
        self.merging_chapters(
            self._config.PDF_MERGED_CHAPTER_FOLDER,
            self._config.PROJECT_ROOT / "learncpp.pdf",
        )

        if self.application_succeeded():
            self._file_mgr.remove_cached()
            self.__task_succeed = True


def sessoin_factory():
    # timeout = aiohttp.ClientTimeout(total=60, connect=10)
    session = aiohttp.ClientSession()
    return session


def app_factory(config: Config):
    sems = asyncio.Semaphore(config.DOWNLOAD_CONCURRENT_MAX)
    progress = Progress()
    dl_service = DownloadService(
        session=sessoin_factory(),
        sems=sems,
        home_url=config.LEARNCPP,
        progress=progress,
    )
    file_mgr = FileManager(config=config)
    pool = Pool(config.COMPUTE_PROCESS_MAX)

    app = Application(
        config=config,
        dl_service=dl_service,
        file_mgr=file_mgr,
        progress=progress,
        worker_pool=pool,
    )
    return app


async def main():
    config = Config.from_env()
    app = app_factory(config)
    async with app:
        await app.run()


if __name__ == "__main__":
    asyncio.run(main())
