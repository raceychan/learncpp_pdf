import asyncio
import shutil
from dataclasses import dataclass
from functools import cached_property
from multiprocessing.pool import Pool
from pathlib import Path

import aiohttp
import pdfkit
import pypdf
from loguru import logger
from selectolax.lexbor import LexborHTMLParser, LexborNode

from src.config import Config

frozen = dataclass(frozen=True, slots=True, kw_only=True)


def namesort(name: str):
    """Used to sort filenames
    0 -> 9 -> "a" -> "z"
    """
    return (0, int(name)) if name.isdigit() else (1, name)


def html_to_pdf(
    html: Path, out: Path, options: dict = {"enable-local-file-access": ""}
):
    if out.exists():
        raise Exception("duplicated convertion should be avoided")

    post = Post.parse_html(name=html.stem, text=html.read_text())
    post.remove_elements()
    try:
        pdfkit.from_string(post.html, str(out), options=options)
    except IOError as ie:
        # TODO: make it in a file
        logger.error(f"Failed to convert {html.stem}, \n Error: {ie}")
    logger.success(f"html is converted to {out}")


def merge_chapters(pdfs: list[Path], out: Path):
    merger = pypdf.PdfWriter()
    for file in pdfs:
        merger.append(file)

    merger.write(out)
    merger.close()
    return out


@frozen
class Post:
    name: str
    root: LexborNode

    @property
    def html(self) -> str:
        assert self.root.html
        return self.root.html

    def remove_elements(self: "Post") -> None:
        prv_nxt = self.root.css_first(".entry-content > .prevnext")
        comments = self.root.css_first(".comments-area")
        if prv_nxt:
            prv_nxt.decompose()
        if comments:
            comments.decompose()

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
    def filename(self):
        no = self.no.replace(".", "_").replace(" ", "_")
        return f"{no}.html"

    @classmethod
    def from_node(cls, chapter_node: LexborNode):
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
    def from_node(cls, table_node: LexborNode):
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
    def parse_html(cls, text: str):
        outline_dom = LexborHTMLParser(text)
        if not outline_dom.body:
            raise ValueError("Failed to parse html")
        content = outline_dom.body.select("div.entry-content").matches[0]
        return cls(root=content)


class DownloadService:
    def __init__(
        self, session: aiohttp.ClientSession, sems: asyncio.Semaphore, home_url: str
    ):
        self._session = session
        self._sems = sems
        self._home_url = home_url

    async def get_content(self, url: str = "/"):
        async with self._sems:
            async with self._session.get(url) as response:
                res = await response.text()
        return res

    async def download_outline(self):
        html = await self.get_content(self._home_url)
        outline = OutLinePage.parse_html(html)
        return outline

    async def download_chapter(self, link: str, dst_f: Path) -> None:
        res = await self.get_content(link)
        dst_f.write_text(res)

    async def download_chapters(self, chapters_folder: Path):
        # we may want to cache outline.html
        outline = await self.download_outline()
        todo = set()
        for table in outline.content_tables():
            table_folder = chapters_folder / table.name
            table_folder.mkdir(exist_ok=True)
            # TODO: refactor, this logic looks weird
            for chapter in table.chapters:
                dst_f = table_folder / f"{chapter.filename}"
                if dst_f.exists():
                    continue
                task = asyncio.create_task(self.download_chapter(chapter.link, dst_f))
                todo.add(task)

        await asyncio.gather(*todo)
        logger.success("all chapters downloaded")

    async def close(self):
        await self._session.close()


class FileManager:
    def __init__(self, config: Config):
        self._config = config
        self.__setup()

    def __setup(self):
        self._config.HTML_FOLDER.mkdir(exist_ok=True)
        self._config.HTML_CHAPTER.mkdir(exist_ok=True)
        self._config.PDF_FOLDER.mkdir(exist_ok=True)
        self._config.PDF_CHAPTER.mkdir(exist_ok=True)
        self._config.PDF_BOOK_FOLDER.mkdir(exist_ok=True)

    @property
    def chapter_folders(self):
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

    def sorted_dir_pairs(self) -> list[tuple[Path, Path]]:
        res: list[tuple[Path, Path]] = []
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

    def remove_cached(self):
        shutil.rmtree(self._config.HTML_FOLDER)
        shutil.rmtree(self._config.PDF_CHAPTER)


class Application:
    def __init__(
        self,
        *,
        config: Config,
        dl_service: DownloadService,
        file_mgr: FileManager,
        worker_pool: Pool,
    ):
        self._config = config
        self._dl_service = dl_service
        self._file_mgr = file_mgr
        self._worker_pool = worker_pool

    @cached_property
    def config(self):
        return self._config

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc, excval, traceback):
        await self.close()
        logger.info("Application Exit")

    async def download_chapters(self):
        await self._dl_service.download_chapters(self._config.HTML_CHAPTER)

    def convert_to_pdf(self):
        tasks = [
            self._worker_pool.apply_async(html_to_pdf, args=(src_f, dst_f))
            for src_f, dst_f in self._file_mgr.sorted_dir_pairs()
        ]
        res = [task.get() for task in tasks]
        logger.success("files converted")
        return res

    def save_book_to(self, dst: Path):
        tasks = []
        for pdf_dirs in self._file_mgr.sorted_dst_dirs():
            chapter_idx = pdf_dirs[0].parent.stem.split("Chapter")[1]
            dst_f = dst / f"chapter_{chapter_idx}.pdf"
            task = self._worker_pool.apply_async(merge_chapters, args=(pdf_dirs, dst_f))
            tasks.append(task)

        merged = [task.get() for task in tasks]
        learncpp = merge_chapters(merged, dst / "learncpp.pdf")
        logger.success(f"files merged and saved to{learncpp}")
        return learncpp

    async def run(self):
        await self.download_chapters()
        self.convert_to_pdf()
        self.save_book_to(self._config.PDF_BOOK_FOLDER)
        self._file_mgr.remove_cached()

    async def close(self):
        self._worker_pool.close()
        await self._dl_service.close()


def sessoin_factory():
    timeout = aiohttp.ClientTimeout(total=60, connect=10)
    session = aiohttp.ClientSession()
    return session


def app_factory(config: Config):
    sems = asyncio.Semaphore(config.DOWNLOAD_CONCURRENT_MAX)
    dl_service = DownloadService(
        session=sessoin_factory(), sems=sems, home_url=config.LEARNCPP
    )
    pool = Pool(config.COMPUTE_PROCESS_MAX)
    file_mgr = FileManager(config=config)
    app = Application(
        config=config,
        dl_service=dl_service,
        file_mgr=file_mgr,
        worker_pool=pool,
    )
    return app


async def main():
    config = Config()
    app = app_factory(config)
    async with app:
        await app.run()


if __name__ == "__main__":
    asyncio.run(main())
