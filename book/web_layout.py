import typing as ty

from selectolax.lexbor import LexborHTMLParser, LexborNode

from book.config import frozen
from book.errors import HTMLParsingError


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
        return cls(name=name, root=root)  # type: ignore


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
        return cls(no=chapter_no, title=title, link=link)  # type: ignore


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
        return cls(name=table_name, chapters=chapters)  # type: ignore


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
        return cls(root=content)  # type: ignore
