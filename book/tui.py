import argparse
import typing as ty
from collections import defaultdict

import rich
import rich.pretty
import textual
import textual.screen
from textual import containers as tc
from textual import widgets as tw
from textual.app import App, ComposeResult
from textual.widgets.selection_list import Selection

SelectedOptions = dict[str, ty.Any]

MENUAL_CSS = """
Screen {
    align: center middle;
    background: $surface;
}

#home-layout {
    width: 100%;
    height: 100%;
    background: $surface;
    border: round $primary;
    padding: 1 2;
}

#home-body-scroll {
    width: 100%;
    height: auto;
    background: $surface;
    padding: 1 2;
}

Horizontal {
    width: 50%;
    height: 50%;
    background: $panel;
    border: round $primary;
    padding: 1 2;
}

SelectionList {
    width: 100%;
    height: 100%;
    border: round $accent;
    margin-right: 1;
}

VerticalScroll {
    width: 90%;
    height: auto;
    background: $boost;
    border: round $primary;
    margin-top: 1;
    padding: 1 2;
}

Label {
    margin-top: 1;
    margin-bottom: 1;
}

Input {
    width: 50%;
    margin-bottom: 2;
}

PreviewModal {
    align: center middle;
}

#preview-container {
    width: 80%;
    height: 80%;
    border: round $accent;
    background: $surface;
}

#preview-title {
    dock: top;
    height: 3;
    background: $accent;
    color: $text;
    content-align: center middle;
}

#preview-content {
    padding: 1;
}


Button.sign{
    width: 25%;
    height: 45%;
}
"""


def split_actions(
    heter_list: list[argparse.Action],
) -> defaultdict[type[argparse.Action], list[argparse.Action]]:
    action_mapping = defaultdict(list)
    actions = (
        argparse._StoreAction,
        argparse._StoreConstAction,
        argparse._AppendAction,
        argparse._AppendConstAction,
        argparse._CountAction,
    )
    while heter_list:
        item = heter_list.pop(0)
        for action in actions:
            if isinstance(item, action):
                action_mapping[action].append(item)

    if heter_list:
        raise Exception("Uncategorized action")

    return action_mapping


def widget_builder(action_groups: dict):
    def type_cvt(param_type: ty.Any):
        if param_type is str or param_type is None:
            return "text"
        elif param_type is int:
            return "integer"
        elif param_type is float:
            return "number"
        else:
            raise Exception(f"Unknown type {param_type}")

    selections = action_groups[argparse._StoreConstAction]
    store_actions = action_groups[argparse._StoreAction]
    append_actions = action_groups[argparse._AppendAction]
    append_const_actions = action_groups[argparse._AppendConstAction]
    count_actions = action_groups[argparse._CountAction]

    home_layout = tc.Vertical(id="home-layout")
    scroll_body = tc.VerticalScroll(
        tw.Static(""),
        id="home-body-scroll",
    )
    with home_layout:
        with scroll_body:
            # StoreConstAction
            selection_list = [
                Selection(
                    prompt=action.dest, value=action.dest, initial_state=action.const
                )
                for action in selections
            ]
            with tc.Horizontal(id="home-selections"):
                yield tw.SelectionList(*selection_list)
                yield tw.Static("pymenual-app", id="home-app-info")

            # StoreAction
            for val in store_actions:
                yield tw.Label(val.help or "")
                yield tw.Input(
                    placeholder=val.dest,
                    value=str(val.default),
                    type=type_cvt(val.type),
                )

            # AppendAction
            for action in append_actions:
                yield tw.Label(action.help or "")
                yield tw.Input(
                    placeholder=f"{action.dest} (comma-separated)",
                    value=str(action.default),
                    type=type_cvt(action.type),
                )

            # AppendConstAction
            for action in append_const_actions:
                yield tw.Label(action.help or "")
                yield tw.Checkbox(action.dest)

            # CountAction
            for action in count_actions:
                yield tw.Label(action.help or "")
                yield tw.Input(placeholder=action.dest, value=str(action.default or 0))


class PreviewModal(textual.screen.ModalScreen):
    BINDINGS = [("ctrl+p", "app.pop_screen", "Close")]

    def __init__(
        self,
        name: ty.Optional[str] = None,
        id: ty.Optional[str] = None,
        classes: ty.Optional[str] = None,
        options: dict[str, ty.Any] = {},
    ):
        super().__init__(name=name, id=id, classes=classes)
        self._options = options

    def compose(self) -> ComposeResult:
        yield tc.Container(
            tw.Static("Preview", id="preview-title"),
            tw.Static(id="preview-content"),
            id="preview-container",
        )
        yield tw.Footer()

    def on_mount(self) -> None:
        self.styles.align = ("center", "middle")
        self.update_content(self._options)

    def update_content(self, content: dict) -> None:
        preview: tw.Static = self.query_one("#preview-content")
        preview.update(rich.pretty.Pretty(content))


class Menual(App[SelectedOptions]):
    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit", "Quit"),
        ("ctrl+o", "confirm", "Confirm Options"),
        ("tab", "next_widget", "Next widget"),
        ("shift+tab", "previous_widget", "Previous widget"),
        ("ctrl+p", "toggle_preview", "Toggle Preview"),
    ]

    CSS = MENUAL_CSS

    def __init__(
        self,
        driver_class: None = None,
        css_path: None = None,
        watch_css: bool = False,
        parser: ty.Optional[argparse.ArgumentParser] = None,
    ):
        super().__init__(
            driver_class=driver_class,
            css_path=css_path,
            watch_css=watch_css,
        )
        self.__action_groups = split_actions(parser._actions)

    async def action_next_widget(self) -> None:
        current = self.screen.focused
        widgets = tuple(self.query("SelectionList, Input, CheckBox").results())
        if current in widgets:
            index = widgets.index(current)
            next_widget = widgets[(index + 1) % len(widgets)]
            self.set_focus(next_widget)

    async def action_previous_widget(self) -> None:
        current = self.screen.focused
        widgets = tuple(self.query("SelectionList, Input, CheckBox").results())
        if current in widgets:
            index = widgets.index(current)
            previous_widget = widgets[(index - 1) % len(widgets)]
            self.set_focus(previous_widget)

    def compose(self) -> ComposeResult:
        yield tw.Header(name="pymenual", show_clock=True)
        # with tc.Container():
        for widget in widget_builder(self.__action_groups):
            yield widget
        yield tw.Footer()

    async def on_mount(self) -> None:
        self.query_one(tw.SelectionList).border_title = "Choose options"
        self.set_focus(self.query_one(tw.SelectionList))

    def _get_widgets_options(self) -> SelectedOptions:
        options: SelectedOptions = {}

        # Get SelectionList options
        selection_list = self.query_one(tw.SelectionList)
        for selection in selection_list.selected:
            options[selection] = True

        # Get Input widget values
        for input_widget in self.query(tw.Input):
            if input_widget.value:  # Only add non-empty inputs
                options[input_widget.placeholder] = input_widget.value

        for checkbox_widget in self.query(tw.Checkbox):
            if checkbox_widget.value:
                options[str(checkbox_widget.label)] = checkbox_widget.value
        return options

    async def action_toggle_preview(self) -> None:
        self._preview_model = PreviewModal(options=self._get_widgets_options())
        await self.push_screen(self._preview_model)

    async def action_confirm(self):
        options = self._get_widgets_options()
        self.exit(result=options, return_code=0)

    @classmethod
    def from_argparse(cls, parser: argparse.ArgumentParser):
        app = Menual(parser=parser)
        return app
