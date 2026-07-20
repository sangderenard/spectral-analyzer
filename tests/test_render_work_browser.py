from types import SimpleNamespace

from bass_viewer import ScrollableSubpanelList
from render_work_browser import RenderWorkBrowser


def test_work_browser_directly_uses_repository_scrolling_widget():
    browser = RenderWorkBrowser()

    assert type(browser.widget) is ScrollableSubpanelList


def test_work_browser_lists_jobs_completed_assets_and_selects_preview(tmp_path):
    preview = tmp_path / "actually.png"
    preview.write_bytes(b"preview")
    request = SimpleNamespace(
        request_key="request-1",
        token_asset=SimpleNamespace(token="pending"),
        target_kind=SimpleNamespace(value="token_sequence"),
        refinement_pass=4,
    )
    artifact = SimpleNamespace(
        preview_path=str(preview), linear_path="", samples=64
    )
    subtype = SimpleNamespace(
        subtype_key="subtype-1",
        text="actually",
        kind="token",
        complete=True,
        artifacts=(artifact,),
    )
    bundle = SimpleNamespace(
        display_name="DejaVu Sans Mono / red ink",
        subtypes=(subtype,),
    )
    browser = RenderWorkBrowser()

    browser.sync(
        [request], [bundle], active=("style-1", "A", 7, True)
    )

    titles = [spec.title for spec in browser.widget.subpanels]
    assert titles == ["ACTIVE  A", "QUEUED  pending", "DONE  actually"]
    browser.widget.selected_key = "asset:subtype-1"
    assert browser.selected_preview_path == str(preview)
    assert browser.selected_payload["subtype"] is subtype