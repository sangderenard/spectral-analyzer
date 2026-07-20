from camera_software import (
    OpenGLContextHost,
    OpenGLContextRequest,
    OpenGLContextSource,
)


def test_context_host_prefers_parent_and_never_destroys_it():
    parent = object()
    factory_calls = []
    host = OpenGLContextHost(
        parent,
        owned_factory=lambda request: factory_calls.append(request),
        current_probe=lambda: None,
    )

    lease = host.acquire(OpenGLContextRequest())
    host.close()

    assert lease.context is parent
    assert lease.source is OpenGLContextSource.PARENT
    assert not lease.owned
    assert factory_calls == []


def test_context_host_builds_and_releases_owned_fallback_once():
    released = []
    owned = object()
    host = OpenGLContextHost(
        owned_factory=lambda _request: (owned, lambda: released.append(owned)),
        current_probe=lambda: None,
    )

    lease = host.acquire(OpenGLContextRequest())
    lease.close()
    host.close()

    assert lease.context is owned
    assert lease.source is OpenGLContextSource.OWNED
    assert released == [owned]


def test_context_host_can_report_no_context_without_creating_one():
    host = OpenGLContextHost(current_probe=lambda: None)
    lease = host.acquire(OpenGLContextRequest(create_if_missing=False))

    assert not lease.available
    assert lease.source is OpenGLContextSource.UNAVAILABLE
    assert "no parent" in lease.error