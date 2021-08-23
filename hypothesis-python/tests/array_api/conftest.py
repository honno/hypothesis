import pytest

from tests.array_api.xputils import COMPLIANT_XP, MOCK_NAME


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "mockable_xp: mocked array module may be used in test"
    )


def pytest_collection_modifyitems(config, items):
    if not COMPLIANT_XP:
        mark = pytest.mark.filterwarnings(
            f"ignore:.*determine.*{MOCK_NAME}.*Array API.*"
        )
        for item in items:
            if "mockable_xp" in item.keywords:
                item.add_marker(mark)
