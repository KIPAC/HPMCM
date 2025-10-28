import pytest

from hpmcm import package_utils


@pytest.fixture(name="setup_data", scope="package")
def setup_data(request: pytest.FixtureRequest) -> int:
    ret_val = package_utils.setupTestDataArea()

    request.addfinalizer(package_utils.teardownTestDataArea)

    return ret_val
