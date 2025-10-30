import pytest

from hpmcm import package_utils


@pytest.fixture(name="setup_data", scope="package")
def setupData(request: pytest.FixtureRequest) -> int:
    """Download test data and unpack it"""
    ret_val = package_utils.setupTestDataArea()

    request.addfinalizer(package_utils.teardownTestDataArea)

    return ret_val
