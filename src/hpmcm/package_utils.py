import os
import urllib
import subprocess


def setupTestDataArea() -> int:  # pragma: no cover
    """Download test files to setup a project testsing area

    Returns
    -------
    int:
       0 for success, error code otherwise

    Notes
    -----
    This will download files into 'examples/test_data', and could take a few
    minutes.

    This will not download the files if they are already present
    """
    if not os.path.exists("examples/test_data"):

        if not os.path.exists("examples/test_data.tgz"):
            urllib.request.urlretrieve(
                "http://s3df.slac.stanford.edu/people/echarles/package_test_data/hpmcm/hpmcm_test_data.tgz",
                "examples/test_data.tgz",
            )
            if not os.path.exists("examples/test_data.tgz"):
                return 1
        
        status = subprocess.run(
            ["tar", "zxvf", "examples/test_data.tgz", "-C", "examples"], check=False
        )
        if status.returncode != 0:
            return status.returncode

    if not os.path.exists("examples/test_data/object_10463.pq"):
        return 2

    return 0


def teardownTestDataArea() -> None:  # pragma: no cover
    if not os.environ.get("NO_TEARDOWN"):
        pass
        #os.system("\\rm -rf examples/test_data")
        #try:
        #    os.unlink("examples/test_data.tgz")
        #except FileNotFoundError:
        #    pass
