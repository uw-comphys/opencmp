## README

This folder contains the scripts for the pytest unit tests and integration tests.



### Running Tests

**!!! Important Note !!!**

Your working directory is assumed to be `/path/to/opencmp`.
If you call pytest from within `opencmp/tests` or elsewhere any tests that use relative file paths (most notably the integration tests) will fail.



pytest can be used to run the tests in a specific file

` pytest <file>`

pytest can also be used to run the tests in any file whose name includes "test"

`pytest`

However, the above command will skip running any tests that are marked as "slow" (ex: transient examples that take several minutes to run).
Include these tests as follows

`pytest --runslow`

It is also recommended that the package `pytest-xdist` be installed using `pip install pytest-xdist`.
This allows for the tests to be run in parallel over `N` threads using the following command `pytest -n X`.


### Creating New Tests

pytest will automatically find any files whose names include "test" and will run any functions whose names begin with "test" (unless marked as "slow"). 

Markers can be added to test functions with the decorator

`@pytest.mark.<name>`

Currently the only defined marker is "slow", which indicates tests that are too slow to run by default. New markers should be defined in conftest.py.

