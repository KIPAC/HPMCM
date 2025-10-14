**********************
Contribution Overview
**********************

HPMCM welcomes all interested developers, regardless of DESC membership or LSST data rights.

=========================
Contributing to ``HPMCM``
=========================

If you're interested in contributing to `HPMCM`, but don't know where to start, take a look 
at the
`list of issues <https://github.com/KIPAC/HPMCM/issues>`_.
Or, `create a new issue <https://github.com/KIPAC/HPMCM/issues/new>`_ to 
suggest a change.

====================
Where to contribute:
====================

In all cases, begin by following the developer installation instructions 
:ref:`Developer Installation` and follow the contribution workflow instructions below.

     
=====================
Contribution workflow
=====================

The ``HPMCM`` repository use an issue-branch-review workflow, 
similar to the standard `GitHub Flow <https://docs.github.com/en/get-started/quickstart/github-flow>`_.
We typically use ``git`` as our version control tool, there are many resources
available online, but here is a `nice cheat sheet <https://education.github.com/git-cheat-sheet-education.pdf>`_
created by GitHub.

-----
Issue
-----

=======
When you identify something that should be done, `make an issue <https://github.com/KIPAC/HPMCM/issues/new>`_
for it.

------
Branch
------

See :ref:`Developer Installation` for installation instructions.

While developing in a branch, don't forget to pull from ``main`` regularly (at 
least daily) to make sure your work is compatible with other recent changes.

When you're ready to merge your branch into the ``main`` branch, create a pull request
("PR") in the rail repository you cloned from. GitHub has instructions 
`here <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_.

Several continuous integration checks will be performed for new pull requests. 
If any of these automatic processes find issues with the code, you should address 
them in the branch before sending for review. These include unit tests (does the 
code function correctly), pylint (code style), or coverage (how much code is 
exercised in unit tests).

Once you are satisfied with your PR, request that other team members review and 
approve it. You could send the request to someone whom you've worked with on the 
topic, or one of the core maintainers of rail.


Merge
-----

Once the changes in your PR have been approved, these are your next steps:

1. the author merges the change by selecting "Squash and merge" on the approved pull request
2. enter ``closes #[#]`` in the comment field to close the resolved issue
3. delete your branch using the button on the merged pull request.



Reviewing a PR
--------------

To review a pull request, it's a good idea to start by pulling the changes and 
running the unit tests locally. If the continuous integration tests have run 
successfully, there is good hope that the unit tests will run locally as well! 

Check the code for complete and accurate docstrings, sufficient comments, and 
ensure any instances of ``#pragma: no cover`` (excluding the code from unit test 
coverage accounting) are extremely well-justified.

Feel free to mark the PR with “Request changes” for necessary changes. e.g. 
writing an exception for an edge case that will break the code, updating names 
to adhere to the naming conventions, etc.

It is also considered good practice to make suggestions for optional improvements, 
such as adding a one-line comment before a clever block of code or including a 
demonstration of new functionality in the example notebooks.


Naming conventions
------------------

We follow the `pep8 <https://peps.python.org/pep-0008/#descriptive-naming-styles>`_ 
recommendations for naming new modules.


Modules
^^^^^^^

Modules should use all lowercase, with underscores where it aids the readability
of the module name. 


Classes
^^^^^^^

Python classes and so should use the CapWords convention.

