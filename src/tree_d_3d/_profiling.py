"""The scope-profiler tag of the library's profiling regions.

Every region is named ``"tree_d_3d.<module>.<operation>"`` and tagged with
:data:`PROFILE_TAG`, which :mod:`tree_d_3d` registers as opt-in: the regions record
nothing unless the application enables the tag, for example with
``scope_profiler.setup(enable_tags=["tree_d_3d"])``.
"""

#: The opt-in tag of the profiling regions, also the prefix of their names.
PROFILE_TAG = "tree_d_3d"

#: The ``tags`` of every profiling region.
_PROFILE_TAGS = (PROFILE_TAG,)
