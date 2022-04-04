_FIXED_MOVING_REGEX_ = \
    r""" # Matches sth like "f0001_m0002"
    (?P<pre>.*?)
    (?P<combined_fm_id>
        f(?P<fixed_id>\d+) # Matches fixed part of id with named group "fixed_id"
        _m(?P<moving_id>\d+) # Matches moving part of id with named group "moving_id"
    )
    (?P<post>.*)
    """

_FORCED_ID_REGEX_ = r"""id_(?P<id>\d+)"""
_MULTIPLE_ID_REGEX_ = r"""[(\d{1,})]+"""
_FILE_REGEX_ = r".*\.*$"
_NIFTI_REGEX_ = r".*\.nii(\.gz)?$"