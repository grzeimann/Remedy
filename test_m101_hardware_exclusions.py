import numpy as np

import diagnose_m101_hierarchical as legacy
from m101_hardware_exclusions import HARDWARE_EXCLUSIONS, exclusion_reason, hardware_excluded


def _excluded(**overrides):
    values = {"date": 20200521, "specid": 1, "ifuslot": 30,
              "ifuid": 1, "amp": "RL", "purpose": "fit"}
    values.update(overrides)
    return hardware_excluded(**values)


def test_half_open_date_interval_and_legacy_slot_amp_matching():
    assert _excluded(date=20200521, ifuslot=30)
    assert _excluded(date=20200521, ifuslot=30, purpose="cube")
    assert not _excluded(date=20200522, ifuslot=30)
    assert not _excluded(date=20200521, ifuslot=31, amp="RU")
    assert _excluded(date=20200525, ifuslot=39, amp="LL")


def test_full_physical_amplifier_identity_is_required():
    assert _excluded(date=None, specid=309, ifuslot=46, ifuid=5, amp="RL")
    for h5 in ("20200521_0000019.h5", "/other/path/20200528_0000015.h5"):
        assert _excluded(date=None, h5=h5, specid=309, ifuslot=46,
                        ifuid=5, amp="RL")
    assert not _excluded(date=None, specid=309, ifuslot=46, ifuid=6, amp="RL")
    assert not _excluded(date=None, specid=309, ifuslot=47, ifuid=5, amp="RL")
    assert _excluded(date=None, specid=402, ifuslot=77, ifuid=61, amp="RL")
    assert _excluded(date=None, specid=402, ifuslot=77, ifuid=61,
                    amp="RL", purpose="cube")


def test_scope_and_no_match_behavior():
    assert _excluded(date=None, specid=309, ifuslot=46, ifuid=5, amp="RL", purpose="cube")
    assert not _excluded(date=None, specid=309, ifuslot=46, ifuid=5, amp="LU", purpose="fit")
    assert not _excluded(date=20210101, specid=1, ifuslot=1, ifuid=1, amp="LL")
    assert exclusion_reason(date=20210101, specid=1, ifuslot=1,
                            ifuid=1, amp="LL", purpose="cube") is None


def test_fit_only_scope_is_not_used_for_cube():
    record = {"start": 20990101, "stop": 20990102, "ifuslot": 12,
              "amp": "LL", "scope": "fit_only", "reason": "test",
              "source": "test"}
    HARDWARE_EXCLUSIONS.append(record)
    try:
        values = {"date": 20990101, "specid": 1, "ifuslot": 12,
                  "ifuid": 1, "amp": "LL"}
        assert hardware_excluded(**values, purpose="fit")
        assert not hardware_excluded(**values, purpose="cube")
    finally:
        HARDWARE_EXCLUSIONS.remove(record)


def test_multiple_matches_report_all_reasons():
    record = {"start": 20200430, "stop": 20200501, "ifuslot": 57,
              "amp": "RL", "scope": "fit_and_cube", "reason": "second",
              "source": "test"}
    HARDWARE_EXCLUSIONS.append(record)
    try:
        reason = exclusion_reason(date=20200430, specid=1, ifuslot=57,
                                  ifuid=1, amp="RL", purpose="fit")
        assert reason is not None
        assert reason.count("[") == 2
        assert "legacy cube mask_dict" in reason
        assert "test" in reason
    finally:
        HARDWARE_EXCLUSIONS.remove(record)


def test_legacy_registry_matches_previous_masked_rows_on_representative_dates():
    for date in (20200430, 20200501, 20200521, 20200525, 20200622, 20200715):
        slots = np.asarray([21, 30, 39, 46, 57, 58, 89, 92, 94, 104, 1])
        amps = np.asarray(["RL", "LL", "RU", "RU", "RL", "RU", "RL",
                           "LU", "RU", "LL", "LL"])
        old = legacy.masked_rows("%d_0000001.h5" % date, slots, amps)
        new = np.asarray([
            hardware_excluded(date=date, specid=-1, ifuslot=slot, ifuid=-1,
                              amp=amp, purpose="fit")
            for slot, amp in zip(slots, amps)
        ])
        np.testing.assert_array_equal(new, old)


def test_exact_h5_physical_exclusion_matches_only_that_h5():
    record = {"h5": "20200528_0000015.h5", "specid": 123,
              "ifuslot": 46, "ifuid": 5, "amp": "RL",
              "scope": "fit_and_cube", "reason": "malformed H5",
              "source": "test"}
    HARDWARE_EXCLUSIONS.append(record)
    try:
        values = {"date": None, "specid": 123, "ifuslot": 46,
                  "ifuid": 5, "amp": "RL", "purpose": "fit"}
        assert hardware_excluded(**values, h5="/data/20200528_0000015.h5")
        assert not hardware_excluded(**values, h5="20200529_0000001.h5")
        assert not hardware_excluded(**values)
        assert "malformed H5" in exclusion_reason(
            **values, h5="20200528_0000015.h5")
    finally:
        HARDWARE_EXCLUSIONS.remove(record)


def test_exact_h5_exclusion_scope_respects_fit_and_cube():
    record = {"h5": "scope-test.h5", "specid": 123,
              "ifuslot": 46, "ifuid": 5, "amp": "RL",
              "scope": "fit_only", "reason": "fit test", "source": "test"}
    HARDWARE_EXCLUSIONS.append(record)
    try:
        values = {"date": None, "h5": "scope-test.h5", "specid": 123,
                  "ifuslot": 46, "ifuid": 5, "amp": "RL"}
        assert hardware_excluded(**values, purpose="fit")
        assert not hardware_excluded(**values, purpose="cube")
    finally:
        HARDWARE_EXCLUSIONS.remove(record)
