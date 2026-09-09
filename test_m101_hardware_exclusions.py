import numpy as np

import diagnose_m101_hierarchical as legacy
from m101_hardware_exclusions import exclusion_reason, hardware_excluded


def _excluded(**overrides):
    values = {"date": 20200521, "specid": 1, "ifuslot": 30,
              "ifuid": 1, "amp": "RL", "purpose": "fit"}
    values.update(overrides)
    return hardware_excluded(**values)


def test_half_open_date_interval_and_legacy_slot_amp_matching():
    assert _excluded(date=20200521, ifuslot=30)
    assert not _excluded(date=20200522, ifuslot=30)
    assert not _excluded(date=20200521, ifuslot=30, amp="RU")
    assert _excluded(date=20200525, ifuslot=39, amp="LL")


def test_full_physical_amplifier_identity_is_required():
    assert _excluded(date=None, specid=309, ifuslot=46, ifuid=5, amp="RL")
    assert not _excluded(date=None, specid=309, ifuslot=46, ifuid=6, amp="RL")
    assert not _excluded(date=None, specid=309, ifuslot=47, ifuid=5, amp="RL")
    assert _excluded(date=None, specid=402, ifuslot=77, ifuid=61, amp="RL")


def test_scope_and_no_match_behavior():
    assert _excluded(date=None, specid=309, ifuslot=46, ifuid=5, amp="RL", purpose="cube")
    assert not _excluded(date=None, specid=309, ifuslot=46, ifuid=5, amp="LU", purpose="fit")
    assert not _excluded(date=20210101, specid=1, ifuslot=1, ifuid=1, amp="LL")


def test_multiple_matches_report_all_reasons():
    reason = exclusion_reason(date=20200521, specid=309, ifuslot=46,
                              ifuid=5, amp="RL", purpose="fit")
    assert reason is not None
    assert reason.count("[") == 2
    assert "legacy cube mask_dict" in reason
    assert "M101 persistent amplifier residual QA" in reason


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
