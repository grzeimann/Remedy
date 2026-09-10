"""Shared M101 hardware exclusions for calibration and future cube building.

Date/slot/amplifier records preserve the historical cube ``mask_dict`` with
half-open date intervals.  Persistent records use the complete physical
amplifier identity ``(SPECID, IFUSLOT, IFUID, AMP)``.  Exact-H5 records add
the H5 basename to that identity for a reduction-specific failure.

The eventual cube builder must call ``hardware_excluded(..., purpose="cube")``
and use this registry as its single source of truth.  Diagnostic residual
outliers are deliberately not added here automatically.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path


_LEGACY_MASKS = {
    "20200430-20200501": [
        "057RL", "057RU", "057LL", "057LU", "058RU", "058RL",
        "021RL", "021RU", "021LL", "021LU",
    ],
    "20200430-20200715": [
        "092LU", "094RU", "046RU", "104LL", "104LU", "028RU",
        "028RL", "027LL", "027LU", "067LL", "067LU", "025LU",
        "106RU", "106RL", "026LL", "026LU", "026RL", "026RU",
        "103LL", "103LU",
    ],
    "20200525-20200526": ["057RL", "057LL", "089RU", "089RL"],
    "20200517-20200522": ["030RL", "030RU", "030LL", "030LU"],
    "20200523-20200526": ["039RL", "039RU", "039LL", "039LU"],
    "20200622-20200715": ["089RL", "089RU", "096RL", "096RU"],
}


def _legacy_records():
    records = []
    for interval, names in _LEGACY_MASKS.items():
        start, stop = (int(value) for value in interval.split("-"))
        for name in names:
            records.append({
                "start": start,
                "stop": stop,
                "ifuslot": int(name[:3]),
                "amp": name[3:],
                "scope": "fit_and_cube",
                "reason": "historical date-dependent slot/amplifier exclusion",
                "source": "legacy cube mask_dict",
            })
    return records


HARDWARE_EXCLUSIONS = _legacy_records() + [
    {
        "specid": 309,
        "ifuslot": 46,
        "ifuid": 5,
        "amp": "RL",
        "scope": "fit_and_cube",
        "reason": "persistent malformed extracted spectra; likely upstream reduction/wavelength failure",
        "source": "M101 persistent amplifier residual QA",
    },
    {
        "specid": 402,
        "ifuslot": 77,
        "ifuid": 61,
        "amp": "RL",
        "scope": "fit_and_cube",
        "reason": "persistent malformed extracted spectra; likely upstream reduction/wavelength failure",
        "source": "M101 persistent amplifier residual QA",
    },
    {
        "specid": 412,
        "ifuslot": 13,
        "ifuid": 43,
        "amp": "RL",
        "scope": "fit_and_cube",
        "reason": "persistent malformed amplifier behavior in full 19-H5 M101 residual QA; likely upstream hardware/reduction/calibration failure",
        "source": "M101 full 19-H5 amplifier residual QA",
    },
    {
        "specid": 12,
        "ifuslot": 106,
        "ifuid": 33,
        "amp": "LL",
        "scope": "fit_and_cube",
        "reason": "persistent malformed amplifier behavior in full 19-H5 M101 residual QA; likely upstream hardware/reduction/calibration failure",
        "source": "M101 full 19-H5 amplifier residual QA",
    },
    {
        "specid": 12,
        "ifuslot": 106,
        "ifuid": 33,
        "amp": "LU",
        "scope": "fit_and_cube",
        "reason": "persistent malformed amplifier behavior in full 19-H5 M101 residual QA; likely upstream hardware/reduction/calibration failure",
        "source": "M101 full 19-H5 amplifier residual QA",
    },
]


def _date_number(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return int(value.strftime("%Y%m%d"))
    text = str(value).strip()
    if "-" in text and len(text) == 10:
        return int(text.replace("-", ""))
    return int(value)


def _matches(record, *, date_number, h5, specid, ifuslot, ifuid, amp, purpose):
    if record["scope"] == "fit_only" and purpose != "fit":
        return False
    if record["scope"] == "fit_and_cube" and purpose not in {"fit", "cube"}:
        return False
    if "h5" in record:
        return (h5 is not None
                and Path(str(h5)).name == record["h5"]
                and int(specid) == record["specid"]
                and int(ifuslot) == record["ifuslot"]
                and int(ifuid) == record["ifuid"]
                and str(amp).upper() == record["amp"])
    if "start" in record:
        return (date_number is not None and record["start"] <= date_number < record["stop"]
                and int(ifuslot) == record["ifuslot"]
                and str(amp).upper() == record["amp"])
    return (int(specid) == record["specid"]
            and int(ifuslot) == record["ifuslot"]
            and int(ifuid) == record["ifuid"]
            and str(amp).upper() == record["amp"])


def _matching_records(*, date, h5, specid, ifuslot, ifuid, amp, purpose):
    date_number = _date_number(date)
    if purpose not in {"fit", "cube"}:
        raise ValueError("purpose must be 'fit' or 'cube'")
    return [record for record in HARDWARE_EXCLUSIONS
            if _matches(record, date_number=date_number, h5=h5, specid=specid,
                        ifuslot=ifuslot, ifuid=ifuid, amp=amp,
                        purpose=purpose)]


def hardware_excluded(*, date, specid, ifuslot, ifuid, amp, purpose, h5=None):
    """Return whether an amplifier is excluded for ``fit`` or ``cube``."""
    return bool(_matching_records(date=date, h5=h5, specid=specid, ifuslot=ifuslot,
                                  ifuid=ifuid, amp=amp, purpose=purpose))


def exclusion_reason(*, date, specid, ifuslot, ifuid, amp, purpose, h5=None):
    """Return compact reason/provenance text for all matching exclusions."""
    records = _matching_records(date=date, h5=h5, specid=specid, ifuslot=ifuslot,
                                ifuid=ifuid, amp=amp, purpose=purpose)
    if not records:
        return None
    return "; ".join("%s [%s]" % (record["reason"], record["source"])
                     for record in records)
