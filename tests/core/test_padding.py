from datetime import date, timedelta

import pytest

from lfmc.core.padding import pad_dates


@pytest.mark.parametrize(
    "end_date, padding, expected_start_date, expected_end_date",
    [
        (date(2021, 1, 1), timedelta(days=30), date(2020, 1, 1), date(2021, 1, 31)),
        (date(2021, 1, 1), timedelta(days=31), date(2020, 2, 1), date(2021, 2, 28)),
        (date(2021, 2, 1), timedelta(days=30), date(2020, 3, 1), date(2021, 3, 31)),
        (date(2021, 3, 15), timedelta(days=30), date(2020, 4, 1), date(2021, 4, 30)),
        (date(2021, 3, 31), timedelta(days=30), date(2020, 4, 1), date(2021, 4, 30)),
        (date(2020, 12, 31), timedelta(days=30), date(2020, 1, 1), date(2021, 1, 31)),
    ],
)
def test_pad_dates(end_date: date, padding: timedelta, expected_start_date: date, expected_end_date: date):
    assert pad_dates(end_date, padding) == (expected_start_date, expected_end_date)
