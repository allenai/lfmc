import calendar
from datetime import date, timedelta

DEFAULT_PADDING = timedelta(days=30)


def pad_dates(end_date: date, padding: timedelta = DEFAULT_PADDING) -> tuple[date, date]:
    new_end_date = end_date + padding
    last_day_of_month = calendar.monthrange(new_end_date.year, new_end_date.month)[1]
    new_end_date = date(new_end_date.year, new_end_date.month, last_day_of_month)
    start_date = date(new_end_date.year - 1, new_end_date.month, 1)
    return start_date, new_end_date
