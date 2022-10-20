
from datetime import datetime, timedelta


def get_date_range(
        first_date: str,
        last_date: str,
        fmt: str = '%Y-%m-%d'
):
    d0 = datetime.strptime(first_date, fmt)
    d1 = datetime.strptime(last_date, fmt)
    out = []
    for i in range((d1 - d0).days + 1):
        out.append(
            (d0 + timedelta(i)).strftime(fmt)
        )
    return out
