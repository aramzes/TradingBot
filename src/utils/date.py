
from datetime import timedelta

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def floor_dt(dt, delta):
    dt = dt - timedelta(minutes=dt.minute % delta,
        seconds=dt.second,
        microseconds=dt.microsecond
    ) 
    return dt