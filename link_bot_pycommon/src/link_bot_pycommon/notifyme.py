import logging
import pathlib
from time import perf_counter
from typing import Callable, Optional

import boto3

logger = logging.getLogger(__file__)


class JobNotifier:

    def __init__(self, phone_number: str):
        if phone_number[0] != '1':
            logger.warning("Phone number should start with 1")
        self.phone_number = phone_number

    def notify(self, msg: str, subject: str = 'Job Notification'):
        client = boto3.client('sns')
        response = client.publish(PhoneNumber=self.phone_number, Message=msg, Subject=subject)
        status_code = response['ResponseMetadata']['HTTPStatusCode']
        if status_code != 200:
            logger.error(f"Status code {status_code}")

    def success(self, job_name: str):
        self.notify(f"Job {job_name} completed successfully", "Success!")

    def failure(self, job_name: str):
        self.notify(f"Job {job_name} raised an exception", "Exception")


def notify(phone_number: Optional[str] = None, ignore_before: int=5):
    """
    A decorator for sending a text message when a function completes successfully (or not)

    @notifyme.notify("6094335864")
    def main():
        # long fragile code here

    Args:
        phone_number: phone number, must start with a 1
        ignore_before: don't send a message if an exception is thrown in under this many seconds

    Returns:
    """
    if phone_number is None:
        with (pathlib.Path.home() / ".phone_number").open() as phone_number_file:
            phone_number = phone_number_file.readline()

    notifier = JobNotifier(phone_number)

    def _notify(func: Callable):
        def wrapper(*args, **kwargs):
            t0 = perf_counter()
            try:
                func(*args, **kwargs)
                notifier.success(job_name=func.__name__)
            except Exception as e:
                print(e)
                dt = perf_counter() - t0
                if dt >= ignore_before:
                    notifier.failure(job_name=func.__name__)

        return wrapper

    return _notify


@notify('16094335864')
def test_succeeding_job():
    from time import sleep
    sleep(1)


@notify('16094335864')
def test_failing_job():
    from time import sleep
    sleep(1)
    raise RuntimeError("failure")


if __name__ == '__main__':
    test_succeeding_job()
    test_failing_job()
