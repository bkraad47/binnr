from celery import Celery

celery = Celery(__name__, broker='redis://redis:6379/0', backend='redis://redis:6379/0')

@celery.task
def long_running_job(param1: str, param2: int):
    # Simulate a long-running job
    import time
    time.sleep(param2)
    return f"Processed {param1} with delay of {param2} seconds"
