import asyncio
from threading import Thread
import time
from fastapi import FastAPI
from queue import Queue
from graphlib import 

# creating a fast application
app = FastAPI()

# initializing the queue
streamer_queue = Queue()

# fake data streamer
def put_data():
    some_i = 20
    for i in range(20):
        streamer_queue.put(some_i + i)

# Creation of thread 
def start_generation():
    thread = Thread(target=put_data)
    time.sleep(0.5)
    thread.start()


# This is an asynchronous function, as it has to wait for
# the queue to be available
async def serve_data():
    # Optinal code to start generation - This can be started anywhere in the code
    start_generation()

    while True:
        # Stopping the retreival process if queue gets empty
        if streamer_queue.empty():
            break
        # Reading and returning the value from the queue
        else:
            value = streamer_queue.get()
            yield str(value)
            streamer_queue.task_done()
        # Providing a buffer timer to the generator, so that we do not
        # break on a false alarm i.e generator is taking time to generate
        # but we are breaking because the queue is empty
        # 2 is an arbitrary number that can be changed based on the choice of
        # the developer
        await asyncio.sleep(2)