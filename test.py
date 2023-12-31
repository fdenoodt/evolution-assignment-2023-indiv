import asyncio
import time


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


@background
def your_function(argument, other_argument):  # Added another argument
    time.sleep(argument)
    print(f"function finished for {argument=} and {other_argument=} \n")


def code_to_run_before():
    print('This runs Before Loop!')


def code_to_run_after():
    print('This runs After Loop!')


def main():
    code_to_run_before()  # Anything you want to run before, run here!

    loop = asyncio.get_event_loop()  # Have a new event loop
    looper = asyncio.gather(*[
        your_function(i, 1) for i in range(1, 5)
    ])  # Run the loop
    results = loop.run_until_complete(looper)  # Wait until finish

    code_to_run_after()


if __name__ == '__main__':
    main()
