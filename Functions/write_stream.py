import time

def user_data(function_name):
    for word in function_name.split():
        yield word + " "
        time.sleep(0.02)
