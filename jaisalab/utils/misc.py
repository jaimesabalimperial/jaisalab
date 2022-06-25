from datetime import datetime 

def get_time_stamp_as_string():
    """Get the current time stamp as a string.
    
    Returns:
        date_time_str (str) : current timestemp
    """
    # Return current timestamp in a saving friendly format.
    date_time = datetime.now()
    date_time_str = date_time.strftime("%d-%b-%Y (%H-%M-%S)")
    return date_time_str
