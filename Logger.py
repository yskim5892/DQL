from time import gmtime, strftime

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        

    def log(self, msg):
        timestr = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        msg = '[' + timestr + '] ' + msg

        print(msg)
        with open(self.log_dir, "a") as f:
            f.write(msg)
