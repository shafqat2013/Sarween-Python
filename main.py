import setup as s
import calibration as c
import tracking as t
import gameplay as g
from datetime import datetime

timestamp = str(datetime.now().strftime("%Y%m%d_%H%M%S")
)


if __name__ == "__main__":
    #print('running s.initialize. Timestamp: ' + timestamp)
    s.initialize()
    #print('running c.calibrate. Timestamp: ' + timestamp)
    c.calibrate()
    #print('running t.begin_session. Timestamp: ' + timestamp)
    t.begin_session()