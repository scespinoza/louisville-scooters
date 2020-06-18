/*
    We should start designing some kind of headers 
    for the files.

    simTime.js
    -----------
    Script intended to be run as a worker that holds the time engine
    for a D3 visualization.
*/

let totalSimMillis, lastMillis, simRatio;
let mainClockInterval;


function dhm(t){
    var days = [
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ]
    var cd = 24 * 60 * 60 * 1000,
        ch = 60 * 60 * 1000,
        d = Math.floor(t / cd),
        h = Math.floor( (t - d * cd) / ch),
        m = Math.round( (t - d * cd - h * ch) / 60000),
        pad = function(n){ return n < 10 ? '0' + n : n; };
  if( m === 60 ){
    h++;
    m = 0;
  }
  if( h === 24 ){
    d++;
    h = 0;
  }
  return days[d] + ' ' + [pad(h), pad(m)].join(':');
}

// Listener expecting an Array with first elements
// a string and the rest parameters
self.addEventListener('message', function (event) {

    function IntervalFunction() {
        var now = new Date;
        var nowMillis = now.getTime();
        totalSimMillis += (nowMillis - lastMillis) * simRatio;
        lastMillis = nowMillis;
        var returnStr = dhm(totalSimMillis);
        self.postMessage(["timerUpdate", returnStr]);
    }
    
    data = event.data;

    if (data[0] === "timerInit") {
        var clock = new Date;
        lastMillis = clock.getTime();
        totalSimMillis = 0;

        simRatio = data[1];

        // Maybe every 100ms is overkill
        mainClockInterval = setInterval(IntervalFunction, 100);
    } else if (data[0] === "updateRatio") {
        simRatio = data[1];
    } else if (data[0] === "stopTimer") {
        clearInterval(mainClockInterval);
        mainClockInterval = null;
        self.postMessage(["timerUpdate", "00:00"]);
    } else if (data[0] === "continueTimer") {
        if (mainClockInterval == null) {
            mainClockInterval = setInterval(IntervalFunction, 100);
        }
    } else if (data[0] === "killTimer") {
        self.close();
    }
});

