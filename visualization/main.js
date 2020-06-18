
// Global variables to project data
let newLayer,
    actualLayer,
    outerBounds,
    transform,
    arc,
    d3Path,
    percFormat,
    map,
    svg,
    statsSVG,
    g;

let bikeEdgesCollection,
    walkEdgesCollection,
    tripsCollection,
    bikeNodesCollection,
    walkNodesCollection,
    scooterLocations;

let simRatio = 1000; // real ms / sim ms
let satisfiedRequests = 0;
let totalRequests = 0;
let serviceLevelStats = [{hour:toDateTime(0), serviceLevel:1}];
let currentTime = 0;

function getCurrentServiceLevel() {
    if (totalRequests > 0) {
        return satisfiedRequests / totalRequests;
    } else {
        return 0;
    }  
}

var batteryColorScale = d3.scaleLinear().domain([0, 100])
                .range(["red", "green"]);

loadMap();
loadLegend();
createInfo();
loadStatsGraph();
loadData();
reset();
map.on('zoom', reset);
map.on('viewreset', reset);


var statsLine = d3.line()
        .x(function(d) {
            return  xScale(d.hour);
        })
        .y(function (d) {
            return yScale(d.serviceLevel);
        });

/*
// -- Worker script test --
//Testing if the browser supports workers
if (typeof (Worker) !== "undefined") {
    var timerWorker = new Worker("simTime.js");
} else {
    //  TODO Stop everything because it's not gonna work
    //  as intended.
}

timerWorker.addEventListener("message", function (event) {
    data = event.data;
    if (data[0] === "timerUpdate") {
        $("#timer").html(data[1]);
    } else if (data[0] == 'stopTimer') {
        $("#timer").html(data[1]);
    }
});
*/

function createInfo() {
    var info = "This graph shows the evolution of the service level of the system i.e. the percentage of potential users that were able to rent a scooter and finished their trips succecsfully."
    var div = d3.select("body").append("div")
                .attr("class", "tooltip")
                .style("opacity", 0);

    d3.select('#graph-info')
        .on('mouseover', function(){
            d3.select(this)
                .transition()
                .attr('opacity', 0.5)

            div.transition()
                .style("opacity", 1);
            div.html(info)
                .style("left", (1100) + "px")
                .style("top", (170) + "px");

                console.log(d3.event.pageX, d3.event.pageY)
        })
        .on('mouseout', function(){
            d3.select(this)
                .transition()
                .attr('opacity', 1)

            

            div.transition()
                .style("opacity", 0);
        });

}
function projectPoint(x, y) {
    var point = map.latLngToLayerPoint(new L.LatLng(y, x));
    this.stream.point(point.x, point.y);
}

function applyLatLngToLayer(d) {
    var x = d[0];
    var y = d[1];

    return map.latLngToLayerPoint(new L.LatLng(y, x));
}

function loadMap() {

    map = L.map('map', {
        zoomControl: false
    }).setView([38.2440549,-85.756548], 14);
    
    
    var tileURL = getTileURL('dark_all');
    actualLayer = L.tileLayer(tileURL, {
        attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, &copy; <a href="http://cartodb.com/attributions">CartoDB</a>'
    })

    map.addLayer(actualLayer);
    
    outerBounds = {
        type: "FeatureCollection",
        features: [
            {type: "Feature", geometry: { type: "Point", coordinates : [ -87.0246651, 37.7490603 ] }},
            {type: "Feature", geometry: { type: "Point", coordinates : [ -85.3402243, 38.6190106 ] }}
        ]
    }
    
    var southWest = L.latLng(37.7490603,-87.024665),
        northEast = L.latLng(38.6190106,-85.3402243);
    
    var bounds = L.latLngBounds(southWest, northEast);
    
    map.setMaxBounds(bounds);
    map.on('drag', function() {
        map.panInsideBounds(bounds, { animate: false });
    });
    // Overlay SVG
    svg = d3.select(map.getPanes().overlayPane).append("svg");
    g = svg.append("g").attr("class", "leaflet-zoom-hide");
    
    transform = d3.geoTransform({
        point: projectPoint
    });
    
    d3Path = d3.geoPath().projection(transform);

    d3.json('data/bounds.geojson').then(function(collection) {
        g.selectAll('path')
            .data(collection.features)
            .enter()
            .append('path')
            .attr('class', 'boundary')
            .style("stroke-dasharray", ("3, 3")) 
            .attr('d', d3Path);
    })

    d3.json('data/grid.geojson').then(function(collection) {
        g.selectAll('path')
            .data(collection.features)
            .enter()
            .append('path')
            .attr('class', 'grid') 
            .attr('d', d3Path);
    })
}

function loadLegend() {
    legendSVG = d3.select("#legend").append("svg");
    legendG = legendSVG.append("g");
    legendData = [{
        x: 0,
        y:0,
        class: 'static-scooter',
        legend: 'Parked Scooter'
    },
    {
        x: 0,
        y: 1,
        class: 'user',
        legend: 'Pedestrian'
    },
    {
        x: 0,
        y: 2,
        class: 'scooter',
        legend: 'User Riding a Scooter'
    },
    {
        x: 0,
        y: 3,
        class: 'unsatisfied-request',
        legend: 'Unsatisfied Request'
    }]

    legendG.selectAll('circle')
            .data(legendData)
            .enter()
            .append('circle')
            .attr("cx", function (d) { return d.x + 10; })
            .attr("cy", function (d) { return (d.y * 25) + 30; })
            .attr("r", 3)
            .attr("class", function (d){return d.class})

    legendG.selectAll('text')
            .data(legendData)
            .enter()
            .append('text')
            .attr("x", function (d) { return d.x + 30; })
            .attr("y", function (d) { return (d.y * 25) + 35; })
            .text(function(d){return d.legend})
            .attr("font-size", "14px")
            .attr("class", "legend-text")



}
function toDateTime(secs) {
    var t = new Date(2020, 4, 25); // Epoch
    t.setSeconds(secs);
    return t;
}


function restart() {
    svg.selectAll()
}

function loadStatsGraph() {
    var height = 260;
    var width = 360;
    var padding = 45;

    var dummyData = [
       [0, 0.3], [1, 0.32], [2, 0.45], [3, 0.54], [4, 0.36]
    ];



    statsSVG = d3.select("#stats-chart")
                    .append("svg")
                    .attr("width", width)
                    .attr("height", height);

    xScale = d3.scaleLinear()
        .domain([toDateTime(0), toDateTime(0)])
        .range([padding, width - padding]);

    yScale = d3.scaleLinear()
        .domain([0, 1])
        .range([height - padding, padding]);

    xAxis = d3.axisBottom(xScale)
                .tickFormat(d3.timeFormat("%a %H hrs."));
    var formatPercent = d3.format(".0%");
    yAxis = d3.axisLeft(yScale)
            .tickFormat(formatPercent);

    xAxisG = statsSVG.append("g")
            .attr("transform", "translate(0," + (height - padding) + ")")
            .attr("class", "axis")    
            .call(xAxis);

    statsSVG.append("g")
        .attr("transform", "translate(" + padding + ", 0)")
        .attr("class", "axis")
        .call(yAxis);
    
    statsSVG.append("path")
            .datum(serviceLevelStats)
            .attr("class", "chart-line")
            .attr("id", "service-level")
            .attr("d", statsLine);
}


function updateStats() {
    var serviceLevel = getCurrentServiceLevel();
    var currentDate = toDateTime(simRatio * currentTime / 1000)
    serviceLevelStats.push({
        hour: currentDate,
        serviceLevel: serviceLevel
    });
    xScale.domain([toDateTime(0), currentDate]);
    xAxisG.transition().call(xAxis);
    
    statsSVG.select("#service-level")
        .transition()
        .attr("d", statsLine);
    
    
}

function getTileURL(mode){
    return 'http://{s}.basemaps.cartocdn.com/' + mode + '/{z}/{x}/{y}.png';
}


function reset() {

    console.log("Reset");
    var bounds = d3Path.bounds(outerBounds),
                topLeft = bounds[0],
                bottomRight = bounds[1];


    svg.attr('width', bottomRight[0] - topLeft[0])
                .attr('height', bottomRight[1] - topLeft[1])
                .style("left", topLeft[0] + "px")
                .style("top", topLeft[1] + "px");

    svg.selectAll('path.street').attr('d', d3Path);
    svg.selectAll('path.boundary').attr('d', d3Path);
    g.selectAll('path.grid').attr('d', d3Path);

    g.attr('transform', 'translate(' + -topLeft[0] + ', ' + -topLeft[1] + ')');

    svg.selectAll(".static-scooter")
    .attr("transform", function (d) {
        var p = applyLatLngToLayer(d.geometry.coordinates);
        return 'translate(' + p.x + ', ' + p.y + ')';
    });

}



function loadData() {

    Promise.all([
        d3.json('data/bike_edges.geojson'),
        d3.json('data/bike_nodes.geojson'),
        d3.json('data/walk_edges.geojson'),
        d3.json('data/walk_nodes.geojson'),
        d3.json('data/test.json'),
    ])
   .then(function (data) {
        bikeEdgesCollection = data[0];
        bikeNodesCollection = data[1];
        walkEdgesCollection = data[2];
        walkNodesCollection = data[3];
        tripsCollection = data[4];
        scooterLocations = locateScooters();
        console.log("Done");
    });
    
}

function locateScooters() {
    return new Promise(function (resolve, reject) {
        d3.json('data/scooter_locations.json').then(function (scootersCollection){
            console.log('Locating Scooters')
            var scooterLocations = bikeNodesCollection.features.filter(node => scootersCollection.includes(String(node.properties.osmid)));
            console.log(scooterLocations)
            g.selectAll('circle')
                .data(scooterLocations)
                .enter()
                .append("circle")
                .attr("r", 4)
                .style("fill", d3.color(d3.interpolateRdYlGn(1)).hex())
                .attr("class", "static-scooter")
                .attr("id", function(d) {
                    d.id = scootersCollection.indexOf(String(d.properties.osmid)) + 1;
                    d.battery = 1
                    return "scooter-" + (scootersCollection.indexOf(String(d.properties.osmid)) + 1);
                })
                .attr("transform", function (d) {
                    var p = applyLatLngToLayer(d.geometry.coordinates);
                    return 'translate(' + p.x + ', ' + p.y + ')';
                }).on('mouseover', function(d) {
                    var div = d3.select("body").append("div")
                                        .attr("class", "tooltip")
                                        .attr("id", "scooter-info-" + d.id)
                                        .style("opacity", 0);
                    var info = 'Scooter ' + d.id + '</br>' + 'Battery: ' + parseInt(d.battery * 100) + '%';

                    div.transition()
                        .style("opacity", 1);
                    coords = d3.select(this)
                            .attr("transform")
                            .replace('translate(', '')
                            .replace(')', '')
                            .split(', ')
                    console.log(coords);
                    div.html(info)
                        .style("left",(parseInt(coords[0]) )+ "px")
                        .style("top", parseInt(coords[1]) + "px");

                    g.append('div')
                        .attr("class", "scooter-info")
                        .html('Scooter ' + d.id + '\nBattery: ' + d.battery)
                }).on('mouseout', function(d) {
                    d3.select('div#scooter-info-' + d.id).transition().style('opacity', 0).on('end', function(){d3.select(this.remove())})
                });
            resolve(scooterLocations)
        }) 
    })
   
}

function animateAllTrips() {
   
    var q = d3.queue();
    trips = tripsCollection.trips;
    
    //trips = [tripsCollection.trips[2]];
    var t = startTimer();

    trips.forEach(function(trip) {
        q.defer(animateTrip, trip, trip.arrival_time);
    })

    q.await(function () {
        t.stop();
    })
    
}

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


function startTimer() {
    var t = d3.timer(function (elapsed) {
        currentTime = elapsed;
        var returnStr = dhm(elapsed * simRatio);

        //console.log(elapsed + ': '+ returnStr);
        $("#timer").html(returnStr);
    }, 100)
    return t;
}

function animateUnsatisfiedRequest(osmid, delay) {

        console.log('Unsatisfied Request!');
        node = walkNodesCollection.features.find(node => node.properties.osmid == osmid);
    
        g.selectAll('circle.unsatisfied-requests')
            .data([node])
            .enter()
            .append("circle")
            .attr("class", "unsatisfied-request")
            .attr("r", 0)
            .attr("transform", function(d) {
                var p = applyLatLngToLayer(d.geometry.coordinates);
                return 'translate(' + p.x + ', ' + p.y + ')';
            })
            .transition()
            .delay(1000 * delay / simRatio)
            .duration(5000)
            .ease(d3.easeLinear)
            .attr("r", 7)
            .style("opacity", 0.3)
            .on("end", function() {
                d3.select(this).remove()
            });

    
}


function animateTrip(trip, delay, callback)  {
    
    
    d3.timeout(function (elapsed) {
        totalRequests++;
        
        if (trip.walk.length == 0 && trip.ride.length == 0) {
            // user request scooter and there is none available
            animateUnsatisfiedRequest(trip.origin, 0);
        }
        else if (trip.walk.length == 0 && trip.ride.length > 0) {
            // user request scooter in the same place where there is an available scooter
            satisfiedRequests++;
            shortestPathRide =[]
            trip.ride.forEach(function(osmid) {
                shortestPathRide.push(bikeEdgesCollection.features.find(edge => String(edge.properties.osmid) == osmid));
            });
            console.log(shortestPathRide)
            animateRide(shortestPathRide, 0, 0, trip.scooter.id);
        }
        else if (trip.walk.length > 0 && trip.ride.length == 0) {
            // user walks towards an available scooter but when arrives
            // it is not available anymore
            shortestPathWalk =[]
            trip.walk.forEach(function(osmid) {
                shortestPathWalk.push(walkEdgesCollection.features.find(edge => String(edge.properties.osmid) == osmid));
            });
            animateWalk(shortestPathWalk, 0, 0);
            animateUnsatisfiedRequest(trip.pickup_node, trip.pickup_time - trip.arrival_time);
        } else {
            // user walks and picks up an scooter
            satisfiedRequests++;
            shortestPathWalk =[]
            trip.walk.forEach(function(osmid) {
                shortestPathWalk.push(walkEdgesCollection.features.find(edge => String(edge.properties.osmid) == osmid));
            })
            
            shortestPathRide =[]
            trip.ride.forEach(function(osmid) {
                
                shortestPathRide.push(bikeEdgesCollection.features.find(edge => String(edge.properties.osmid) == osmid));
            })
            
            scooter = trip.scooter;
            ride_delay = trip.pickup_time - trip.arrival_time;
            animateWalk(shortestPathWalk, 0, 0);
            animateRide(shortestPathRide, 0, ride_delay, scooter);
        }
        updateStats();
        callback(null);
    }, delay * 1000 / simRatio)
    
}

function animateWalk(path, ipath, delay)  {
    
    var currentEdge = path[ipath];
    var edge = g.append("path")
                .attr("id", function () {
                    return "edge-" + currentEdge.properties.osmid;
                })
                .attr("class", "walk")
                .attr("d", function(){
                    return d3Path(currentEdge.geometry);
                });
    var circle = g.append("circle")
                    .attr("class", "user")
                    .transition()
                    .delay(function() {
                        return 1000 * delay / simRatio;
                    })
                    .duration(function() {
                        var velocity = 1.4;
                        return (1000 * currentEdge.properties.time) / simRatio;
                    })
                    .ease(d3.easeLinear)
                    .attrTween("transform", function () {
                        return function(t) {
                            var pathLength = edge.node().getTotalLength();
                            var p = edge.node().getPointAtLength(t * pathLength);
                            return 'translate(' + p.x + ', ' + p.y + ')';
                        }

                    })
                    .on("start", function() {
                        d3.select(this).attr("r", 3);
                        console.log(delay);
                    })
                    .on("end", function () {
                        d3.select(this).remove()                        
                        d3.select("path#edge-" + currentEdge.properties.osmid).remove()
                        if (ipath < path.length - 1) {
                            animateWalk(path, ipath + 1, 0);
                        }
                    
                    })
}


function animateRide(path, ipath, delay, scooter) {
    
    var currentEdge = path[ipath];
    
    var edge = g.append("path")
                .attr("id", function () {
                    return "edge-" + currentEdge.properties.osmid;
                })
                .attr("class", "walk")
                .attr("d", function(){
                    return d3Path(currentEdge.geometry);
                });
    
    var circle = g.select("circle#scooter-" + scooter.id)
    circle.transition()
                    .delay(function() {
                        return 1000 * delay / simRatio;
                    })
                    .duration(function() {
                        var velocity = 2.16;
                        return (1000 * currentEdge.properties.time) / simRatio;
                    })
                    .ease(d3.easeLinear)
                    .attrTween("transform", function () {
                        return function(t) {
                            var pathLength = edge.node().getTotalLength();
                            var p = edge.node().getPointAtLength(t * pathLength);
                            return 'translate(' + p.x + ', ' + p.y + ')';
                        }

                    })
                    .on("start", function() {
                        d3.select(this).attr("class", "scooter");
                    })
                    .on("end", function () {                     
                        d3.select("path#edge-" + currentEdge.properties.osmid).remove()
                        if (ipath < path.length - 1) {
                            animateRide(path, ipath + 1, 0, scooter);
                        } else {
                            var thisScooter = d3.select("circle#scooter-" + scooter.id)
                                                    .attr("class", "static-scooter")
                                                    .style("fill", function (d) {
                                                        d.battery = scooter.battery_level_dropoff / 100
                                                        return d3.color(d3.interpolateRdYlGn(scooter.battery_level_dropoff / 100)).hex();
                                                    })
                            thisScooter.datum().geometry.coordinates = currentEdge.geometry.coordinates.slice(-1)[0].slice(-1)[0];
                                
                        }
                    
                    })
}

function adjustVelocity() {
    var slideValue = $("#simratio-slider").val();
    var ratio = ((slideValue - 1) * 100);
    if (ratio == 0) {
        simRatio = 1;
    } else {
        simRatio = ratio;
    }
    console.log("X" + simRatio);
    $("#velocity-display").html("X" + simRatio);
}


function pause() {
    g.selectAll('circle')
        .transition()
        .duration(0)
}