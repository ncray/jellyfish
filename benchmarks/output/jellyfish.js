function draw_with_data(data, parent_id) {
  var g = d3.select(parent_id)
            .append("svg")
              .attr("width", "152.4mm")
              .attr("height", "152.4mm")
              .attr("viewBox", "0 0 152.4 152.4")
              .attr("stroke-width", "0.5")
              .attr("style", "stroke:black;fill:black");
  g.append("defs");
  var t = {"scale": 1.0};
(function (g) {
      d3.select("defs")
      .append("svg:clipPath")
        .attr("id", parent_id + "_clippath0")
        .append("svg:path")
          .attr("d", "M 5 5 L 26.09 5 26.09 134.94 5 134.94 z");
g.attr("clip-path", "url(#" + parent_id + "_clippath0)");
  (function (g) {
    g.attr("stroke", "none")
     .attr("fill", "#4C404B")
     .attr("font-family", "PT Sans,Helvetica Neue,Helvetica,sans")
     .attr("font-size", "3.18")
     .attr("class", "guide ylabels");
    g.append("svg:text")
       .attr("x", 25.09)
       .attr("y", -21.97)
       .attr("text-anchor", "end")
       .style("dominant-baseline", "central")
    .call(function(text) {
  text.text("2.5");
})
;
    g.append("svg:text")
       .attr("x", 25.09)
       .attr("y", -83.26)
       .attr("text-anchor", "end")
       .style("dominant-baseline", "central")
    .call(function(text) {
  text.text("3.5");
})
;
    g.append("svg:text")
       .attr("x", 25.09)
       .attr("y", 8.68)
       .attr("text-anchor", "end")
       .style("dominant-baseline", "central")
    .call(function(text) {
  text.text("2");
})
;
    g.append("svg:text")
       .attr("x", 25.09)
       .attr("y", 39.32)
       .attr("text-anchor", "end")
       .style("dominant-baseline", "central")
    .call(function(text) {
  text.text("1.5");
})
;
    g.append("svg:text")
       .attr("x", 25.09)
       .attr("y", -52.61)
       .attr("text-anchor", "end")
       .style("dominant-baseline", "central")
    .call(function(text) {
  text.text("3");
})
;
    g.append("svg:text")
       .attr("x", 25.09)
       .attr("y", 253.85)
       .attr("text-anchor", "end")
       .style("dominant-baseline", "central")
    .call(function(text) {
  text.text("-2");
})
;
    g.append("svg:text")
       .attr("x", 25.09)
       .attr("y", 131.26)
       .attr("text-anchor", "end")
       .style("dominant-baseline", "central")
    .call(function(text) {
  text.text("0");
})
;
    g.append("svg:text")
       .attr("x", 25.09)
       .attr("y", 69.97)
       .attr("text-anchor", "end")
       .style("dominant-baseline", "central")
    .call(function(text) {
  text.text("1");
})
;
    g.append("svg:text")
       .attr("x", 25.09)
       .attr("y", 192.55)
       .attr("text-anchor", "end")
       .style("dominant-baseline", "central")
    .call(function(text) {
  text.text("-1");
})
;
    g.append("svg:text")
       .attr("x", 25.09)
       .attr("y", 161.91)
       .attr("text-anchor", "end")
       .style("dominant-baseline", "central")
    .call(function(text) {
  text.text("-0.5");
})
;
    g.append("svg:text")
       .attr("x", 25.09)
       .attr("y", 100.62)
       .attr("text-anchor", "end")
       .style("dominant-baseline", "central")
    .call(function(text) {
  text.text("0.5");
})
;
    g.append("svg:text")
       .attr("x", 25.09)
       .attr("y", -113.91)
       .attr("text-anchor", "end")
       .style("dominant-baseline", "central")
    .call(function(text) {
  text.text("4");
})
;
    g.append("svg:text")
       .attr("x", 25.09)
       .attr("y", 284.49)
       .attr("text-anchor", "end")
       .style("dominant-baseline", "central")
    .call(function(text) {
  text.text("-2.5");
})
;
    g.append("svg:text")
       .attr("x", 25.09)
       .attr("y", 223.2)
       .attr("text-anchor", "end")
       .style("dominant-baseline", "central")
    .call(function(text) {
  text.text("-1.5");
})
;
    g.append("svg:text")
       .attr("x", 25.09)
       .attr("y", -144.55)
       .attr("text-anchor", "end")
       .style("dominant-baseline", "central")
    .call(function(text) {
  text.text("4.5");
})
;
  }(g.append("g")));
  (function (g) {
    g.attr("stroke", "none")
     .attr("fill", "#362A35")
     .attr("font-family", "PT Sans,Helvetica Neue,Helvetica,sans")
     .attr("font-size", "3.88");
    g.append("svg:text")
       .attr("x", 9.33)
       .attr("y", 69.97)
       .attr("text-anchor", "middle")
       .style("dominant-baseline", "central")
       .attr("transform", "rotate(-90, 9.33, 69.97)")
    .call(function(text) {
  text.text("y");
})
;
  }(g.append("g")));
}(g.append("g")));
(function (g) {
      d3.select("defs")
      .append("svg:clipPath")
        .attr("id", parent_id + "_clippath1")
        .append("svg:path")
          .attr("d", "M 26.09 134.94 L 147.4 134.94 147.4 147.4 26.09 147.4 z");
g.attr("clip-path", "url(#" + parent_id + "_clippath1)");
  (function (g) {
    g.attr("stroke", "none")
     .attr("fill", "#4C404B")
     .attr("font-family", "PT Sans,Helvetica Neue,Helvetica,sans")
     .attr("font-size", "3.18")
     .attr("class", "guide xlabels");
    g.append("svg:text")
       .attr("x", 86.75)
       .attr("y", 138.75)
       .attr("text-anchor", "middle")
    .call(function(text) {
  text.text("50000");
})
;
    g.append("svg:text")
       .attr("x", 143.97)
       .attr("y", 138.75)
       .attr("text-anchor", "middle")
    .call(function(text) {
  text.text("100000");
})
;
    g.append("svg:text")
       .attr("x", 29.53)
       .attr("y", 138.75)
       .attr("text-anchor", "middle")
    .call(function(text) {
  text.text("1");
})
;
  }(g.append("g")));
  (function (g) {
    g.attr("stroke", "none")
     .attr("fill", "#362A35")
     .attr("font-family", "PT Sans,Helvetica Neue,Helvetica,sans")
     .attr("font-size", "3.88");
    g.append("svg:text")
       .attr("x", 86.75)
       .attr("y", 145.4)
       .attr("text-anchor", "middle")
    .call(function(text) {
  text.text("x");
})
;
  }(g.append("g")));
}(g.append("g")));
(function (g) {
  g.on("mouseover", guide_background_mouseover(parent_id, "#CACACD"))
   .on("mouseout", guide_background_mouseout(parent_id, "#F4F4F8"))
   .call(zoom_behavior(parent_id, t));
  (function (g) {
        d3.select("defs")
      .append("svg:clipPath")
        .attr("id", parent_id + "_clippath2")
        .append("svg:path")
          .attr("d", "M 26.09 5 L 147.4 5 147.4 134.94 26.09 134.94 z");
g.attr("clip-path", "url(#" + parent_id + "_clippath2)");
    (function (g) {
      g.attr("class", "guide background")
       .attr("stroke", "#F4F4F8")
       .attr("fill", "#FCFCFC");
      g.append("svg:path")
         .attr("d", "M 26.09 5 L 147.4 5 147.4 134.94 26.09 134.94 z");
    }(g.append("g")));
    (function (g) {
      g.attr("stroke", "#F4F4F8")
       .attr("stroke-width", "0.2")
       .attr("class", "guide ygridlines");
      g.append("svg:path")
         .attr("d", "M 26.09 -21.97 L 147.4 -21.97");
      g.append("svg:path")
         .attr("d", "M 26.09 -83.26 L 147.4 -83.26");
      g.append("svg:path")
         .attr("d", "M 26.09 8.68 L 147.4 8.68");
      g.append("svg:path")
         .attr("d", "M 26.09 39.32 L 147.4 39.32");
      g.append("svg:path")
         .attr("d", "M 26.09 -52.61 L 147.4 -52.61");
      g.append("svg:path")
         .attr("d", "M 26.09 253.85 L 147.4 253.85");
      g.append("svg:path")
         .attr("d", "M 26.09 131.26 L 147.4 131.26");
      g.append("svg:path")
         .attr("d", "M 26.09 69.97 L 147.4 69.97");
      g.append("svg:path")
         .attr("d", "M 26.09 192.55 L 147.4 192.55");
      g.append("svg:path")
         .attr("d", "M 26.09 161.91 L 147.4 161.91");
      g.append("svg:path")
         .attr("d", "M 26.09 100.62 L 147.4 100.62");
      g.append("svg:path")
         .attr("d", "M 26.09 -113.91 L 147.4 -113.91");
      g.append("svg:path")
         .attr("d", "M 26.09 284.49 L 147.4 284.49");
      g.append("svg:path")
         .attr("d", "M 26.09 223.2 L 147.4 223.2");
      g.append("svg:path")
         .attr("d", "M 26.09 -144.55 L 147.4 -144.55");
    }(g.append("g")));
    (function (g) {
      g.attr("stroke", "#F4F4F8")
       .attr("stroke-width", "0.2")
       .attr("class", "guide xgridlines");
      g.append("svg:path")
         .attr("d", "M 86.75 5 L 86.75 134.94");
      g.append("svg:path")
         .attr("d", "M 143.97 5 L 143.97 134.94");
      g.append("svg:path")
         .attr("d", "M 29.53 5 L 29.53 134.94");
    }(g.append("g")));
  }(g.append("g")));
  (function (g) {
        d3.select("defs")
      .append("svg:clipPath")
        .attr("id", parent_id + "_clippath3")
        .append("svg:path")
          .attr("d", "M 26.09 5 L 147.4 5 147.4 134.94 26.09 134.94 z");
g.attr("clip-path", "url(#" + parent_id + "_clippath3)");
    (function (g) {
      g.attr("stroke", "#4682B4")
       .attr("class", "geometry")
       .attr("fill", "none")
       .attr("stroke-width", "0.2");
      g.append("svg:path")
    }(g.append("g")));
  }(g.append("g")));
}(g.append("g")));
}

var data = [
];

var draw = function(parent_id) {
    draw_with_data(data, parent_id);
};