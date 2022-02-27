var top_categ = ["Sport","Transport","Art","Security"];

var sport_categ = ['chess','skiing','weightlifting','climbing','cricket','flying','hockey','soccer','volleyball','tennis','skateboarding','swimming','rowing','roller_skating','horse_racing','steeplechase','jogger','gymnastics','golf','diving','car_racing','boxing','bowling','billiard','beach_volleyball','basketball','baseball','jumping','running','acrobatics','figure_skating','motorcyclist-cross','motorcycle racing','fisherman','hunter','hang_gliding','rhythmic gymnastics','racing_bicycle','sleigh','handball'];

var transport_categ = ['airplane','glider','helicopter','hot-air_balloon','bicycle','camper','convertible','jeep','limousine','sedan','taxi','wagon','carriage','motorcycle','bus','minibus','tram','trolleybus','boat','ferry','gondola','motorboat','sailing_vessel','ship','yacht','astronaut','rocket','spaceship','train','car_transporter','dumper','garbage_truck','lorry','pickup','tow_truck','truck','van','bulldozer','digger','forklift','tractor','baby_carriage','wheelchair','horse_sleigh','dog sleigh','double-decker','road_sign','zebra_crossing','rickshaw','scooter'];

var art_categ = ['artist','sculptor','accordionist','piper','cellist','clarinetist','conductor','flute_player','guitar_player','opera_singer','percussionist','piano player','rapper','saxophonist','singer','trombonist','trumpeter','violin','ballet_dancer','cameraman','clown','dancer','makeup_artist','photographer','writer'];

var security_categ = ['traffic_police','fire engine','fireman','police car','police helicopter','mounted police','policeman','tank','ambulance','military helicopter','military_truck_NEW','police boat','motorized police','soldier','APC_NEW da stane APC'];

var topc;
var form1;
var sel1;

function init_data() {
    form1 = document.getElementById("form1");
        
    topc = document.getElementById("top_categ");
    for (let i = 0; i < top_categ.length; i++) {
        let option = document.createElement("option");
        option.text = top_categ[i];
        topc.add(option);
    }
    gen_subcateg();
}

function gen_subcateg() {
    let val = topc.value;
    console.log("click");
    
    form1.innerHTML = "";
    sel1 = document.createElement("select");
    sel1.name = "select1";
    sel1.id = "select1";
    form1.append("Select subcategory:");
    form1.append(sel1);
    let sub_categs;
    switch (val) {
        case "Sport": sub_categs = sport_categ; break;
        case "Transport": sub_categs = transport_categ; break;
        case "Art": sub_categs = art_categ; break;
        case "Security": sub_categs = security_categ; break;
    }
    for (let i = 0; i < sub_categs.length; i++) {
        let option = document.createElement("option");
        option.text = sub_categs[i];
        sel1.add(option);
    }
    show_images();
    $("#select1").change(function() {show_images();});
}

var image_list = [];

function show_images() {
    console.log("show images");
    let val = sel1.value
    $.getJSON('/get_image_links?categ_name='+val, function(data) {
        image_list = data;
        let str_buf = '';
        for (let i = 0; i < image_list.length; i++) {
            str_buf = str_buf + '<img src="' + image_list[i] + '" width=100px>'
        }
        $("#image_container").html(str_buf);
    });
}
                             