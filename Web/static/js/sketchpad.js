var contentImg;
var styleImg;
var ModelName = 'e2s'

function chooseModel(name)
{
    if(name == 'shoes')
        ModelName = 'e2s';
    else if(name == 'handbags')
        ModelName = 'e2h';
    else
        ModelName = 'identity';
}

function previewFile() {
    var preview = document.getElementById('previewStyle');
    var file    = document.getElementById('styleImg').files[0];
    var reader  = new FileReader();
  
    reader.addEventListener("load", function () {
      preview.setAttribute('src', reader.result);
      styleImg = reader.result;
    }, false);
  
    if (file) {
      reader.readAsDataURL(file);
    }
  }

function uploadStyle() {
    $.post("/api/transfer", { style: styleImg, modelname: 'e2s', modeltype: 'vanilla' },
            function (data, status) {
                //alert("Data: " + data + "\nStatus: " + status);
                var result = data
                document.getElementById('previewPort').setAttribute('src', "/"+result['url'])
                contentImg = result['url']
            });
}

function createSketchpad() {
    //Turn a canvas element into a sketch area
    $("#sketcharea").drawr({
        "enable_tranparency": false,
        "color_mode": "presets",
        "canvas_width": 900,
        "canvas_height": 600,
        "clear_on_init": true
    });

    //Enable drawing mode, show controls
    $("#sketcharea").drawr("start");

    //add custom save button.
    var buttoncollection = $("#sketcharea").drawr("button", {
        "icon": "mdi mdi-folder-open mdi-24px"
    }).on("touchstart mousedown", function () {
        $("#file-picker").click();
    });
    var buttoncollection = $("#sketcharea").drawr("button", {
        "icon": "mdi mdi-content-save mdi-24px"
    }).on("touchstart mousedown", function () {
        var imagedata = $("#sketcharea").drawr("export", "image/jpeg");
        /*
        var element = document.createElement('a');
        element.setAttribute('href', imagedata);
        element.setAttribute('download', "test.jpg");
        element.style.display = 'none';
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);*/
        $.post("/api/generate", { img: imagedata, modelname: ModelName, modeltype: 'vanilla' },
            function (data, status) {
                //alert("Data: " + data + "\nStatus: " + status);
                var result = data
                document.getElementById('previewPort').setAttribute('src', "/"+result['url'])
                contentImg = result['url']
            });
    });
    $("#file-picker")[0].onchange = function () {
        var file = $("#file-picker")[0].files[0];
        if (!file.type.startsWith('image/')) { return }
        var reader = new FileReader();
        reader.onload = function (e) {
            $("#sketcharea").drawr("load", e.target.result);
        };
        reader.readAsDataURL(file);
    };
}

function resetPad()
{
    
}

function destroy() {
    $("#sketcharea").drawr("destroy");
}



$(document).ready(function () {
    createSketchpad();
});


