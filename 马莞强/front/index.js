window.onload = function() {
    var getImageListRequest = new XMLHttpRequest();
    getImageListRequest.onreadystatechange = function() {
        if (getImageListRequest.readyState === XMLHttpRequest.DONE) {
            if (getImageListRequest.status === 200) {
                var imageList = JSON.parse(getImageListRequest.responseText);
                imageList.forEach(function(item) {
                    var option=document.createElement("option");
                    option.text = item;
                    document.getElementById("image-list").add(option, null);
                });
            } else {
                alert("failed to get image list");
            }
        }
    }
    getImageListRequest.open("GET", "http://127.0.0.1:5000/get-image-list", true);
    getImageListRequest.send();

    var deleteImageRequest = new XMLHttpRequest();
    deleteImageRequest.onreadystatechange = function() {
        if (deleteImageRequest.readyState === XMLHttpRequest.DONE) {
            if (deleteImageRequest.status === 200) {
                var data = JSON.parse(deleteImageRequest.responseText);
                document.getElementById("image-list").remove(data.index);
                document.getElementById("image").src = null;
            } else {
                alert("failed to delete image");
            }
        }
    }

    var uploadImageRequest = new XMLHttpRequest();
    uploadImageRequest.onreadystatechange = function() {
        if (uploadImageRequest.readyState === XMLHttpRequest.DONE) {
            if (uploadImageRequest.status === 200) {
                var data = JSON.parse(uploadImageRequest.responseText);
                var option=document.createElement("option");
                option.text = data.filename;
                document.getElementById("image-list").add(option, null);
            } else {
                alert("failed to upload image");
            }
        }
    }

    var clearRequest = new XMLHttpRequest();
    clearRequest.onreadystatechange = function() {
        if (clearRequest.readyState === XMLHttpRequest.DONE) {
            if (clearRequest.status === 200) {
                document.getElementById("image-list").innerHTML = '';
            } else {
                alert("failed to clear images");
            }
        }
    }

    document.getElementById("add-image").onclick = function() {
        document.getElementById("file-input").click();
    };
    document.getElementById("file-input").addEventListener("change", function() {
        var modelSelect = document.getElementById("model");
        var model = modelSelect.options[modelSelect.selectedIndex].text;
        var formData = new FormData();
        formData.append("file", this.files.item(0));
        uploadImageRequest.open("POST", "http://127.0.0.1:5000/upload-image/" + model, true);
        uploadImageRequest.send(formData);
        console.log(model);
    }, false);
    document.getElementById("image-list").onchange = function() {
        document.getElementById("image").src = "http://127.0.0.1:5000/get-image/" + this.value
    };
    document.getElementById("delete-image").onclick = function() {
        var filename = document.getElementById("image-list").value;
        var index = document.getElementById("image-list").selectedIndex.toString();
        if(filename !== "") {
            deleteImageRequest.open("GET", "http://127.0.0.1:5000/delete-image/" + index + "/" + filename, true);
            deleteImageRequest.send();
        }
    }
    document.getElementById("clear").onclick = function() {
        clearRequest.open("GET", "http://127.0.0.1:5000/clear", true);
        clearRequest.send();
    }
}
