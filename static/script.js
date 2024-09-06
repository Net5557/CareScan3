let stream = null;
let mybutton = document.getElementById("myBtn");
window.onscroll = function () { scrollFunction() };

function scrollFunction() {
    if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
        mybutton.style.display = "block";
    } else {
        mybutton.style.display = "none";
    }
}

function topFunction() {
    document.body.scrollTop = 0;
    document.documentElement.scrollTop = 0;
}

function uploadImage() {
    const imageInput = document.getElementById('imageInput');
    const file = imageInput.files[0];

    if (!file) {
        alert('Please select an image file.');
        return;
    }

    const formData = new FormData();
    formData.append('image', file);

    const reader = new FileReader();
    reader.onload = function (event) {
        const img = document.getElementById('uploadedImage');
        img.src = event.target.result;
        img.style.display = 'block';
    };
    reader.readAsDataURL(file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

function displayResults(data) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = '';

    if (data.length === 0) {
        resultDiv.innerHTML = '<div class="alert alert-warning">No objects detected.</div>';
        return;
    }

    data.forEach(prediction => {
        const div = document.createElement('div');
        div.className = 'alert alert-info';
        div.innerHTML = `<strong>Class:</strong> ${prediction.class_id}<br>
                         <strong>Coordinates:</strong> ${prediction.coordinates}<br>
                         <strong>Confidence:</strong> ${prediction.confidence}%`;
        resultDiv.appendChild(div);
    });
}

$(document).on('change', '.custom-file-input', function (event) {
    var inputFile = event.currentTarget;
    $(inputFile).parent()
        .find('.custom-file-label')
        .html(inputFile.files[0].name);
});
