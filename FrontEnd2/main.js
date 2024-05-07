const btnSubmit = document.getElementById('btnSubmit');
const btnClose = document.querySelector('.closeButton');
const newsContentElement = document.getElementById('newsContent');
const containtValidate = document.getElementById('containtValidate');
const containResult = document.getElementById('containResult');
const checkBoxMode = document.getElementById('flexSwitchCheckChecked');

const displayResult = (prediction, probability) => {
    let trueIcon = document.getElementById('trueIcon');
    let fakeIcon = document.getElementById('fakeIcon');
    let resultText = document.getElementById('resultText');
    let descrpitionForPercent = document.getElementById('descrpitionForPercent');

    var progressBar = document.querySelector('.progress-bar');
    var progressBarBefore = window.getComputedStyle(progressBar, '::before');

    let percentResult = Math.round(probability * 100);

    if (prediction == 'fake') {
        trueIcon.style.display = 'none';
        fakeIcon.style.display = 'inline-block';
        resultText.innerText = 'Fake news';
        descrpitionForPercent.innerText = `The news is ${percentResult}% fake!`;
        progressBar.style.setProperty('--progress-content', '"' + percentResult + '%"');
        // progressBar.style.background = `radial-gradient(closest-side, white 79%, transparent 80% 100%), conic-gradient(rgb(65, 138, 235) ${percentResult}, rgb(136, 205, 236) 0)`;
        progressBar.style.background = `radial-gradient(closest-side, white 79%, transparent 80% 100%), conic-gradient(rgb(227, 24, 24) ${percentResult}%, rgb(214, 140, 140) 0)`;
    } else {
        trueIcon.style.display = 'inline-block';
        fakeIcon.style.display = 'none';
        resultText.innerText = 'True news';
        descrpitionForPercent.innerText = `The news is ${percentResult}% true!`;
        progressBar.style.setProperty('--progress-content', '"' + percentResult + '%"');
        progressBar.style.background = `radial-gradient(closest-side, white 79%, transparent 80% 100%), conic-gradient(rgb(37 203 19) ${percentResult}%, rgb(160 231 177) 0)`;
    }

}

btnSubmit.addEventListener('click', function(event) {
    event.preventDefault();

    if (newsContentElement.value == '') {
        containtValidate.style.display = 'block';
        return;
    } else {
        containtValidate.style.display = 'none';
    }

    containResult.style.display = 'block';
    containResult.style.animation = 'fadeIn ease-in .5s';

    fetch('http://127.0.0.1:5000/api/predict_news', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ news_text: newsContentElement.value })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log(data);
            displayResult(data.prediction, data.probability);
        })
        .catch(error => {
            console.error('Error:', error);
        });
});

newsContentElement.addEventListener('input', function() {
    containtValidate.style.display = 'none';
});

btnClose.addEventListener('click', function() {
    document.getElementById('containResult').style.display = 'none';
    containResult.style.animation = 'fadeOut ease-out .5s';
    newsContentElement.value = '';
});


const handleChangeCheckbox = () => {
    let labelMode = document.querySelector('.form-check-label');
    if (checkBoxMode.checked) {
        labelMode.innerText = 'Light mode';
        document.querySelector('body').style.backgroundColor = '#fff';
    } else {
        labelMode.innerText = 'Dark mode';
        document.querySelector('body').style.backgroundColor = '#170f23';
    }
};

// Thêm sự kiện onchange để gọi hàm khi trạng thái của checkbox thay đổi
checkBoxMode.addEventListener('change', handleChangeCheckbox);

// Gọi hàm một lần để cập nhật trạng thái ban đầu của checkbox
handleChangeCheckbox();