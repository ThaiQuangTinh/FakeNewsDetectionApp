// Declare variables
const newsTextArea = document.getElementById('textarea-content');
const containResult = document.getElementById('contain-results');
const result = document.getElementById('result');
const description = document.getElementById('description');
const validateMessage = document.getElementById('validateMessage');

var feedbackResult = '';

// This function is used to display result of fake news detection
const displayResult = (prediction, probability) => {
    let percentValue = Math.round(probability * 100);
    let progressBar = document.querySelector('.progress-bar');
    containResult.style.display = 'block';
    if (prediction === 'fake') {
        result.innerHTML = "Fake";
        result.style.color = 'red';
        description.innerHTML = `The news is ${percentValue}% fake`;
        progressBar.style.width = `${percentValue}%`;
    } else {
        result.innerHTML = "Real";
        result.style.color = 'rgb(11, 156, 23)';
        description.innerHTML = `The news is ${percentValue}% real`;
        progressBar.style.width = `${percentValue}%`;
    }
}

// This function is used refresh Result
const refreshResult = () => {
    newsTextArea.addEventListener('focus', () => {
        containResult.style.display = 'none';
        newsTextArea.value = '';
    });
}

// This function is used to handle when user clicks on the predict button
document.getElementById('btn-predict').addEventListener('click', function() {
    let newsText = newsTextArea.value;

    if (newsText === '') {
        validateMessage.style.visibility = 'visible';
        return;
    } else {
        validateMessage.style.visibility = 'hidden';
    }

    fetch('http://127.0.0.1:5000/api/predict_news', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ news_text: newsText })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log(data)
            displayResult(data.prediction, data.probability);
            refreshResult();
        })
        .catch(error => {
            console.error('Error:', error);
        });
});

// Handle feedback

// Declare variables
var btnFeedbackReal = document.getElementById('btn-feedback-real');
var btnFeedbackFake = document.getElementById('btn-feedback-fake');

var btnConfirnFeedback = document.getElementById("btn-confirn-feedback");
var btnCancelFeddback = document.getElementById("btn-cancel-feddback");

var containMessageBox = document.querySelector('.contain-message-box');
var messageBox = document.querySelector('.message-box');
var messageContent = document.querySelector('.messsage-content');

// Handle when user click to the btnFeedbackReal button
btnFeedbackReal.addEventListener('click', function() {
    containMessageBox.style.display = 'flex';
    messageContent.innerHTML = 'Are you sure this a real news?';
    feedbackResult = 'real';
});

// Handle when user click to the btnFeedbackFake button
btnFeedbackFake.addEventListener('click', function() {
    containMessageBox.style.display = 'flex';
    messageContent.innerHTML = 'Are you sure this a fake news?';
    feedbackResult = 'fake';
});

// Handle when user click to the containMessageBox panel
containMessageBox.addEventListener('click', function() {
    containMessageBox.style.display = 'none';
});

// Stop propagation when user click to the message box panel
messageBox.addEventListener('click', function(event) {
    event.stopPropagation();
});

// Handle when user click to the btnConfirnFeedback button
btnConfirnFeedback.addEventListener('click', function(event) {
    event.preventDefault();
    const data = {
        text: newsTextArea.value,
        type: feedbackResult == 'fake' ? 'fake' : 'real'
    };

    fetch("http://127.0.0.1:5000/api/receive_news", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error("Failed to send news data.");
            }
            return response.json();
        })
        .then(data => {
            console.log(data.message); // In ra thông báo từ server
            let messageResponse = document.getElementById('message-response');
            containMessageBox.style.display = 'none'
            messageResponse.style.display = 'block';
            setTimeout(() => {
                messageResponse.style.animation = 'hideResponse ease 1s';
                setTimeout(() => {
                    messageResponse.style.display = 'none';
                    messageResponse.style.animation = 'showResponse ease 1s';
                }, 500);
            }, 2000);
        })
        .catch(error => {
            console.error("Error:", error);
        });

});

// Handle when user click to the btnCancelFeddback button
btnCancelFeddback.addEventListener('click', function() {
    containMessageBox.style.display = 'none';
});