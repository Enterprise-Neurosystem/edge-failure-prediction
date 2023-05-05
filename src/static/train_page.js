
// Once model has been trained, display the loss graph.
function trainGraph(){
    //console.log("trainGraph()");
    makeGraphBtnObj.disabled = true;
    trainBtnObj.disabled = false;
    let url = '/display-train-graph';
    let formObj = document.getElementById("trainForm")
    let formData = new FormData(formObj);
    postTrainGraphRequest(url, formData)
        .then(serializedImage => displayTrainGraphHTML(serializedImage))
        .catch(error => console.error(error))
}
async function postTrainGraphRequest(url, formData) {
    return fetch(url, {
        method: 'POST',
        body: formData
    })
        .then((response) => response.text());
}

function displayTrainGraphHTML(serializedImage){
    graphDivObj.style.display = 'block';
    let graphHTML =  "<img src='data:image/png;base64, " + serializedImage +
        "' class='imgSize'/>" ;
    graphDivObj.innerHTML = graphHTML;
    testTabBtn.disabled = false;
    predTabBtn.disabled = false;
}
// Train model
function trainModel(){
    trainBtnObj.disabled = true;
    let url = '/train-model';
    graphDivObj.innerHTML = "<img src='static/TrainMsg.png'/>";
    let formObj = document.getElementById("trainForm")
    let formData = new FormData(formObj);
    postTrainRequest(url, formData)
        .then(txtMsg => displayMsg(txtMsg))
        . catch(error => console.error(error))
}
async function postTrainRequest(url, formData){
    return fetch(url,{
        method: 'POST',
        body: formData
    })
        .then((response) => response.text());
}
function displayMsg(txtMsg){
    graphDivObj.innerHTML = txtMsg;
    makeGraphBtnObj.disabled = false;
}
const graphDivObj = document.getElementById("trainGraphDiv");
const makeGraphBtnObj = document.getElementById("trainGraphBtn");
const trainBtnObj = document.getElementById("trainingBtn");
makeGraphBtnObj.addEventListener("click", trainGraph);
trainBtnObj.addEventListener('click', trainModel);
