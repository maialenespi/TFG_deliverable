const socket = io();

function setCookie() {

    let x = document.forms["form_1"];
    let pathology = x.elements[0].value;
    let age = x.elements[1].value;
    let sex = x.elements[2].value;
    let start_date = x.elements[3].value;
    let start_side = x.elements[4].value;
    let motor_scale_1 = x.elements[5].value;
    let motor_scale_2 = x.elements[6].value;


    if (pathology === "") {
        document.getElementById("avisos").innerHTML = "<p>* Input your pathology</p>";
    } else if (age === "") {
        document.getElementById("avisos").innerHTML = "<p>* Input your age</p>";
    } else if (sex === "") {
        document.getElementById("avisos").innerHTML = "<p>* Input your sex</p>";
    } else if (start_side === "") {
        document.getElementById("avisos").innerHTML = "<p>* Input the pathology start date</p>";
    } else {
        let message = {
            'Pathology': pathology,
            'Age': age,
            'Sex': sex,
            'Start Date': start_date,
            'Affected Side': start_side,
            'Motor Scale 1': motor_scale_1,
            'Motor Scale 2': motor_scale_2
        };
        socket.emit('form', message);
        change_loc();
    }
}

function change_loc() {
    window.location.assign("/instructions");
}