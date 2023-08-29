const socket = io();
let GameOver = false;

function RetrieveGameState() {
    socket.emit('retrieve_values');
}

RetrieveGameState();

socket.addEventListener('game_state', (game_state) => {
    if (game_state['decisions'] > 0) {
        document.getElementById('LeftA').innerHTML = game_state['deck_a'] + " left";
        document.getElementById('LeftB').innerHTML = game_state['deck_b'] + " left";
        document.getElementById('LeftC').innerHTML = game_state['deck_c'] + " left";
        document.getElementById('LeftD').innerHTML = game_state['deck_d'] + " left";
        document.getElementById("DrawsLeft").innerHTML = game_state['decisions'];
        document.getElementById("Score").innerHTML = game_state['score'];}
    else {
        GameOver = false;
    }
})

function TakeCardA() {
    if (GameOver === false) {
        socket.emit('draw_card', 'A');
    }
}

function TakeCardB() {
    if (GameOver === false) {
        socket.emit('draw_card', 'B');
    }
}

function TakeCardC() {
    if (GameOver === false) {
        socket.emit('draw_card', 'C');
    }
}

function TakeCardD() {
    if (GameOver === false) {
        socket.emit('draw_card', 'D');
    }
}

socket.addEventListener('game_over', () => {
    GameOver = true;
    document.getElementById("cards").innerHTML = "<button class=deck id=game_over style=\'font-size: 300%;\'>Game Over - Thank you for playing!</button>";
})


socket.addEventListener('card', (card) => {
    console.log(card[0]);
    if (GameOver == false) {
        if (card[0] === null) {
            document.getElementById("CardValue").innerHTML = "";
        } else {
            document.getElementById("CardValue").innerHTML =
                "<div id=\"deck\">" + card[0] + "</div>"
                + "<div id=\"value\" class=\"container_column\"><span style=\"color: #91C796\">+" + card[1] + "</span><break></break><span style=\"color: #FA5252\">" + card[2] + "</span></div>"
                + "<div id=\"card_foot\"></div>"
        }
    RetrieveGameState();
    }
})