
// Make a move in the Tic Tac Toe game
async function make_move(i, j) {
    let game_over = await eel.eel_is_game_over()();
    if (game_over == true) {
        return;
    }

    eel.eel_make_move(i, j)();
    eel.eel_update_game();
    update_grid();
}


async function init_grid() {
    let board = await eel.eel_get_board()();
    let player = await eel.eel_get_current_player()();
    let rows = board.length;
    let columns = board[0].length;

    let grid = document.getElementById("grid");
    // grid-template-columns: 100px * columns
    grid.style.gridTemplateColumns = "repeat(" + columns + ", 100px)";
    // grid-template-rows: 100px * rows
    grid.style.gridTemplateRows = "repeat(" + rows + ", 100px)";


    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < columns; j++) {
            let cell = document.createElement("button");
            cell.id = i + "-" + j;
            cell.setAttribute("onclick", `make_move(${i}, ${j})`);

            if (board[i][j] == 1) {
                cell.className = "cell square filled";
            } else if (board[i][j] == 2) {
                cell.className = "cell circle filled";
            } else if (board[i][j] == 0) {
                if (player == 1) {
                    cell.className = "cell square";
                } else {
                    cell.className = "cell circle";
                }
            }
            grid.appendChild(cell);
        }
    }
}

// Make the Tic Tac Toe grid
async function update_grid() {
    // 0: empty, 1: X, 2: O
    let board = await eel.eel_get_board()();
    let player = await eel.eel_get_current_player()();
    let is_human = await eel.eel_is_current_player_human()();
    let rows = board.length;
    let columns = board[0].length;

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < columns; j++) {
            let cell = document.getElementById(i + "-" + j);
            if (board[i][j] == 1) {
                cell.className = "cell square filled";
            } else if (board[i][j] == 2) {
                cell.className = "cell circle filled";
            } else if (board[i][j] == 0) {
                if (player == 1) {
                    cell.className = "cell square";
                } else {
                    cell.className = "cell circle";
                }
            }
        }
    }
    
    if (player == 2) {
        document.getElementById("player").className = "o";
        if (is_human) {
            document.getElementById("player").innerHTML = "O's turn";
        } else {
            document.getElementById("player").innerHTML = "AI is thinking...";
            document.getElementById("player").className = "o thinking";
        }
    }
    else {
        document.getElementById("player").className = "x";
        if (is_human) {
            document.getElementById("player").innerHTML = "X's turn";
        } else {
            document.getElementById("player").innerHTML = "AI is thinking...";
            document.getElementById("player").className = "x thinking";
        }
    }
}

async function update_game() {
    let game_over = await eel.eel_is_game_over()();
    
    if (game_over == true) {
        let winner = await eel.eel_get_winner()();
        if (winner == null) {
            document.getElementById("player").innerHTML = "It's a draw!";
            document.getElementById("player").className = "x";
            setTimeout(update_game, 1000);
            return;
        }
        else if (winner == 1) {
            winner = "X";
        }
        else if (winner == 2) {
            winner = "O";
        }
        document.getElementById("player").innerHTML = winner + " wins!";
        document.getElementById("player").className = winner.toLowerCase();
        setTimeout(update_game, 1000);
        return;
    }
    await update_grid();
    let current_player = await eel.eel_is_current_player_human()();
    if (current_player == false) {
        await eel.eel_update_game();
    }
    await update_grid();
    // Recursive call
    setTimeout(update_game, 10);
}


async function main() {
    init_grid();
    update_game();
}



// Exit the python process when the browser window is closed
// window.addEventListener("beforeunload", function () {
//     eel.eel_stop();
// });


main();

// Functions for the buttons
async function reset() {
    eel.eel_reset_game();
    update_grid();
}

async function undo() {
    let is_human = await eel.eel_is_current_player_human()();
    if (is_human == false) {
        return;
    }
    eel.eel_undo();
    update_grid();
}


function quit() {
    eel.eel_stop();
    window.close();
}