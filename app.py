""" App module """
import os
from threading import Lock

from flask import Flask, render_template
from flask_socketio import SocketIO

from computational_modeling.computational_modeling import Computational_Modeling
from data_visualization.data_visualization import Data_Visualization
from igt_website.backend.igt import IGT

working_dir = os.getcwd()
template_dir = os.path.join(working_dir, r"igt_website\templates")
static_dir = os.path.join(working_dir, r"igt_website\static")

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.config['SECRET_KEY'] = 'secret'
socket_io = SocketIO(app)
lock = Lock()

igt = IGT()
model = Computational_Modeling()
dv = Data_Visualization()

global player_info
global decisions_record
global connection_weights
global patient_id


# player_info = df -> csv
# decisions_record = df -> csv
# connection_weights = df -> csv


@app.route('/')
def index():
    """ Render homepage """
    return render_template('0_home.html')


@app.route('/instructions')
def instructions():
    """ Render instructions page """
    return render_template('1_instructions.html')


@app.route('/game')
def game():
    """ Render game page """
    return render_template('2_game.html')


@socket_io.on('form')
def form(form_data):
    """ Save form data upon reception from web """
    global player_info
    player_info = igt.save_player_info(form_data)
    # patient_id = ...


@socket_io.on('draw_card')
def draw_card(message):
    """ Draw card from deck upon reception of command from web """
    lock.acquire()  # Race conditions
    igt.draw_card(message)
    socket_io.emit('card', igt.last_card)
    lock.release()
    if igt.game_over:
        # socket_io.emit('game_over', patient_id)
        global decisions_record
        decisions_record = igt.data_records
        computational_modeling()


def computational_modeling():
    x, t = model.data_cleaning(decisions_record)
    model.train(x, t, 500, 100)
    global connection_weights
    connection_weights = model.w_xo, model.w_xd, model.w_do
    data_visualization()


def data_visualization():
    dv.create_folder(patient_id)  # -> create dv.folder
    dv.df_to_csv(player_info)  # -> in dv.folder
    dv.df_to_csv(decisions_record)
    dv.df_to_csv(connection_weights)
    dv.create_report()


@socket_io.on('retrieve_values')
def retrieve_values():
    """ Retrieve game state upon reception of command from web """
    socket_io.emit('game_state', igt.print_state())


if __name__ == '__main__':
    socket_io.run(app, debug=True)
