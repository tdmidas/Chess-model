from flask import Flask, request, jsonify
import chess
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained model
model = tf.saved_model.load('./AIChess.pb')
predict_fn = model.signatures["serving_default"]


def one_hot_encode_piece(piece):
    pieces = list('rnbqkpRNBQKP.')
    arr = np.zeros(len(pieces))
    piece_to_index = {p: i for i, p in enumerate(pieces)}
    index = piece_to_index[piece]
    arr[index] = 1
    return arr


def encode_board(board):
    board_str = str(board).replace(' ', '')
    board_list = []
    for row in board_str.split('\n'):
        row_list = []
        for piece in row:
            row_list.append(one_hot_encode_piece(piece))
        board_list.append(row_list)
    return np.array(board_list)


def play_nn(fen, show_move_evaluations=False):
    board = chess.Board(fen=fen)
    moves = []
    input_vectors = []
    for move in board.legal_moves:
        candidate_board = board.copy()
        candidate_board.push(move)
        moves.append(move)
        input_vectors.append(encode_board(
            str(candidate_board)).astype(np.float32).flatten())
    input_vectors = np.stack(input_vectors)

    # Create a tensor input for the model
    input_tensor = tf.convert_to_tensor(input_vectors)

    # Use the predict function
    predictions = predict_fn(input_tensor)

    # Print the keys of the predictions dictionary
    print(predictions.keys())

    # Use the correct key from the printed keys
    # Replace 'dense_1' with the correct key
    scores = predictions['dense_1'].numpy()

    if board.turn == chess.BLACK:
        index_of_best_move = np.argmax(scores)
    else:
        index_of_best_move = np.argmax(-scores)

    if show_move_evaluations:
        print(list(zip(moves, scores)))

    best_move = moves[index_of_best_move]
    return str(best_move)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    fen = data.get('fen', None)
    if not fen:
        return jsonify({'error': 'FEN string is required'}), 400
    try:
        best_move = play_nn(fen)
        return jsonify({'best_move': best_move})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
