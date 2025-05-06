import numpy as np

EMPTY = 0

def evaluate_position(board, player):
    """
    An evaluation function that assigns a score to the board state
    by adding the possible score for all directions.
    Returns the score.
    """
    ########### TODO: Your code here ###########
    # NOTE: player is int type
    # NOTE: make the checking a matrix/array and slide it over the board

    def slide(arr, window_shape):
        """
        Slides a window of shape window_shape over arr starting from bottom-left,
        moving left→right, then up one row, until all windows are covered.
        Returns a list of sub-arrays.
        """
        wins = []
        h, w = window_shape
        max_row = arr.shape[0] - h
        max_col = arr.shape[1] - w
        for i in range(max_row, -1, -1):      # bottom up
            for j in range(0, max_col+1):    # left → right
                wins.append(arr[i:i+h, j:j+w])
        return wins
    
    def create_slider(player: int, num_empty: int) -> list[np.ndarray]:
        sliders = []

        def add_orients(pat1d):
            # horizontal
            sliders.append(pat1d.reshape(1,4).copy())
            # vertical
            sliders.append(pat1d.reshape(4,1).copy())
            # \ diagonal in 4×4
            m = np.zeros((4,4), int)
            np.fill_diagonal(m, pat1d)
            sliders.append(m.copy())
            # / diagonal
            sliders.append(np.fliplr(m).copy())
                    
            # 0 empties: all-player
        if num_empty == 0:
            base = np.full(4, player, int)
            add_orients(base)
            return sliders

        # 1,2,3 empties: choose positions by nested loops
        if num_empty == 1:
            for e in range(4):
                p = np.array([player]*4, int)
                p[e] = EMPTY
                add_orients(p)

        elif num_empty == 2:
            for e1 in range(4):
                for e2 in range(e1+1, 4):
                    p = np.array([player]*4, int)
                    p[e1] = EMPTY; p[e2] = EMPTY
                    add_orients(p)

        elif num_empty == 3:
            for pi in range(4):
                p = np.array([EMPTY]*4, int)
                p[pi] = player
                add_orients(p)

        else:
            raise ValueError("num_empty must be 0,1,2 or 3")

        return sliders
            
    SCORE = {0:0, 1:1, 2:10, 3:100, 4:1000000}
    opponent = 3 - player
    total = 0

    for p, sign in [(player, +1), (opponent, -1)]:
        for empties in range(0,4):
            val = SCORE[4-empties]
            if val == 0:
                continue
            for slider in create_slider(p, empties):
                for window in slide(board, slider.shape):
                    if np.array_equal(window, slider):
                        total += sign * val

    return total

test_arr = np.array(
    [[0,0,0,0,0,0],
     [0,0,0,0,0,0],
     [0,0,1,1,0,0],
     [0,1,1,1,1,0]]
)

score = evaluate_position(test_arr, 2)

print(score)