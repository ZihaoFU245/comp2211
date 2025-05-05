import numpy as np
from numba import njit
import math

#------GLOBALS------#
SPLIT_ROW = "+---+---+---+"
SPLIT_COL = "| "
#-------------------#

@njit
def minimax(board, depth, is_maximizing, player_val, bot_val):
    """Apply MiniMax algorithm with numeric board representation."""
    result = check_game(board)
    if result == 0:  # Draw
        return 0
    if result == player_val:  # Player win
        return -1
    if result == bot_val:  # Bot win
        return 1
    if depth == 0:
        return 0
    
    # Game continue case
    if is_maximizing:
        best_score = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:  # Empty cell
                    board[i][j] = bot_val
                    score = minimax(board, depth-1, False, player_val, bot_val)
                    board[i][j] = 0  # Reset
                    best_score = max(best_score, score)
        return best_score
    else:
        best_score = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:  # Empty cell
                    board[i][j] = player_val
                    score = minimax(board, depth-1, True, player_val, bot_val)
                    board[i][j] = 0  # Reset
                    best_score = min(best_score, score)
        return best_score

@njit
def best_move(board, player_val, bot_val):
    best_score = -math.inf
    move = (0, 0)  # Default move
    has_valid_move = False
    
    # If board is empty, pick the center
    is_empty = True
    for i in range(3):
        for j in range(3):
            if board[i][j] != 0:
                is_empty = False
                break
    
    if is_empty:
        return (1, 1), True  # Return the center position
    
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:  # Empty cell
                has_valid_move = True
                board[i][j] = bot_val
                score = minimax(board, 9, False, player_val, bot_val)
                board[i][j] = 0  # Reset
                if score > best_score:
                    best_score = score
                    move = (i, j)
    
    return move, has_valid_move

@njit
def check_game(board):
    """Check if a game is finished using numeric values.
    Returns: 0 for draw, -1 for not finished, player_val or bot_val for winner
    """
    # Check rows, columns and diagonals
    for i in range(3):
        # Check rows
        if board[i, 0] != 0 and board[i, 0] == board[i, 1] == board[i, 2]:
            return board[i, 0]
        # Check columns
        if board[0, i] != 0 and board[0, i] == board[1, i] == board[2, i]:
            return board[0, i]
    
    # Check diagonals
    if board[0, 0] != 0 and board[0, 0] == board[1, 1] == board[2, 2]:
        return board[0, 0]
    if board[0, 2] != 0 and board[0, 2] == board[1, 1] == board[2, 0]:
        return board[0, 2]
    
    # Check for remaining moves
    for i in range(3):
        for j in range(3):
            if board[i, j] == 0:
                return -1  # Not finished
    
    return 0  # Draw

class MiniMaxPlay:
    def __init__(self, depth=10):
        """
        depth: Max Search Depth
        board: Initialize with value of 0 (empty). 1 for player, 2 for AI
        """
        self.depth: int = depth
        self.board = np.zeros((3, 3), dtype=np.int32)
        self._template = np.copy(self.board)
        self.player_val = 1
        self.bot_val = 2
        self.player_symbol = 'o'  # For display
        self.bot_symbol = 'x'     # For display
        self.winner = None

    def _draw_board(self):
        """Plot the game board."""
        print(SPLIT_ROW)
        for i in range(3):
            for j in range(3):
                val = self.board[i][j]
                symbol = ' ' if val == 0 else (self.player_symbol if val == self.player_val else self.bot_symbol)
                print(SPLIT_COL, symbol, end='')
            print(SPLIT_COL)
            print(SPLIT_ROW)

    def _check_game(self, board):
        """Check if a game is finished using the external function."""
        result = check_game(board)
        if result == 0:
            return 'draw'
        elif result == -1:
            return 'not finished'
        elif result == self.player_val:
            return 'player'
        else:
            return 'bot'
    
    def _check_valid_moves(self):
        """Check if there are any valid moves left."""
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    return True
        return False
    
    def _human_turn(self):
        """Player play the move."""
        condition = self._check_game(self.board)
        if condition != "not finished":
            return
            
        # Check if there are valid moves
        if not self._check_valid_moves():
            return
            
        while True:
            try:
                pos = int(input(f"Enter your position for {self.player_symbol} [1 - 9]: "))
                if pos < 1 or pos > 9:
                    print("Invalid input. Enter a number between 1 and 9.")
                    continue
                row = (pos - 1) // 3
                col = (pos - 1) % 3
                if self.board[row][col] != 0:
                    print("Cell already taken. Choose another.")
                    continue
                break
            except ValueError:
                print("Invalid input. Enter a number between 1 and 9.")
        
        self.board[row][col] = self.player_val
        result = self._check_game(self.board)
        if result == 'player':
            self.winner = "Player"
        elif result == 'draw':
            self.winner = None

    def _ai_turn(self):
        """AI predict a move."""
        condition = self._check_game(self.board)
        if condition != "not finished":
            return
        
        # Check if there are valid moves
        if not self._check_valid_moves():
            return
            
        self._template = np.copy(self.board)
        wise_choice, has_valid_move = best_move(self._template, self.player_val, self.bot_val)
        
        if has_valid_move:
            self.board[wise_choice[0]][wise_choice[1]] = self.bot_val
            result = self._check_game(self.board)
            if result == 'bot':
                self.winner = "Bot"
            elif result == 'draw':
                self.winner = None

    def play(self):
        """Start the main loop."""
        p_sym = input("Enter your symbol, 1 char only: ")
        self.player_symbol = p_sym if len(p_sym) == 1 else 'o'
        self.bot_symbol = 'x' if self.player_symbol != 'x' else 'o'

        # === START === #
        if self.player_symbol == 'x':
            while self._check_game(self.board) == 'not finished' and self._check_valid_moves():
                print("=== Before Move ===")
                self._draw_board()
                self._human_turn()
                if self._check_game(self.board) != 'not finished' or not self._check_valid_moves():
                    break
                self._ai_turn()
                print("=== After Move ===")
                self._draw_board()
        else:
            while self._check_game(self.board) == 'not finished' and self._check_valid_moves():
                print("=== Before Move ===")                
                self._ai_turn()
                self._draw_board()
                if self._check_game(self.board) != 'not finished' or not self._check_valid_moves():
                    break
                self._human_turn()
                print("=== After Move ===")
                self._draw_board()   

        print("=== Game Over ===")
        self._draw_board()
        if self.winner:
            print(f"The winner is {self.winner}")
        else:
            print("Draw")

if __name__ == "__main__":
    game = MiniMaxPlay()
    game.play()