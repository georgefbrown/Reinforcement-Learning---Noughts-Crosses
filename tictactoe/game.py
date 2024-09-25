import random
from learning import RL

board = ['-' for x in range(9)]


def introduction():
    print("Welcome to Geo's tic tac tow game!")

    player = input("Would you like to be X or O?")

    while player != 'X' and player != 'O':
        player = input("Please enter correct letter: X or O")

    if player == 'X':
        robot = 'O'
        print("You are X")
    elif player == 'O':
        robot = 'X'
        print("You are O")

    return player, robot

def draw_board():

    print(board[0] + " | " + board[1] + " | " + board[2])
    print(board[3] + " | " + board[4] + " | " + board[5])
    print(board[6] + " | " + board[7] + " | " + board[8])

def play(player, robot, agent):

    prev_state = None  # Stores the previous state of the board
    prev_action = None  # Stores the last action taken by the RL agent

    while True:

        draw_board()

        #empty spots
        empty_spots = [i for i in range(9) if board[i] == '-']
        move = input("Where would you like to play?")

        # check incorrect input
        if move not in ("1", "2", "3", "4", "5", "6", "7", "8", "9"):
            print("Invalid, please try again")
            draw_board()

        else:

            if int(move) - 1 not in empty_spots:
                print("This is already taken, try again")
                continue

            else:

                # human makes move
                move = int(move) - 1

                #update the board
                board[move] = player

                # recalculate empty spots
                empty_spots = [i for i in range(9) if board[i] == '-'] 

                if '-' not in board:
                    print("Draw!")
                    break

                #check if the human wins
                if check_win(player):
                    # penalize robot for losing
                    agent.Q_learning(prev_state, prev_action, -1, None, [])
                    break

                #robot chooses action based on q table
                state =  tuple(board)
                action = agent.choose_action(state , empty_spots)
                board[action] = robot

                #update states
                prev_state = state
                prev_action = action
                
                #check if the robot agent wings
                if check_win(robot):
                    # prize agent for winning
                    agent.Q_learning(prev_state, prev_action, 1, None, [])
                    break


def random_turn(robot):

    empty_spots = [i for i in range(9) if board[i] == '-'] 

    if empty_spots:
        robot_move = random.choice(empty_spots)
        board[robot_move] = robot
    else:
        print("no more empty spots!")
    return


def check_win(turn):
    win_combinations = [
        [0, 1, 2],  # Top row
        [3, 4, 5],  # Middle row
        [6, 7, 8],  # Bottom row
        [0, 3, 6],  # Left column
        [1, 4, 7],  # Middle column
        [2, 5, 8],  # Right column
        [0, 4, 8],  # Main diagonal
        [2, 4, 6]   # Anti-diagonal
    ]

    for combo in win_combinations:
        # Check if all positions in the current combination have the same symbol as 'turn'
        if board[combo[0]] == board[combo[1]] == board[combo[2]] == turn:
            draw_board()  # Show the winning board
            print(f"{turn} wins!")
            return True  # Indicate the game is over
    
    return False  # No win yet



if __name__ == '__main__':

    player, robot = introduction()
    agent = RL(epsilon=0.1, alpha=0.5, gamma=0.9)
    agent.load_q_table()

    while True:

        play(player, robot, agent)

        #save q_table
        agent.save_q_table()

        #refresh the board
        board = ['-' for x in range(9)]
       
        answer = input("Do you want to play again yes/no")

        if answer == 'no':
            break

