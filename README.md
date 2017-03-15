
# AI Game-playing Agent for Isolation

I developed an adversarial search agent to play the game "Isolation". My game agent can implement various kinds of searches including iterative deepening, standard minimax, and minimax with alpha-beta pruning. To further improve on the results of my agent, I also developed four unique evaluation heuristics. 

Isolation is a deterministic, two-player game of perfect information in which the players alternate turns moving a single piece from one cell to another on a board.  Whenever either player occupies a cell, that cell becomes blocked for the remainder of the game.  The first player with no remaining legal moves loses, and the opponent is declared the winner.

This project uses a version of Isolation where each agent is restricted to L-shaped movements (like a knight in chess) on a rectangular grid (like a chess or checkerboard).  The agents can move to any open cell on the board that is 2-rows and 1-column or 2-columns and 1-row away from their current position on the board. Movements are blocked at the edges of the board (the board does not wrap around), however, the player can "jump" blocked or occupied spaces (just like a knight in chess).

There is a fixed time limit on each turn for my game agent to search for the best move and respond.  If the time limit expires during a player's turn, that player forfeits the match, and the opponent wins.

These rules are implemented in the `isolation.Board` class provided in the repository. 

### Tournament

The `tournament.py` script measures relative performance of my agent (called "Student") in a round-robin tournament against several other pre-defined agents.  The Student agent uses time-limited Iterative Deepening and custom developed heuristics to hopefully "beat" the other agent. 
